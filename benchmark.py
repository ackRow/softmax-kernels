from typing import Callable
import torch
import triton
import matplotlib.pyplot as plt

import cuda_softmax_kernel
from triton_kernels.article1 import triton_fused_softmax
from triton_kernels.article2 import triton_softmax_v2
from triton_kernels.article3 import triton_online_softmax, triton_online_softmax_hybrid

configs = [
    triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2 ** i for i in range(10, 21)],
            line_arg="provider",
            line_vals=[
                # "torch",
                # "triton_tutorial",
                # "cuda_multi_block_v1",
                "cuda_multi_block_v2",
                "triton_multi_block",
                "cuda_online_v1",
                "cuda_online_v2",
                "triton_online",
                "triton_online_hybrid"
            ],
            line_names=[
                # "Torch",
                # "Triton v1 (tutorial)",
                # "Cuda multi-block v1 (mine)",
                "Cuda multi-block v2 (mine)",
                "Triton multi-block (mine)",
                "Cuda online v1 (mine)",
                "Cuda online v2 (mine)",
                "Triton online (mine)",
                "Triton online hybrid (Liger Kernel)"
            ],
            ylabel="TFLOPS",
            plot_name="softmax-performance",
            args={},
        )
]


def _gbps(x: torch.Tensor, *times_ms: float) -> tuple[float, ...]:
    """Convert one or more timings in ms to GB/s."""
    bytes_moved = 2 * x.numel() * x.element_size()  # read + write
    return tuple(bytes_moved * 1e-9 / (ms * 1e-3) for ms in times_ms)


@triton.testing.perf_report(configs)
def benchmark(N: int, provider: str, quantiles: list[float] = [0.5, 0.2, 0.8]):
    x = torch.randn((128, N), device="cuda")

    def run_provider() -> Callable:
        if provider == "torch":
            return lambda: torch.softmax(x, dim=-1)
        elif provider == "triton_tutorial":
            return lambda: triton_fused_softmax(x)
        elif provider == "cuda_v1":
            return lambda: cuda_softmax_kernel.softmax_cuda_v1(x)
        elif provider == "cuda_multi_block_v1":
            return lambda: cuda_softmax_kernel.softmax_cuda_multi_block_v1(x)
        elif provider == "cuda_multi_block_v2":
            return lambda: cuda_softmax_kernel.softmax_cuda_multi_block_v2(x)
        elif provider == "triton_multi_block":
            return lambda: triton_softmax_v2(x)
        elif provider == "cuda_online_v1":
            return lambda: cuda_softmax_kernel.softmax_cuda_online_v1(x)
        elif provider == "cuda_online_v2":
            return lambda: cuda_softmax_kernel.softmax_cuda_online_v2(x)
        elif provider == "triton_online":
            return lambda: triton_online_softmax(x)
        elif provider == "triton_online_hybrid":
            return lambda: triton_online_softmax_hybrid(x)
        else:
            raise KeyError(f"Unknown provider {provider!r}.")
    
    try:
        ms, min_ms, max_ms = triton.testing.do_bench(run_provider(), quantiles=quantiles)
        return _gbps(x, ms, max_ms, min_ms)
    except (triton.CompilationError, ValueError):
        return 0.0, 0.0, 0.0


df = benchmark.run(return_df=True)[0]
print(df)

# Plot the results
fig = plt.figure(figsize=(10, 6))
for column in df.columns[1:]:
    plt.plot(df['N'], df[column], label=column, marker='o')

plt.xscale('log', base=2)
plt.xlabel('Tensor width')
plt.ylabel('GB/s')
plt.title('Softmax Implementation Performance')
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tight_layout()

fig.savefig("images/softmax_performances.png")
