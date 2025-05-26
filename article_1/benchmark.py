import torch
import triton
import matplotlib.pyplot as plt

import cuda_softmax_kernel
from triton_tutorial import triton_fused_softmax

configs = [
    triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2 ** i for i in range(8, 13)],
            line_arg="provider",
            line_vals=[
                "torch",
                "triton",
                "cuda_v1",
            ],
            line_names=[
                "Torch",
                "Triton v1 (tutorial)",
                "Cuda v1 (mine)",
            ],
            ylabel="TFLOPS",
            plot_name="softmax-performance",
            args={},
        )
]


@triton.testing.perf_report(configs)
def benchmark(N, provider):
    x = torch.randn((128, N), device="cuda")
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_fused_softmax(x), quantiles=quantiles)
    if provider == 'cuda_v1':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: cuda_softmax_kernel.softmax_cuda_v1(x), quantiles=quantiles)
    
    perf = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


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

fig.savefig("softmax_performances.png")
