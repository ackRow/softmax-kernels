import torch
import triton
import matplotlib.pyplot as plt

from utils import MAX_FUSED_SIZE

from .triton_online_softmax import triton_online_softmax, triton_online_softmax_v2
from article_2.triton_softmax_kernel import triton_softmax_v2
from article_1.triton_tutorial import triton_fused_softmax

configs = [
    triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2 ** i for i in range(10, 22)],
            line_arg="provider",
            line_vals=[
                "triton_fused_tutorial",
                "triton_multi_block",
                "triton_online_v1",
                "triton_online_v2",
            ],
            line_names=[
                "Triton Tutorial",
                "Triton multi-block (mine)",
                "Triton online (mine)",
                "Triton online multi-block (mine)",
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
    try:
        if provider == 'triton_fused_tutorial':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_fused_softmax(x), quantiles=quantiles)
        elif provider == 'triton_multi_block':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_softmax_v2(x), quantiles=quantiles)
        elif provider == 'triton_online_v1':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_online_softmax(x), quantiles=quantiles)
        elif provider == 'triton_online_v2':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_online_softmax_v2(x), quantiles=quantiles)
        else:
            raise ValueError(f"Provider not found: {provider}")
        perf = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)
    except triton.CompilationError:
        return 0, 0, 0
    
    


df = benchmark.run(return_df=True)[0]
print(df)

# Plot the results
fig = plt.figure(figsize=(10, 6))
for column in df.columns[1:]:
    plt.plot(df['N'], df[column], label=column, marker='o')

plt.axvline(x = MAX_FUSED_SIZE, color = 'b', linestyle='--', label = 'Start multi-block')
plt.axvline(x = 2**20, color = 'r', linestyle='--', label = 'Max blocksize')
plt.xscale('log', base=2)
plt.xlabel('Tensor width')
plt.ylabel('GB/s')
plt.title('Softmax Implementation Performance')
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tight_layout()

fig.savefig("article_3/images/softmax_performances.png")
