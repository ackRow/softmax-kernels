import triton
import triton.language as tl
import torch

from .utils import calculate_settings, compute_row_ptrs, get_row


@triton.jit
def kernel_softmax_adapt_blocksize(
    x_ptr, x_row_stride: int,
    out_ptr, out_row_stride: int,
    n_rows: int, n_cols: tl.constexpr,
    block_size: tl.constexpr,
):
    # might handle multiple rows if the number of rows is high
    pid = tl.program_id(0)  # starting row for the given program
    row_step = tl.num_programs(0)  # rows to skip before the next one to process
    col_offsets = tl.arange(0, block_size)  # chunk the row in block_size (more memory read)

    init_max = float("-inf")
    
    for row_idx in tl.range(pid, n_rows, step=row_step, num_stages=2):
        row_max = init_max
        for col_idx in tl.range(0, n_cols, step=block_size, loop_unroll_factor=0):
            row = get_row(x_ptr, row_idx, x_row_stride, col_idx, col_offsets, n_cols, other=init_max)
            row_max = max(row_max, tl.max(row, axis=0))

        exp_sum = 0.
        for col_idx in tl.range(0, n_cols, step=block_size, loop_unroll_factor=0):
            row = get_row(x_ptr, row_idx, x_row_stride, col_idx, col_offsets, n_cols, other=init_max)
            exp_sum += tl.sum(tl.exp(row - row_max), axis=0)
        
        for col_idx in tl.range(0, n_cols, step=block_size, loop_unroll_factor=0):
            row = get_row(x_ptr, row_idx, x_row_stride, col_idx, col_offsets, n_cols, other=init_max)
            result = tl.exp(row - row_max) / exp_sum

            out_row_ptrs = compute_row_ptrs(out_ptr + col_idx, out_row_stride, row_idx, col_offsets)
            tl.store(out_row_ptrs, result, mask=(col_idx + col_offsets) < n_cols)


def triton_softmax_v2(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    assert x.is_cuda and x.ndim == 2 and x.is_contiguous()

    n_rows, n_cols = x.shape
    block_size, num_warps = calculate_settings(n_cols)

    kernel = kernel_softmax_adapt_blocksize[(n_rows,)](
        x, x.stride(0),
        out, out.stride(0),
        n_rows, n_cols,
        block_size=block_size,
        num_warps=num_warps)
 
    return out