import triton
import triton.language as tl
import torch

from .article1 import kernel_fused_softmax_baseline
from .utils import calculate_settings, compute_row_ptrs, get_row


@triton.jit
def bitcast_unmerge(merged):
    tl.static_assert(merged.dtype == tl.int64)
    b = (merged & 0xFFFFFFFF).to(tl.int32).to(tl.float32, bitcast=True)
    a = (merged >> 32).to(tl.int32).to(tl.float32, bitcast=True)  # shifted by 32 bits
    return a, b


@triton.jit
def bitcast_merge(a, b):
    tl.static_assert(a.dtype == tl.float32)
    tl.static_assert(b.dtype == tl.float32)
    a = a.to(dtype=tl.int32, bitcast=True).to(tl.int64)  # directly converted to int32
    a = a << 32  # shifted by 32 bits
    b = b.to(dtype=tl.int32, bitcast=True).to(tl.int64)  # directly converted to int32
    return a | b

@triton.jit
def online_softmax_divisor_update(a, b):
    max_a, div_a = bitcast_unmerge(a)
    max_b, div_b = bitcast_unmerge(b)

    max_out = tl.maximum(max_a, max_b)

    div_out = div_a * tl.exp(max_a - max_out) + div_b * tl.exp(max_b - max_out)
    return bitcast_merge(max_out, div_out)


@triton.jit
def kernel_online_softmax_merge(
    x_ptr, x_row_stride: int,
    out_ptr, out_row_stride: int,
    n_rows: int, n_cols: tl.constexpr,
    block_size: tl.constexpr,
):
    # might handle multiple rows if the number of rows is high
    pid = tl.program_id(0)  # starting row for the given program
    row_step = tl.num_programs(0)  # rows to skip before the next one to process
    col_offsets = tl.arange(0, block_size)

    init_max = float("-inf")
    init_divisor = 1.0
    init_divisors = tl.full((block_size,), init_divisor, dtype=tl.float32)

    for row_idx in tl.range(pid, n_rows, step=row_step, num_stages=2):
        
        current_state = bitcast_merge(init_max, init_divisor)
        for col_idx in tl.range(0, n_cols, step=block_size, loop_unroll_factor=0):
            row = get_row(x_ptr, row_idx, x_row_stride, col_idx, col_offsets, n_cols, other=init_max)
            row_states = bitcast_merge(row, init_divisors)
            incoming_state  = tl.reduce(row_states, 0, online_softmax_divisor_update)
            current_state = online_softmax_divisor_update(current_state, incoming_state)
        
        maximum, divisor = bitcast_unmerge(current_state)

        for col_idx in tl.range(0, n_cols, step=block_size, loop_unroll_factor=0):
            row = get_row(x_ptr, row_idx, x_row_stride, col_idx, col_offsets, n_cols, other=init_max)
            result = tl.exp(row - maximum) / divisor

            out_row_ptrs = compute_row_ptrs(out_ptr + col_idx, out_row_stride, row_idx, col_offsets)
            tl.store(out_row_ptrs, result, mask=(col_idx + col_offsets) < n_cols, cache_modifier=".cs")
    

def triton_online_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    assert x.is_cuda and x.ndim == 2 and x.is_contiguous()

    n_rows, n_cols = x.shape
    block_size, num_warps = calculate_settings(n_cols)

    kernel_online_softmax_merge[(n_rows,)](
        x, x.stride(0),
        out, out.stride(0),
        n_rows, n_cols=n_cols,
        block_size=block_size,
        num_warps=num_warps,
    )
    
    return out


@triton.jit
def kernel_online_softmax_hybrid(
    x_ptr, x_row_stride: int,
    out_ptr, out_row_stride: int,
    n_rows: int, n_cols: int,
    block_size: tl.constexpr,
):
    """Inspired by [Liger kernels](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/softmax.py)"""
    
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, block_size)

    init_max = float("-inf")

    for row_idx in tl.range(pid, n_rows, step=row_step, num_stages=2):
        
        maximum = init_max
        divisor = 1.0
        for col_idx in tl.range(0, n_cols, step=block_size):
            row = get_row(x_ptr, row_idx, x_row_stride, col_idx, col_offsets, n_cols, other=init_max)
            row_max = max(row_max, tl.max(row, axis=0))
            new_max = tl.maximum(maximum, row_max)

            divisor = divisor * tl.exp(maximum - new_max) + tl.sum(tl.exp(row - new_max), axis=0)
            maximum = new_max

        for col_idx in tl.range(0, n_cols, step=block_size):
            row = get_row(x_ptr, row_idx, x_row_stride, col_idx, col_offsets, n_cols, other=init_max)
            result = tl.exp(row - maximum) / divisor

            out_row_ptrs = compute_row_ptrs(out_ptr + col_idx, out_row_stride, row_idx, col_offsets)
            tl.store(out_row_ptrs, result, mask=(col_idx + col_offsets) < n_cols, cache_modifier=".cs")
    

def triton_online_softmax_hybrid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    assert x.is_cuda and x.ndim == 2 and x.is_contiguous()

    n_rows, n_cols = x.shape
    block_size, num_warps = calculate_settings(n_cols)

    if n_cols <= block_size:
        kernel_fused_softmax_baseline[(n_rows,)](
            x, x.stride(0),
            out, out.stride(0),
            n_rows, n_cols,
            block_size=block_size,
            num_warps=num_warps
        )
    else:
        kernel_online_softmax_merge[(n_rows,)](
            x, x.stride(0),
            out, out.stride(0),
            n_rows, n_cols=n_cols,
            block_size=block_size,
            num_warps=num_warps,
        )
    
    return out