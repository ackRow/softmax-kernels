import triton
import triton.language as tl
import torch

from utils import calculate_settings


@triton.jit
def compute_row_ptrs(tensor_ptr, row_stride, row_index, col_offsets):
    # base of the row
    row_ptr = tensor_ptr + row_index * row_stride
    # assume col_stride is 1
    return row_ptr + col_offsets

@triton.jit
def _get_row(tensor_ptr, row_index, row_stride, col_index, col_offsets, n_cols, other):
    row_ptr = tensor_ptr + row_index * row_stride
    mask = (col_index + col_offsets) < n_cols
    return tl.load(row_ptr + col_index + col_offsets, mask=mask, other=other, cache_modifier=".ca")


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
def kernel_online_softmax(
    x_ptr, x_row_stride: int,
    out_ptr, out_row_stride: int,
    n_rows: int, n_cols: int,
    block_size: tl.constexpr,
):
    # might handle multiple rows if the number of rows is high
    pid = tl.program_id(0)  # starting row for the given program
    row_step = tl.num_programs(0)  # rows to skip before the next one to process
    
    for row_idx in tl.range(pid, n_rows, step=row_step, num_stages=2):
        col_offsets = tl.arange(0, block_size)  # block_size is always greater than n_cols
        mask = col_offsets < n_cols # mask for boundary check

        x_row_ptrs = compute_row_ptrs(x_ptr, x_row_stride, row_idx, col_offsets)
        row = tl.load(x_row_ptrs, mask=mask, other=float("-inf"))
        initial_div = tl.full((block_size,), 1.0, dtype=tl.float32)
        
        vf = bitcast_merge(row, initial_div)
        z  = tl.reduce(vf, 0, online_softmax_divisor_update)
        max, divisor = bitcast_unmerge(z)

        res = tl.exp(row - max)/divisor
        out_row_ptrs = compute_row_ptrs(out_ptr, out_row_stride, row_idx, col_offsets)
        tl.store(out_row_ptrs, res, mask=mask)
    

def triton_online_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    assert x.is_cuda and x.ndim == 2 and x.is_contiguous()

    n_rows, n_cols = x.shape
    _, num_warps = calculate_settings(n_cols)
    block_size = triton.next_power_of_2(n_cols)

    kernel_online_softmax[(n_rows,)](
        x, x.stride(0),
        out, out.stride(0),
        n_rows, n_cols,
        block_size=block_size,
        num_warps=num_warps,
    )
    
    return out

@triton.jit
def kernel_online_softmax_v2(
    x_ptr, x_row_stride: int,
    out_ptr, out_row_stride: int,
    n_rows: int, n_cols: tl.constexpr,
    block_size: tl.constexpr,
):
    # might handle multiple rows if the number of rows is high
    pid = tl.program_id(0)  # starting row for the given program
    row_step = tl.num_programs(0)  # rows to skip before the next one to process
    col_offsets = tl.arange(0, block_size)
    initial_divisors = tl.full((block_size,), 1.0, dtype=tl.float32)

    for row_idx in tl.range(pid, n_rows, step=row_step, num_stages=2):
        
        divisor = 1.0
        max = float("-inf")
        for col_idx in tl.range(0, n_cols, step=block_size, loop_unroll_factor=0):
            row = _get_row(x_ptr, row_idx, x_row_stride, col_idx, col_offsets, n_cols, other=float("-inf"))
        
            vf = bitcast_merge(row, initial_divisors)
            z  = tl.reduce(vf, 0, online_softmax_divisor_update)
            incoming_max, incoming_divisor = bitcast_unmerge(z)
            new_max = tl.maximum(max, incoming_max)
            divisor = divisor * tl.exp(max - new_max) + incoming_divisor * tl.exp(incoming_max - new_max)
            max = new_max

        for col_idx in tl.range(0, n_cols, step=block_size, loop_unroll_factor=0):
            row = _get_row(x_ptr, row_idx, x_row_stride, col_idx, col_offsets, n_cols, other=float("-inf"))
            result = tl.exp(row - max) / divisor

            out_row_ptrs = compute_row_ptrs(out_ptr + col_idx, out_row_stride, row_idx, col_offsets)
            tl.store(out_row_ptrs, result, mask=(col_idx + col_offsets) < n_cols, cache_modifier=".cs")
    

def triton_online_softmax_v2(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    assert x.is_cuda and x.ndim == 2 and x.is_contiguous()

    n_rows, n_cols = x.shape
    block_size, num_warps = calculate_settings(n_cols)

    kernel_online_softmax_v2[(n_rows,)](
        x, x.stride(0),
        out, out.stride(0),
        n_rows, n_cols=n_cols,
        block_size=block_size,
        num_warps=num_warps,
    )
    
    return out