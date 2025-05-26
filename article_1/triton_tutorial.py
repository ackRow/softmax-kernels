import triton
import triton.language as tl
import torch
from triton.runtime import driver


properties = driver.active.utils.get_device_properties(0)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]


@triton.jit
def compute_row_ptrs(tensor_ptr, row_stride, row_index, col_offsets):
    # base pointer of the row
    row_ptr = tensor_ptr + row_index * row_stride
    # assume col_stride is 1
    return row_ptr + col_offsets

@triton.jit
def kernel_fused_softmax(
    x_ptr, x_row_stride: int,
    out_ptr, out_row_stride: int,
    n_rows: int, n_cols: int,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)  # starting row for the given program
    row_step = tl.num_programs(0)  # rows to skip before the next one to process
    
    # handle multiple rows if the number of rows is above grid size
    for row_idx in tl.range(pid, n_rows, step=row_step, num_stages=4):
        col_offsets = tl.arange(0, block_size)  # block_size is always greater than n_cols
        mask = col_offsets < n_cols # mask for boundary check

        x_row_ptrs = compute_row_ptrs(x_ptr, x_row_stride, row_idx, col_offsets)
        row = tl.load(x_row_ptrs, mask=mask, other=float("-inf"))
        
        stable_row = row - tl.max(row, axis=0)  # substract for stability
        numerator = tl.exp(stable_row)
        denominator = tl.sum(numerator, axis=0)
        res = numerator / denominator
        
        out_row_ptrs = compute_row_ptrs(out_ptr, out_row_stride, row_idx, col_offsets)
        tl.store(out_row_ptrs, res, mask=mask)


def triton_fused_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    assert x.is_cuda and x.ndim == 2 and x.is_contiguous()

    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    
    num_warps = 4
    if block_size >= 1024:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = kernel_fused_softmax.warmup(
        x, x.stride(0),
        out, out.stride(0),
        n_rows, n_cols,
        block_size=block_size,
        num_warps=num_warps, grid=(1, ))
    kernel._init_handles()

    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    # Theorical maximum of launchable triton program
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](
        x, x.stride(0), out, out.stride(0), n_rows, n_cols
    )
    
    return out