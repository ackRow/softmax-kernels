import triton
import triton
import triton.language as tl

MAX_FUSED_SIZE = int(2**15)

def calculate_settings(n) -> tuple[int, int]:
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    BLOCK_SIZE = triton.next_power_of_2(min(n, MAX_FUSED_SIZE))

    num_warps = 4
    if BLOCK_SIZE >= 2**15:
        num_warps = 32
    elif BLOCK_SIZE >= 2**13:
        num_warps = 16
    elif BLOCK_SIZE >= 2**11:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.jit
def compute_row_ptrs(tensor_ptr, row_stride, row_index, col_offsets):
    # base of the row
    row_ptr = tensor_ptr + row_index * row_stride
    # assume col_stride is 1
    return row_ptr + col_offsets

@triton.jit
def get_row(tensor_ptr, row_index, row_stride, col_index, col_offsets, n_cols, other):
    row_ptr = tensor_ptr + row_index * row_stride
    mask = (col_index + col_offsets) < n_cols
    return tl.load(row_ptr + col_index + col_offsets, mask=mask, other=other)