import torch

import triton
import triton.language as tl


@triton.jit
def awq_dequantize_kernel(qweight_ptr,   # quantized matrix
                          scales_ptr,    # scales, per group
                          zeros_ptr,     # zeros, per group
                          split_k_iters, # Not used
                          thx,           # Not used
                          thy,           # Not used
                          group_size,    # Should always be 128
                          result_ptr,    # Output matrix
                          num_cols,      # input num cols in qweight
                          num_rows,      # input num rows in qweight
                          BLOCK_SIZE_X: tl.constexpr,
                          BLOCK_SIZE_Y: tl.constexpr):
    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)

    # print(f"pid_y = {pid_y}, pid_x = {pid_x}")
    # print(f"BLOCK_SIZE_Y = {BLOCK_SIZE_Y}, BLOCK_SIZE_X = {BLOCK_SIZE_X}")

    # qweight offsets for qweight_ptr
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols  * offsets_y[:, None] + offsets_x[None, :]

    # print(f"offsets_y = {offsets_y}")
    # print(f"offsets_x = {offsets_x}")

    # Scale offsets for scales_ptr
    scale_offsets_y = (pid_y * BLOCK_SIZE_Y 
                      + tl.arange(0, BLOCK_SIZE_Y) //group_size)
    scale_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X) * 8
    scale_offsets = (num_cols * scale_offsets_y[:, None] +
                     scale_offsets_x[None,:])

    # Zero offsets for scales_ptr
    zero_offsets_y = (pid_y * BLOCK_SIZE_Y 
                      + tl.arange(0, BLOCK_SIZE_Y) //group_size)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    # Output offsets for result_ptr
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X) * 8
    result_offsets = (num_cols * result_offsets_y[:, None] +
            result_offsets_x[None, :])

    # print(f"offsets = {offsets}")

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols 

    # print(f"masks_y = {masks_y}")
    # print(f"masks_x = {masks_x}")
    masks = masks_y[:, None] & masks_x[None, :]

    iweights = tl.load(qweight_ptr + offsets, masks)
    zeros = tl.load(zeros_ptr + zero_offsets, masks)

    # There are 8 values packed per int, loop over them and
    # do block-wise computations w.r.t the order and write
    # the results out to result_ptr w.r.t. the reverse order.
    for i in range(8):
        shift = i

        # Use reverse_awq_order to write result in reverse_awq_order.
        reverse_awq_order = 0
        if i == 0:
            reverse_awq_order = 0
        elif i == 1:
            reverse_awq_order = 4
        elif i == 2:
            reverse_awq_order = 1 
        elif i == 3:
            reverse_awq_order = 5
        elif i == 4:
            reverse_awq_order = 2
        elif i == 5:
            reverse_awq_order = 6
        elif i == 6:
            reverse_awq_order = 3
        elif i == 7:
            reverse_awq_order = 7

        # Use awq_order to load scales in awq_order.
        awq_order = 0
        if i == 0:
            awq_order = 0
        elif i == 1:
            awq_order = 2 
        elif i == 2:
            awq_order = 4 
        elif i == 3:
            awq_order = 6
        elif i == 4:
            awq_order = 1
        elif i == 5:
            awq_order = 3
        elif i == 6:
            awq_order = 5
        elif i == 7:
            awq_order = 7

        # Load the scales in AWQ order so that the equation:
        #  (iweights_shift - zeros_shift) * scales
        # computes the correct values.
        scales = tl.load(scales_ptr + scale_offsets + awq_order, masks)

        # Shift and extract the packed value, but its still in AWQ order.
        iweights_shifted = ((iweights >> shift) & 0xF)
        zeros_shifted = ((zeros >> shift) & 0xF)

        # Compute the dequantized results and write them in reverse
        # AWQ order.
        tl.store(result_ptr + result_offsets + reverse_awq_order,
                 (iweights_shifted - zeros_shifted) * scales,
                 masks)

# Example input: 
#   qweight.size=torch.Size([3584, 576]),
#   qweight.dtype = torch.int32,
#   scales.size=torch.Size([28, 4608]),
#   scales.dtype=torch.float16,
#   zeros.size=torch.Size([28, 576]),
#   zeros.dtype=torch.int32
#   split_k_iters=0
#   thx=0
#   thy=0
def awq_dequantize_triton(qweight: torch.Tensor,
                         scales: torch.Tensor,
                         zeros: torch.Tensor,
                         split_k_iters: int,
                         thx: int,
                         thy: int) -> torch.Tensor:
    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(qweight.shape[0],
                         qweight.shape[1] * 8,
                         device = qweight.device,
                         dtype = torch.float16)

    Y = qweight.shape[0] # num rows
    X = qweight.shape[1] # num cols
    group_size = 128
    grid = lambda META: (
        triton.cdiv(X, META['BLOCK_SIZE_X']), triton.cdiv(Y,
                    META['BLOCK_SIZE_Y']), )
    awq_dequantize_kernel[grid](qweight, scales, zeros, split_k_iters, 
            thx, thy, group_size, result, X, Y,
            BLOCK_SIZE_X = 32, BLOCK_SIZE_Y = 64)

    return result

def main():
    qweight_rows = 3584
    qweight_cols = 576
    group_size = 128
    small_test_size = True
    if small_test_size:
        qweight_rows = 256
        qweight_cols = 128
    print(f"qweight_rows = {qweight_rows}, qweight_cols = {qweight_cols}")
    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = qweight_cols * 8
    scales_dtype = torch.float16
    zeros_rows = scales_rows
    zeros_cols = qweight_cols
    zeros_dtype = torch.int32
    split_k_iters=0
    thx=0
    thy=0
    device='cuda'
    qweight = torch.randint(0, 1, (qweight_rows,
                         qweight_cols),
                         dtype=qweight_dtype,
                         device=device)
    scales = torch.zeros(scales_rows,
                        scales_cols,
                        dtype=scales_dtype,
                        device=device)
    zeros = torch.zeros(zeros_rows,
                       zeros_cols,
                       dtype=zeros_dtype,
                       device=device)
    print(f"qweight = {qweight}")
    awq_dequantize_triton(qweight, scales, zeros, split_k_iters, thx, thy)

if __name__ == '__main__':
    main()
