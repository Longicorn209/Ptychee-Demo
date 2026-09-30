"""
Author: Zeyu Wang
Date: September 2025
Description: "4D-STEM data preprocessing tools"
"""

import numpy as np
import torch
import time

from .utils import print_and_log


def realSpace_bin(data_4D, scan_step, bin_factor):
    time_i0 = time.time()

    sy, sx, dy, dx = data_4D.shape
    assert sy % bin_factor == 0 and sx % bin_factor == 0

    data_4D_binned = data_4D.reshape(sy//bin_factor, bin_factor,
                                    sx//bin_factor, bin_factor,
                                    dy, dx).sum(axis=(1,3))
    scan_step *= bin_factor
    new_sy, new_sx, dy, dx = data_4D_binned.shape

    print_and_log(f'real space bin {bin_factor} finished in {time.time()-time_i0} s')
    print_and_log(f'data shape from {(sy, sx, dy, dx)} to {(new_sy, new_sx, dy, dx)}') 
    return data_4D_binned, scan_step



def recipSpace_bin(data_4D, bin_factor):
    time_i0 = time.time()

    sy, sx, dy, dx = data_4D.shape
    assert dy % bin_factor == 0 and dx % bin_factor == 0

    data_4D_binned = data_4D.reshape(sy, sx,
                                    dy//bin_factor, bin_factor,
                                    dx//bin_factor, bin_factor,
                                    ).sum(axis=(3,5))

    sy, sx, new_dy, new_dx = data_4D_binned.shape
    
    print_and_log(f'reciprocal space bin {bin_factor} finished in {time.time()-time_i0} s')
    print_and_log(f'data shape from {(sy, sx, dy, dx)} to {(sy, sx, new_dy, new_dx)}') 
    return data_4D_binned



def recipSpace_pad(data_4D, pad_pixels):
    time_i0 = time.time()

    sy, sx, dy, dx = data_4D.shape 
    data_4D_padded  = np.pad(data_4D, ((0,0),(0,0),(pad_pixels,pad_pixels),(pad_pixels,pad_pixels)), 'constant')
    sy, sx, new_dy, new_dx = data_4D_padded.shape

    print_and_log(f'reciprocal space pad {pad_pixels} pixels finished in {time.time()-time_i0} s')
    print_and_log(f'data shape from {(sy, sx, dy, dx)} to {(sy, sx, new_dy, new_dx)}') 
    return data_4D_padded



def realSpace_interpolation(data_4D, scan_step, interp_factor, block_size=32):
    time_i0 = time.time()

    data_4D = torch.as_tensor(data_4D, device='cpu')
    sy, sx, dy, dx = data_4D.shape
    new_sy = int(round(sy * interp_factor))
    new_sx = int(round(sx * interp_factor))

    data_4D_interp = torch.empty((new_sy, new_sx, dy, dx), dtype=torch.float32)

    scale = (new_sy * new_sx) / (sy * sx)

    for r0 in range(0, dy, block_size):
        r1 = min(r0 + block_size, dy)

        for c0 in range(0, dx, block_size):
            c1 = min(c0 + block_size, dx)

            block = data_4D[:, :, r0:r1, c0:c1].to(torch.float32)

            F = torch.fft.fftshift(torch.fft.fft2(block, dim=(0, 1)), dim=(0, 1))

            if interp_factor >= 1:
                F_interp = torch.zeros((new_sy, new_sx, r1-r0, c1-c0), dtype=F.dtype)

                y0 = (new_sy - sy) // 2
                x0 = (new_sx - sx) // 2

                F_interp[y0:y0+sy, x0:x0+sx] = F

            else:
                y0 = (sy - new_sy) // 2
                x0 = (sx - new_sx) // 2

                F_interp = F[y0:y0+new_sy, x0:x0+new_sx]

            block_interp = torch.fft.ifft2(
                torch.fft.ifftshift(F_interp, dim=(0, 1)),
                dim=(0, 1)
            ).real

            block_interp *= scale
            data_4D_interp[:, :, r0:r1, c0:c1] = block_interp

    if not data_4D.dtype.is_floating_point:
        info = torch.iinfo(data_4D.dtype)
        data_4D_interp = data_4D_interp.round().clamp(info.min, info.max).to(data_4D.dtype)
    else:
        data_4D_interp = data_4D_interp.to(data_4D.dtype)

    scan_step /= interp_factor

    print_and_log(f'real space interpolate {interp_factor}x finished in {time.time()-time_i0} s')
    print_and_log(f'data shape from {(sy, sx, dy, dx)} to {(new_sy, new_sx, dy, dx)}')
    return data_4D_interp.numpy(), scan_step



def recipSpace_interpolation(data_4D, interp_factor, block_size=32):
    time_i0 = time.time()

    data_4D = torch.as_tensor(data_4D, device='cpu')
    sy, sx, dy, dx = data_4D.shape
    new_dy = int(round(dy * interp_factor))
    new_dx = int(round(dx * interp_factor))

    data_4D_interp = torch.empty((sy, sx, new_dy, new_dx), dtype=torch.float32)

    scale = (new_dy * new_dx) / (dy * dx)

    for y0_block in range(0, sy, block_size):
        y1_block = min(y0_block + block_size, sy)

        for x0_block in range(0, sx, block_size):
            x1_block = min(x0_block + block_size, sx)

            block = data_4D[y0_block:y1_block, x0_block:x1_block, :, :].to(torch.float32)

            F = torch.fft.fftshift(torch.fft.fft2(block, dim=(2, 3)), dim=(2, 3))

            if interp_factor >= 1:
                F_interp = torch.zeros(
                    (y1_block-y0_block, x1_block-x0_block, new_dy, new_dx),
                    dtype=F.dtype
                )

                y0 = (new_dy - dy) // 2
                x0 = (new_dx - dx) // 2

                F_interp[:, :, y0:y0+dy, x0:x0+dx] = F

            else:
                y0 = (dy - new_dy) // 2
                x0 = (dx - new_dx) // 2

                F_interp = F[:, :, y0:y0+new_dy, x0:x0+new_dx]

            block_interp = torch.fft.ifft2(
                torch.fft.ifftshift(F_interp, dim=(2, 3)),
                dim=(2, 3)
            ).real

            block_interp *= scale
            data_4D_interp[y0_block:y1_block, x0_block:x1_block, :, :] = block_interp

    if not data_4D.dtype.is_floating_point:
        info = torch.iinfo(data_4D.dtype)
        data_4D_interp = data_4D_interp.round().clamp(info.min, info.max).to(data_4D.dtype)
    else:
        data_4D_interp = data_4D_interp.to(data_4D.dtype)

    print_and_log(f'reciprocal space interpolate {interp_factor}x finished in {time.time()-time_i0} s')
    print_and_log(f'data shape from {(sy, sx, dy, dx)} to {(sy, sx, new_dy, new_dx)}')
    return data_4D_interp.numpy()