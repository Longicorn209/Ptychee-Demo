"""
Author: Zeyu Wang
Date: September 2025
Description: "4D-STEM data preprocessing tools"
"""

import numpy as np
from scipy.ndimage import zoom
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

    print_and_log(f'real space bin {bin_factor} finished in {time.time()-time_i0} s')
    return data_4D_binned, scan_step



def recipSpace_bin(data_4D, bin_factor):
    time_i0 = time.time()
    sy, sx, dy, dx = data_4D.shape
    assert dy % bin_factor == 0 and dx % bin_factor == 0

    data_4D_binned = data_4D.reshape(sy, sx,
                                    dy//bin_factor, bin_factor,
                                    dx//bin_factor, bin_factor,
                                    ).sum(axis=(3,5))
    
    print_and_log(f'reciprocal space bin {bin_factor} finished in {time.time()-time_i0} s')
    return data_4D_binned



def recipSpace_pad(data_4D, pad_pixels):
    time_i0 = time.time()
    data_4D_padded  = np.pad(data_4D, ((0,0),(0,0),(pad_pixels,pad_pixels),(pad_pixels,pad_pixels)), 'constant')

    print_and_log(f'reciprocal space pad {pad_pixels} finished in {time.time()-time_i0} s')
    return data_4D_padded



def realSpace_interpolation(data_4D, scan_step, interp_factor, interp_order=2):
    time_i0 = time.time()
    scan_step /= interp_factor
    zoom_factors = (interp_factor, interp_factor, 1, 1)
    data_4D_interp = zoom(data_4D, zoom_factors, order=interp_order)

    print_and_log(f'real space interpolate {interp_factor} with order {interp_order} finished in {time.time()-time_i0} s')
    return data_4D_interp, scan_step



def recipSpace_interpolation(data_4D, interp_factor, order=3):
    time_i0 = time.time()
    zoom_factors = (1, 1, interp_factor, interp_factor)
    data_4D_interp = zoom(data_4D, zoom_factors, order=order)

    print_and_log(f'reciprocal space interpolate {interp_factor} with order {order} finished in {time.time()-time_i0} s')
    return data_4D_interp
