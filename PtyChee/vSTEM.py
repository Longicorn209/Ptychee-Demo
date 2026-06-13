"""
Author: Zeyu Wang
Date: September 2025
Description: "virtual STEM"
"""

import numpy as np
import time

from .utils import print_and_log, log_to_file, annular_mask



def vSTEM(
        data_4D,
        center_dy,
        center_dx,
        aperture_radius,
        radius_list,
):
    print_and_log('')
    print_and_log(f'#################### vSTEM Parameters ####################')
    print_and_log(f'Radius Range List:')
    for radius_range in radius_list:
        print_and_log(f'({radius_range[0]} - {radius_range[1]}) Alpha')

    sy, sx, dy, dx = data_4D.shape

    n_pic = len(radius_list)

    vSTEM_siries = np.zeros((n_pic, sy, sx))

    start_time = time.time()
    print_and_log('')
    print_and_log(f'##################### vSTEM Process ######################')
    print("\rvSTEM progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0,"*"*0,"."*10,0,0,0),end = "")
    for i in range(n_pic):
        time_i0 = time.time()

        inner_r = radius_list[i][0]
        outer_r = radius_list[i][1]

        mask = annular_mask(dy, dx, inner_r*aperture_radius, outer_r*aperture_radius, center_dy, center_dx)

        masked_4D = data_4D*mask

        vSTEM_siries[i] = masked_4D.sum(axis=(2,3))

        time_is = time.time()-time_i0
        speed = 1/time_is
        process = (i+1)/n_pic*100
        aa = "*" * int(process/10)
        bb = "." * (10-int(process/10))
        dur = int(time.time() - start_time)
        time_remain = int((n_pic-i-1)*time_is)
        dur_h = dur//3600
        dur_m = (dur-dur_h*3600)//60
        dur_s = dur-dur_h*3600-dur_m*60
        time_remain_h = time_remain//3600
        time_remain_m = (time_remain-time_remain_h*3600)//60
        time_remain_s = time_remain-time_remain_h*3600-time_remain_m*60
        print("\rvSTEM progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s),end = "")
    print('')
    total_time = time.time() - start_time

    log_to_file("vSTEM progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s))
    print_and_log(f'vSTEM process finished in {total_time} s')

    return vSTEM_siries
