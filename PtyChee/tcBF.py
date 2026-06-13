"""
Author: Zeyu Wang
Date: September 2025
Description: "tilt corrected BF"
"""

import numpy as np
import time
from .utils import print_and_log, log_to_file


def tcBF(data_4D,
        alpha,
        scan_step,
        scan_rotation_angle,
        scan_flip, 
        aperture_radius,
        center_dx, 
        center_dy,
        defocus,
        inner_radius = 0, 
        outer_radius = 1,
        BF_padding = 30,
):
    print_and_log('')
    print_and_log(f'#################### tcBF Parameters #####################')
    print_and_log(f'vSTEM shift rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'vSTEM shift flip x: {scan_flip}')
    print_and_log(f'inner_radiuss: {inner_radius} Alpha')
    print_and_log(f'outer_radius: {outer_radius} Alpha')

    sy, sx, dy, dx = data_4D.shape
    theta_r = np.pi*scan_rotation_angle/180
    data_4D_p = np.transpose(data_4D, (2, 3, 0, 1))
    sxx, syy = np.meshgrid(np.arange(sx+BF_padding*2)-sx/2-BF_padding, np.arange(sy+BF_padding*2)-sy/2-BF_padding)

    start_time = time.time()
    print_and_log('')
    print_and_log(f'##################### tcBF Process #######################')
    print("\rtcBF progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0,"*"*0,"."*10,0,0,0),end = "")
    Aligned_BF = np.zeros((sy,sx))
    counter = 1
    total_area = (outer_radius**2-inner_radius**2)*aperture_radius**2*np.pi
    for jj in range(dy):
        time_i0 = time.time()
        for ii in range(dx):
            yj = jj+1-center_dy
            xi = ii+1-center_dx
            rr = np.sqrt(yj**2+xi**2)

            if inner_radius*aperture_radius < rr < outer_radius*aperture_radius:
                BF_ji = data_4D_p[jj][ii]
                BF_ji_pad = np.pad(BF_ji, ((BF_padding, BF_padding),(BF_padding, BF_padding)), 'constant')

                y_shift = yj/aperture_radius*alpha*defocus/scan_step
                x_shift = xi/aperture_radius*alpha*defocus/scan_step

                shift = x_shift+1j*y_shift
                shift_rotate = np.abs(shift)*np.exp(1j*(np.angle(shift)-theta_r))
                x_shift = np.real(shift_rotate)
                y_shift = np.imag(shift_rotate)

                if scan_flip:
                    x_shift *= -1

                BF_ji_pad_fft = np.fft.fftshift(np.fft.fft2(BF_ji_pad))
                BF_ji_pad_fft *= np.exp(-2j*np.pi*(sxx*x_shift/sx+syy*y_shift/sy))
                BF_ji_pad_moved = np.fft.ifft2(np.fft.ifftshift(BF_ji_pad_fft))

                BF_ji_moved = np.real(BF_ji_pad_moved[BF_padding:-BF_padding,BF_padding:-BF_padding])

                Aligned_BF += BF_ji_moved

                counter += 1

                if counter > total_area:
                    counter = total_area

                time_is = time.time()-time_i0
                if time_is == 0:
                    time_is = 1e-6
                speed = 1/time_is
                process = (counter/total_area*100)
                aa = "*" * int(process/10)
                bb = "." * (10-int(process/10))
                dur = int(time.time() - start_time)
                time_remain = int((total_area-counter)*time_is)
                dur_h = dur//3600
                dur_m = (dur-dur_h*3600)//60
                dur_s = dur-dur_h*3600-dur_m*60
                time_remain_h = time_remain//3600
                time_remain_m = (time_remain-time_remain_h*3600)//60
                time_remain_s = time_remain-time_remain_h*3600-time_remain_m*60
                print("\rtcBF progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s),end = "")
    print('')
    log_to_file("tcBF progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s))

    total_time = time.time() - start_time
    print_and_log(f'tcBF process finished in {total_time} s')

    shift_max = int(np.abs(outer_radius*alpha*defocus/scan_step+1))
    intensity_min = np.min(Aligned_BF[shift_max:-shift_max,shift_max:-shift_max])

    return Aligned_BF, intensity_min

