"""
Author: Zeyu Wang
Email: zywang209@gmail.com
Date: September 2025
Description: "analytical ptychogrphay"
"""

import numpy as np
import time
import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from .utils import print_and_log, log_to_file, Chi_defocus, annular_mask, r_map



def p_fft(data_4D):
    time_i0 = time.time()

    data_4D = torch.from_numpy(data_4D)#.to(device)
    data_4D = fftshift((fft2(data_4D,dim=(0,1))),dim=(0,1))

    time_ie = time.time()-time_i0
    print_and_log(f'fft along p finished in {time_ie} s')

    return np.array(data_4D.cpu())



def rP_ifft(data_4D):
    time_i0 = time.time()

    data_4D = torch.from_numpy(data_4D)#.to(device)
    data_4D = ifft2((ifftshift(data_4D,dim=(2,3))),dim=(2,3))

    time_ie = time.time()-time_i0
    print_and_log(f'ifft along rPrime finished in {time_ie} s')

    return np.array(data_4D.cpu())



def r_fft(data_4D):
    time_i0 = time.time()

    data_4D = torch.from_numpy(data_4D)#.to(device)
    data_4D = fftshift((fft2(data_4D,dim=(2,3))),dim=(2,3))

    time_ie = time.time()-time_i0
    print_and_log(f'fft along r finished in {time_ie} s')

    return np.array(data_4D.cpu())



def calculate_pPrime_calibration(scan_step, data_shape):
    sy, sx, dy, dx = data_shape

    p_pixel_size = scan_step
    p_size_sy = p_pixel_size*sy
    p_size_sx = p_pixel_size*sx
    pPrime_pixel_size_dy = 1/p_size_sy
    pPrime_pixel_size_dx = 1/p_size_sx

    return pPrime_pixel_size_dy, pPrime_pixel_size_dx



def ApeFunc(yc, xc, dy, dx, aperture_radius, reciprocalSpace_pixel_size, Voltage, defocus):
    mask = annular_mask(dy, dx, 0, aperture_radius, yc, xc)
    Chi, *_ = Chi_defocus(dy, dx, reciprocalSpace_pixel_size, Voltage, defocus, yc, xc)

    return mask * np.exp(-1j*Chi)



def WDD_engine(data_4D, 
               pPrime_pixel_size_dy, 
               pPrime_pixel_size_dx, 
               center_dy, 
               center_dx, 
               scan_rotation_angle, 
               scan_flip, 
               aperture_radius, 
               reciprocalSpace_pixel_size, 
               Voltage, 
               defocus,
               epsilon_WDD = 0.1,
               ):
    print_and_log('')
    print_and_log('##################### WDD Parameters ######################')

    sy, sx, dy, dx = data_4D.shape
    data_4D[data_4D<0] = 0

    if not scan_flip:
        scan_rotation_angle *= -1
    theta_r = np.pi*scan_rotation_angle/180

    print_and_log(f'Trotter Pattern rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'Trotter Pattern flip x: {scan_flip}')
    print_and_log(f'WDD epsilon: {epsilon_WDD}')

    if sy%2==0:
        center_sy = int(sy/2)
    else:
        center_sy = int((sy-1)/2)
    if sx%2==0:
        center_sx = int(sx/2)
    else:
        center_sx = int((sx-1)/2)

    print_and_log('')
    print_and_log('####################### WDD Process #######################')

    data_4D_fft01 = p_fft(data_4D)
    del data_4D

    data_4D_fft01_ifft23 = rP_ifft(data_4D_fft01)
    del data_4D_fft01

    ApeFunc0 = ApeFunc(center_dy, center_dx, dy, dx, aperture_radius, reciprocalSpace_pixel_size, Voltage, defocus)
    ApeFunc_WDD = np.zeros((sy, sx, dy, dx), dtype=np.complex64)
    ApeFunc_WDD_0 = np.fft.ifft2(np.fft.ifftshift(ApeFunc0*np.conj(ApeFunc0)))

    start_time = time.time()
    print("\rApeFunc_WDD progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0,"*"*0,"."*10,0,0,0),end = "")
    for sj in range(sy):
        time_i0 = time.time()
        for si in range(sx):
            py = sj - center_sy
            px = si - center_sx
            ry = py*pPrime_pixel_size_dy/reciprocalSpace_pixel_size
            rx = px*pPrime_pixel_size_dx/reciprocalSpace_pixel_size
            rx, ry = np.dot(np.array([[np.cos(theta_r),
                                    np.sin(theta_r)],
                                    [-np.sin(theta_r),
                                    np.cos(theta_r)]]),
                                    np.array([rx, ry]))
            if scan_flip:
                rx *= -1

            ApeFunc_move = ApeFunc(center_dy-ry, center_dx-rx, dy, dx, aperture_radius, reciprocalSpace_pixel_size, Voltage, defocus)
            ApeFunc_WDD[sj][si] = np.fft.ifft2(np.fft.ifftshift(np.conj(ApeFunc_move)*ApeFunc0))

        time_is = time.time()-time_i0
        speed = 1/time_is
        process = (sj/sy*100)
        aa = "*" * int(process/10)
        bb = "." * (10-int(process/10))
        dur = int(time.time() - start_time)
        time_remain = int((sy-sj-1)*time_is)
        dur_h = dur//3600
        dur_m = (dur-dur_h*3600)//60
        dur_s = dur-dur_h*3600-dur_m*60
        time_remain_h = time_remain//3600
        time_remain_m = (time_remain-time_remain_h*3600)//60
        time_remain_s = time_remain-time_remain_h*3600-time_remain_m*60
        print("\rApeFunc_WDD progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s),end = "")
    print('')
    log_to_file("ApeFunc_WDD progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s))

    total_time = time.time() - start_time
    print_and_log(f'ApeFunc_WDD process finished in {total_time} s')


    epsilon = np.sum(np.abs(ApeFunc_WDD_0)**2)*epsilon_WDD
    obj_WDD = np.conj(ApeFunc_WDD)*data_4D_fft01_ifft23/(ApeFunc_WDD*np.conj(ApeFunc_WDD)+epsilon)
    del data_4D_fft01_ifft23, ApeFunc_WDD

    obj_WDD_fft23 = r_fft(obj_WDD)
    del obj_WDD


    D00 = np.sqrt(obj_WDD_fft23[center_sy][center_sx][int(center_dy)][int(center_dx)])
    obj_fft_minus_rho = np.conj(obj_WDD_fft23[:,:,int(center_dy),int(center_dx)])/D00

    
    obj_fft = np.flip(obj_fft_minus_rho)
    objFunc = np.flip(np.fft.ifft2(np.fft.ifftshift(obj_fft_minus_rho)))

    return obj_fft, objFunc



def SSB_engine(data_4D, 
               pPrime_pixel_size_dy, 
               pPrime_pixel_size_dx, 
               center_dy, 
               center_dx, 
               scan_rotation_angle, 
               scan_flip, 
               aperture_radius, 
               reciprocalSpace_pixel_size, 
               ):
    print_and_log('')
    print_and_log('###################### SSB Parameters #####################')

    sy, sx, dy, dx = data_4D.shape
    data_4D[data_4D<0] = 0

    if not scan_flip:
        scan_rotation_angle *= -1
    theta_r = np.pi*scan_rotation_angle/180

    print_and_log(f'Trotter Pattern rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'Trotter Pattern flip x: {scan_flip}')

    if sy%2==0:
        center_sy = int(sy/2)
    else:
        center_sy = int((sy-1)/2)
    if sx%2==0:
        center_sx = int(sx/2)
    else:
        center_sx = int((sx-1)/2)

    BFdisk = annular_mask(dy, dx,  0, aperture_radius, center_dy, center_dx)
    obj_fft = np.zeros((sy, sx), dtype=np.complex64)

    print_and_log('')
    print_and_log('####################### SSB Process #######################')
    
    data_4D_fft01 = p_fft(data_4D)
    del data_4D

    start_time = time.time()
    print("\rSSB progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0,"*"*0,"."*10,0,0,0),end = "")
    for sj in range(sy):
        time_i0 = time.time()
        for si in range(sx):
            py = sj - center_sy
            px = si - center_sx
            ry = py*pPrime_pixel_size_dy/reciprocalSpace_pixel_size
            rx = px*pPrime_pixel_size_dx/reciprocalSpace_pixel_size
            rx, ry = np.dot(np.array([[np.cos(theta_r),
                                    np.sin(theta_r)],
                                    [-np.sin(theta_r),
                                    np.cos(theta_r)]]),
                                    np.array([rx, ry]))
            if scan_flip:
                rx *= -1

            mask_p = BFdisk.copy()
            mask_n = BFdisk.copy()

            if sj!=center_sy or si != center_sx:
                mask_p *= (r_map(dy,dx,center_dy-ry,center_dx-rx,)<aperture_radius)
                mask_p *= (r_map(dy,dx,center_dy+ry,center_dx+rx)>=aperture_radius)
                mask_n *= (r_map(dy,dx,center_dy+ry,center_dx+rx)<aperture_radius)
                mask_n *= (r_map(dy,dx,center_dy-ry,center_dx-rx,)>=aperture_radius)

                mask_used = (mask_p-mask_n)

                mask_sum = np.sum(np.abs(mask_used))
                if mask_sum == 0:
                    mask_sum = 1

                obj_fft[sj][si] = np.sum(data_4D_fft01[sj][si]*mask_used)

            else:
                obj_fft[sj][si] = np.sum(data_4D_fft01[sj][si])

        time_is = time.time()-time_i0
        speed = 1/time_is
        process = (sj/sy*100)
        aa = "*" * int(process/10)
        bb = "." * (10-int(process/10))
        dur = int(time.time() - start_time)
        time_remain = int((sy-sj-1)*time_is)
        dur_h = dur//3600
        dur_m = (dur-dur_h*3600)//60
        dur_s = dur-dur_h*3600-dur_m*60
        time_remain_h = time_remain//3600
        time_remain_m = (time_remain-time_remain_h*3600)//60
        time_remain_s = time_remain-time_remain_h*3600-time_remain_m*60
        print("\rSSB progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s),end = "")
    print('')
    log_to_file("SSB progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s))

    total_time = time.time() - start_time
    print_and_log(f'SSB process finished in {total_time} s')

    obj_fft *= 1j
    objFunc = np.fft.ifft2(np.fft.ifftshift(obj_fft))

    return obj_fft, objFunc
