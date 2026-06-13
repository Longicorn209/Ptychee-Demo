"""
Author: Zeyu Wang
Email: zywang209@gmail.com
Date: September 2025
Description: "CoM and iCoM"
"""

import numpy as np

from .utils import print_and_log



def calculate_CoM(COM, scan_rotation_angle, scan_flip):

    print_and_log('')
    print_and_log(f'##################### CoM Parameters #####################')

    scan_rotation_angle *= -1
    theta_r = np.pi*scan_rotation_angle/180

    print_and_log(f'CoM rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'CoM flip x: {scan_flip}')

    amp_COM = np.abs(COM)
    phs_COM = np.angle(COM)
    phs_COM_rotate = phs_COM + theta_r

    COM_rotate = amp_COM*np.exp(1j*phs_COM_rotate)

    if scan_flip:
        COM_rotate = - np.real(COM_rotate) + 1j*np.imag(COM_rotate)

    return COM_rotate




def iter_iCoM(g, lr=0.1, n_iter=50):

    print_and_log('')
    print_and_log(f'############### Iterative iCoM Parameters ################')
    print_and_log(f'learning rate: {lr}')
    print_and_log(f'Number of Iterations: {n_iter}')

    gx = np.real(g)
    gy = np.imag(g)
    
    sy, sx = g.shape
    iCoM = np.zeros((sy, sx))

    loss = np.zeros(n_iter)
    
    for it in range(n_iter):
        iCoM_x = np.zeros_like(iCoM)
        iCoM_y = np.zeros_like(iCoM)
        
        iCoM_x[:, :-1] = iCoM[:, 1:] - iCoM[:, :-1]
        iCoM_y[:-1, :] = iCoM[1:, :] - iCoM[:-1, :]
        
        res_x = gx - iCoM_x
        res_y = gy- iCoM_y
        
        grad = np.zeros_like(iCoM)
        grad[:, :-1] += res_x[:, :-1]
        grad[:, 1:]  -= res_x[:, :-1]
        grad[:-1, :] += res_y[:-1, :]
        grad[1:, :]  -= res_y[:-1, :]
        
        iCoM -= lr * grad
        
        loss[it] = np.sum(res_x**2 + res_y**2)
            
    
    return iCoM, loss



def jacobi_iCoM(g, n_iter=50):

    print_and_log('')
    print_and_log(f'################# Jacobi iCoM Parameters #################')
    print_and_log(f'Number of Iterations: {n_iter}')

    gx = np.real(g)
    gy = np.imag(g)
    
    sy, sx = g.shape
    iCoM = np.zeros((sy, sx))
    iCoM_new = np.zeros_like(iCoM)

    loss = np.zeros(n_iter)

    for it in range(n_iter):
        iCoM_new[1:-1, 1:-1] = 0.25 * (
            iCoM[1:-1, 2:] + iCoM[1:-1, :-2] +
            iCoM[2:, 1:-1] + iCoM[:-2, 1:-1] -
            (gx[1:-1, 1:-1] - gx[1:-1, :-2]) -
            (gy[1:-1, 1:-1] - gy[:-2, 1:-1])
        )

        iCoM_x = np.zeros_like(iCoM)
        iCoM_y = np.zeros_like(iCoM)
        iCoM_x[:, :-1] = iCoM[:, 1:] - iCoM[:, :-1]
        iCoM_y[:-1, :] = iCoM[1:, :] - iCoM[:-1, :]

        res_x = gx - iCoM_x
        res_y = gy - iCoM_y
        loss[it] = np.sum(res_x**2 + res_y**2)

        iCoM[:, :] = iCoM_new

    return iCoM, loss




def anal_iCOM(g, epsilon_iCoM = 1e-6, HPF_pixel = 0):

    print_and_log('')
    print_and_log(f'############### Analytical iCoM Parameters ###############')
    print_and_log(f'iCoM epsilon: {epsilon_iCoM}')
    print_and_log(f'High Pass Filter radius: {HPF_pixel} pixels')

    sy, sx = g.shape

    gx_rotate_flip = np.real(g)
    gy_rotate_flip = np.imag(g)

    (ny, nx) = gx_rotate_flip.shape
    iky = np.fft.fftfreq(ny)
    ikx = np.fft.fftfreq(nx)
    grid_iky, grid_ikx = np.meshgrid(iky, ikx, indexing='ij')
    k = grid_ikx ** 2 + grid_iky ** 2
    k[k < epsilon_iCoM] = epsilon_iCoM
    That = (np.fft.fft2(gx_rotate_flip) * grid_ikx + np.fft.fft2(gy_rotate_flip) * grid_iky) / (2j * np.pi * k)
    iCOM = np.real(np.fft.ifft2(That))
    #iCOM = np.abs(np.fft.ifft2(That))
    #iCOM -= iCOM.min()

    xx,yy = np.meshgrid(np.arange(sx),np.arange(sy))
    XX = xx - int(sx/2)
    YY = yy - int(sy/2)
    rr = np.sqrt(XX**2+YY**2)
    fft_mask  = 1 - np.ones((sy, sx)) * (rr < HPF_pixel)

    iCoM_fft = np.fft.fftshift(np.fft.fft2(iCOM))
    iCoM_fft *= fft_mask
    iCOM_HPF = np.real(np.fft.ifft2(np.fft.ifftshift(iCoM_fft)))

    return iCOM_HPF



def calculate_dCoM_from_CoM(CoM, dx=1.0, dy=1.0):

    print_and_log('')
    print_and_log(f'#################### dCoM Parameters #####################')
    print_and_log(f'calculating dCoM from: CoM')

    dcom = np.zeros(CoM.shape)

    dcom[:, :-1] += np.real(CoM[:, 1:] - CoM[:, :-1]) * -1 / dx
    dcom[:-1, :] += np.imag(CoM[1:, :] - CoM[:-1, :]) * -1 / dy

    return dcom



def calculate_dCoM_from_iCoM(iCoM, dx=1.0, dy=1.0):

    print_and_log('')
    print_and_log(f'#################### dCoM Parameters #####################')
    print_and_log(f'calculating dCoM from: iCoM')

    dCoM = (
        (iCoM[1:-1, 2:] - 2*iCoM[1:-1, 1:-1] + iCoM[1:-1, :-2]) / dx**2
      + (iCoM[2:, 1:-1] - 2*iCoM[1:-1, 1:-1] + iCoM[:-2, 1:-1]) / dy**2
    ) * -1
    return dCoM