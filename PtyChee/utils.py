"""
Author: Zeyu Wang
Email: zywang209@gmail.com
Date: September 2025
Description: "PtyChee utilities"
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from datetime import datetime



_LOG_DIR = None
def log_to_file(message, filename="log.txt"):
    global _LOG_DIR
    if _LOG_DIR is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _LOG_DIR = os.path.join("results", timestamp)
        os.makedirs(_LOG_DIR, exist_ok=True)

    filepath = os.path.join(_LOG_DIR, filename)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")
    return _LOG_DIR



def print_and_log(message):
    print(message)
    _LOG_DIR = log_to_file(message)
    return _LOG_DIR



def FFT_Log(objectFunc):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(objectFunc))+1e-5))



def Power_Spectrum_Log(objectFunc):
    obj_fft = np.fft.fftshift(np.fft.fft2(objectFunc))
    return np.log(np.abs(obj_fft*np.conj(obj_fft)))



def get_subplot_grid(n):
    row = int(n**0.5)
    col = (n + row - 1) // row 

    if col < row:
        row, col = col, row
    return row, col



def RGB_Complex_Plot(a):
    H = ((np.angle(a)+np.pi) % (2*np.pi)) / (2*np.pi)
    S = np.ones_like(H)
    V = np.abs(a) / np.max(np.abs(a))

    HSV = np.stack((H, S, V), axis=-1)
    RGB = hsv_to_rgb(HSV)

    plt.imshow(RGB)
    plt.xticks([])
    plt.yticks([])



def calculate_wavelength(Voltage):
    emass = 510.99906
    hc = 12.3984244
    return hc/np.sqrt(Voltage * (2*emass + Voltage))



def r_map(y, x, yc, xc):
    xx, yy = np.meshgrid(np.arange(x)-xc, np.arange(y)-yc)
    return np.sqrt(xx**2+yy**2)



def annular_mask(y, x, inner_r, outer_r, yc=None, xc=None):
    if yc == None:
        yc = y/2
    if xc == None:
        xc = x/2
    rr = r_map(y, x, yc, xc)
    return  np.ones((y, x)) * (rr >= inner_r) * (rr < outer_r)



def Chi_defocus(dy, dx, reciprocalSpace_pixel_size, Voltage, defocus, yc=None, xc=None, Cs=0):
    if yc == None:
        yc = dy/2
    if xc == None:
        xc = dx/2

    wavelength = calculate_wavelength(Voltage)

    rr = r_map(dy, dx, yc, xc)
    k_theta = rr*reciprocalSpace_pixel_size*wavelength
    Chi = 2*np.pi/wavelength*(0.5*defocus*k_theta**2+0.25*Cs*k_theta**4)

    return Chi, k_theta



def PACBED_identify(data_4D, BF_threshold=0.5, forced_aperture_radius = None):
    
    print_and_log('')
    print_and_log(f'#################### Data Information ####################')
    print_and_log(f'data shape: {data_4D.shape}')
    print_and_log(f'data type: {data_4D.dtype}')
    print_and_log(f'CBED intensity range: {np.min(data_4D)} – {np.max(data_4D)}')
    average_intensity = np.mean(np.sum(data_4D, axis=(2,3), dtype=np.float32))
    print_and_log(f'average CBED intensity: {average_intensity}')

    paCBED = np.mean(data_4D,(0,1))
    paCBED[paCBED<0] = 0
    aperture_radius = forced_aperture_radius
    CBED_norm = (paCBED - np.amin(paCBED)) / np.ptp(paCBED)
    BFdisk = np.ones(CBED_norm.shape) * (CBED_norm > BF_threshold)
    BFedge = (np.sum(np.abs(np.gradient(BFdisk)), axis=0)) > BF_threshold
    xx,yy = np.meshgrid(np.arange(0,paCBED.shape[1]),np.arange(0,paCBED.shape[0]))
    center_x = np.sum(BFdisk*xx/np.sum(BFdisk))
    center_y = np.sum(BFdisk*yy/np.sum(BFdisk))
    
    if forced_aperture_radius is None:
        aperture_radius = np.average(np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)[BFedge])
    else:
        aperture_radius = forced_aperture_radius

    print_and_log(f'BF center_Y: {center_y}')
    print_and_log(f'BF center_X: {center_x}')
    print_and_log(f'BF threshold: {BF_threshold}')
    print_and_log(f'Aperture Radius: {aperture_radius} pixels')

    return paCBED, aperture_radius, center_x, center_y



def calculate_recipro_pixeSize(Voltage, alpha, aperture_radius):

    wavelength = calculate_wavelength(Voltage)

    reciprocalSpace_pixel_size = alpha/wavelength/aperture_radius

    print_and_log('')
    print_and_log(f'################ Calculatied Calibration #################')
    print_and_log(f'reciprocal space pixel size: {reciprocalSpace_pixel_size} 1/Å/pixel')

    return reciprocalSpace_pixel_size



def calcualte_CoM_origin(data_4D, aperture_radius, center_dx, center_dy,
                  inner_radius, 
                  outer_radius, 
                  ):
    
    print_and_log('')
    print_and_log(f'################## Original CoM Process ##################')
    print_and_log(f'inner_radiuss: {inner_radius} Alpha')
    print_and_log(f'outer_radius: {outer_radius} Alpha')


    paCBED = np.mean(data_4D,(0,1))
    dy, dx = paCBED.shape

    Annular_ROI = annular_mask(dy, dx, aperture_radius*inner_radius, aperture_radius*outer_radius, center_dy, center_dx)
    masked_4D = data_4D * Annular_ROI

    X, Y = np.meshgrid(np.arange(dx)-center_dx, np.arange(dy)-center_dy)
    gx = np.average(masked_4D * X, axis=(2, 3))
    gy = np.average(masked_4D * Y, axis=(2, 3))

    return gx + 1j*gy