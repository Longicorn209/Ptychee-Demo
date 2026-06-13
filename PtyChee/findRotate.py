"""
Author: Zeyu Wang
Email: zywang209@gmail.com
Date: September 2025
Description: "find best rotate angle and flip"
"""

import numpy as np
#import matplotlib.pyplot as plt

from .utils import print_and_log


def curl_curve(
        g, 
        test_angle_range = np.arange(-89.0, 90.0, 1.0), 
        dx = 1.0, 
        dy = 1.0
):
    """
    g: complex ndarray (ny, nx)
    test_angle_range: array of angles (degrees)
    dx, dy: grid spacing
    returns:
        angles (n_angles,)
        mean_curl_no_flip (n_angles,)
        mean_curl_flipx (n_angles,)
    """

    print_and_log('')
    print_and_log(f'################# Calculating Curl Curve #################')

    amp = np.abs(g)              # (ny, nx)
    phs = np.angle(g)            # (ny, nx)
    theta_r = np.deg2rad(test_angle_range)[:, None, None]  # (n_angles, 1, 1)

    g_rot = amp[None, :, :] * np.exp(1j * (phs[None, :, :] + theta_r))

    Fx = np.real(g_rot)  # (n_angles, ny, nx)
    Fy = np.imag(g_rot)  # (n_angles, ny, nx)

    dFy_dx = np.gradient(Fy, dx, axis=-1)
    dFx_dy = np.gradient(Fx, dy, axis=-2)
    curl_z = dFy_dx - dFx_dy

    mean_curl = np.mean(np.abs(curl_z), axis=(-2, -1))

    Fx_flip = -Fx
    Fy_flip = Fy
    dFy_dx_flip = np.gradient(Fy_flip, dx, axis=-1)
    dFx_dy_flip = np.gradient(Fx_flip, dy, axis=-2)
    curl_z_flip = dFy_dx_flip - dFx_dy_flip

    mean_curl_flip = np.mean(np.abs(curl_z_flip), axis=(-2, -1))

    ind_min = np.argmin(mean_curl).item()
    ind_flip_min = np.argmin(mean_curl_flip).item()

    if mean_curl[ind_min] <= mean_curl_flip[ind_flip_min]:
        bast_angle_1 = - test_angle_range[ind_min]
        if bast_angle_1 <= 0:
            bast_angle_2 = bast_angle_1 + 180
        else:
            bast_angle_2 = bast_angle_1 - 180
        print_and_log(f'Best Scan Rotate Angle: {bast_angle_1} or {bast_angle_2} degrees')
        print_and_log('No Need to Flip')
    else:
        bast_angle_1 = - test_angle_range[ind_flip_min]
        if bast_angle_1 <= 0:
            bast_angle_2 = bast_angle_1 + 180
        else:
            bast_angle_2 = bast_angle_1 - 180
        print_and_log(f'Best Scan Rotate Angle: {bast_angle_1} or {bast_angle_2} degrees')
        print_and_log('And Do Flip!')

    '''
    if mean_curl[ind_min] <= mean_curl_flip[ind_flip_min]:
        print_and_log('Best CoM Rotate Angle: {test_angle_range[ind_min]} degrees')
        print_and_log('No Need to Flip')
    else:
        print_and_log('Best CoM Rotate Angle: {test_angle_range[ind_flip_min]} degrees')
        print_and_log('And Do Flip!')
    #'''

    return test_angle_range, mean_curl, mean_curl_flip

    




