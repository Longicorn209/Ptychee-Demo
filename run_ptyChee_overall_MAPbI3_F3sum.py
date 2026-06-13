"""
Author: Zeyu Wang
Email: zywang209@gmail.com
Date: October 2025
Description: "PtyChee run example"
"""

import numpy as np
from PtyChee.preProcessing import *
from PtyChee import run


Voltage = 200     #keV
alpha = 0.013     #rad
scan_step = 0.3   #A
defocus = -60     #A

file_read = 'data/'
file = 'MAPbI3_HT200_a13_ss0.3_31_F3sum'
data_4D = np.load(file_read+file+'.npy')



'''
# Find Best Rotate and Flip From Curl
run.run_findRotateFromCurl(file, data_4D,
        inner_radius = 0,              # inner radius for the annular ROI on CBED
        outer_radius = 3,              # outer radius for the annular ROI on CBED
        save_results = True,
)#'''



scan_rotation_angle = -176    #degrees
scan_flip = False             #flip X axis or not




# CoM & iCoM
run.run_iCoM(file, data_4D, scan_rotation_angle, scan_flip, 
        inner_radius = 0,              # inner radius for the annular ROI on CBED
        outer_radius = 3,              # outer radius for the annular ROI on CBED

        #mode = 'iter',                 # 'anal' or 'iter' or 'jacobi'
        #lr = 5e-2,                      # learning rate for iterative iCoM
        #n_iter = 500,                   # iteration nunber for iterative iCoM

        mode = 'jacobi',                 # 'anal' or 'iter' or 'jacobi'
        n_iter = 100,                   # iteration nunber for jacobi iCoM

        #mode = 'anal',                 # 'anal' or 'iter' or 'jacobi'
        #epsilon_iCoM = 1e-3,           # epsilon for analytical iCoM
        #HPF_pixel = 0,                # radius for the low pass filter on analytical iCoM

        #plot_circle = True,
        save_results = True,
)#'''



'''
# multi-slice LSQML
#data_4D = data_4D[400:,400:,:,:]
#data_4D = data_4D[400:,:400,:,:]
#data_4D = data_4D[500:700,:200,:,:]
run.run_msLSQML(file, data_4D, Voltage, alpha, scan_step, scan_rotation_angle, scan_flip,
        defocus,
        n_state = 2,                               # probe states number
        n_slice = 1,                               # object slices number
        slice_thickness = 20,                      # object slice thickness in Å
        iter_max = 50,                            # maximum iteration numner
        s_O = 0.05,                                # step size for updating Object Function
        s_P = 0.05,                                # step size for updating Probe Function
        #n_block = 800,                            # block number: The CBEDs are divided into n_block blocks for batch computing, must be divisible by the number of scanning positions
        #BF_threshold = 0.5,                       # for calclulating aperture radius
        forced_aperture_radius = 5.909,             # pixels, force the aperture radius to the given value, set to None if not needed
        #e_f = 1e-9,                               # epsilon for reciprocal space updating
        #e_g = 5e0,                               # epsilin for real space updating
        #e_LSQ = 5e-3,                             # eplison for LSQ step calculation
        #probe_orthog_constr = True,              # orthogonal constraint for mix-state probes

        #POA = True,                              # phase object approximation constraint
        ks_softThreshold = 0.5,                    # soft threshold for sparse constraint on object FFT
        #kh_hardThreshold = 0,                     # hard threshold for sparse constraint on object FFT
        #kz_regularization = 1,                    # kz constraint for the 'missing cone problem'
        #rh_positive_phase = True,               # rh constraint: If True, all negative values in the phase result are clipped
        #FFT_phase_offset = 0.1,

        plot_and_save_err = True,
        #plot_and_save_LSQ_step = True,
        plot_and_save_object = True,
        plot_and_save_probe = True,
        show_probe_comparison = True,
        show_proben_Phase = True,
        save_results = True,
)#'''



'''
# multi-slice ePIE
data_4D = data_4D[-400:-200,-600:-400,:,:]
run.run_msePIE(file, data_4D, Voltage, alpha, scan_step, scan_rotation_angle, scan_flip, 
        defocus,
        n_state = 2,                               # probe states number
        n_slice = 2,                               # object slices number      
        slice_thickness = 20,                      # object slice thickness in Å
        iter_max = 1,                              # maximum iteration numner
        s_O = 0.1,                                 # step size for updating Object Function
        s_P = 0.01,                                # step size for updating Probe Function
        #BF_threshold = 0.5,                       # for calclulating aperture radius
        #forced_aperture_radius = 5.909,             # pixels, force the aperture radius to the given value, set to None if not needed
        #e_f = 1e-9,                               # epsilon for reciprocal space updating
        #probe_orthog_constr = False,              # orthogonal constraint for mix-state probes
        #POA = False,                              # phase object approximation constraint
        ks_softThreshold = 0.5,                    # soft threshold for sparse constraint on object FFT
        #kh_hardThreshold = 0,                     # hard threshold for sparse constraint on object FFT
        #kz_regularization = 0,                    # kz constraint for the 'missing cone problem'
        plot_and_save_err = True,
        plot_and_save_object = True,
        plot_and_save_probe = True,
        show_probe_comparison = True, 
        show_proben_Phase = True, 
        save_results = True,
)#'''



'''
# WDD
#data_4D = data_4D[-400:,-800:-400,:,:]
#data_4D = data_4D[:,:,15:-15,15:-15]
run.run_WDD(file, data_4D, Voltage, alpha, scan_step, scan_rotation_angle, scan_flip, 
        defocus,
        epsilon_WDD = 0.1,             # epsilon for calculating WDD
        plot_FFT = True,
        save_results = True,
)#'''



'''
# SSB
#data_4D = data_4D[-400:,-800:-400,:,:]
#data_4D = data_4D[:,:,15:-15,15:-15]
run.run_SSB(file, data_4D, Voltage, alpha, scan_step, scan_rotation_angle, scan_flip,
        plot_FFT = True,
        save_results = True,
)#'''



'''
# tcBF
#data_4D = data_4D[-400:-200,-600:-400,:,:]
#data_4D, scan_step = realSpace_interpolation(data_4D, scan_step, 3)
run.run_tcBF(file, data_4D, alpha, scan_step, scan_rotation_angle, scan_flip,
        defocus=-60,
        inner_radius = 0,              # inner radius for the annular ROI on CBED
        outer_radius = 1,              # outer radius for the annular ROI on CBED
        save_results = True,
)#'''



'''
# vSTEM
#data_4D = data_4D[-400:,-800:-400,:,:]
data_4D = recipSpace_bin(data_4D, bin_factor=2)
run.run_vSTEM(file, data_4D,
        radius_list = [ [0, 1],
                        [0, 0.5],
                        [0.5, 1],
                        [1, 2],
                        [1, 3],
                        [2, 3],
                       ],              # list of inner of outer radiuses for the annular ROIs on CBED
        save_results = True,
)#'''