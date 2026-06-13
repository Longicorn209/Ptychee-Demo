"""
Author: Zeyu Wang
Date: September 2025
Description: "PtyChee run scripts"
"""

from .utils import *
from .findRotate import *
from .iterPtycho import *
from .analPtycho import *
from .iCoM import *
from .tcBF import *
from .vSTEM import *
from .output import *





def run_findRotateFromCurl(file, data_4D,
        inner_radius = 0, 
        outer_radius = 1,
        save_results = True,
):
    print_and_log('')
    print_and_log(f'find best rotate angle and flip')
    print_and_log(f'File: {file}')

    *_ , aperture_radius, center_dx, center_dy = PACBED_identify(data_4D, BF_threshold=0.5)

    g_origin = calcualte_CoM_origin(data_4D, 
                aperture_radius, 
                center_dx, 
                center_dy,
                inner_radius = inner_radius, 
                outer_radius = outer_radius, 
    )

    test_angle_range, mean_curl, mean_curl_flip = curl_curve(
        g_origin, 
        test_angle_range = np.arange(-89.0, 90.0, 1.0), 
    )

    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    curl_curve_plot(test_angle_range, mean_curl, mean_curl_flip, save_results=save_results)




def run_iCoM(file, data_4D, scan_rotation_angle, scan_flip, 
        inner_radius = 0,                        # inner radius for the annular ROI on CBED
        outer_radius = 10,                       # outer radius for the annular ROI on CBED
        mode = 'iter',                           # 'anal' or 'iter' or 'jacobi'
        n_iter = 500,                            # iteration nunber for 'iter' or 'jacobi' iCoM
        lr = 5e-2,                               # learning rate for 'iter' iCoM
        epsilon_iCoM = 1e-3,                     # epsilon for 'anal' iCoM
        HPF_pixel = 0,                           # radius for the low pass filter on 'anal' iCoM
        dCoM_from = 'iCoM',                      # calculating dCoM from 'CoM' or 'iCoM'
        plot_circle = False,
        save_results = True,
):
    print_and_log('')
    print_and_log(f'iCoM')
    print_and_log(f'File: {file}')
    print_and_log('')
    print_and_log(f'################ Experimental Information ################')
    print_and_log(f'scan rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'scan flip x: {scan_flip}')

    *_ , aperture_radius, center_dx, center_dy = PACBED_identify(data_4D, BF_threshold=0.5)
    g = calcualte_CoM_origin(data_4D, aperture_radius, center_dx, center_dy, inner_radius, outer_radius)
    
    CoM = calculate_CoM(g, scan_rotation_angle, scan_flip)
    if mode == 'iter':
        iCoM, loss = iter_iCoM(CoM, lr = lr, n_iter = n_iter)
    elif mode == 'jacobi':
        iCoM, loss = jacobi_iCoM(CoM, n_iter = n_iter) 
    elif mode == 'anal':
        iCoM = anal_iCOM(CoM, epsilon_iCoM = epsilon_iCoM, HPF_pixel = HPF_pixel)
        loss = None

    if dCoM_from == 'CoM':
        dCoM = calculate_dCoM_from_CoM(CoM)
    elif dCoM_from == 'iCoM':
        dCoM = calculate_dCoM_from_iCoM(iCoM)

    g_cir = None
    if plot_circle:
        xx, yy = np.meshgrid(np.arange(128)-64,np.arange(128)-64)
        rr = np.sqrt(xx**2+yy**2)
        XX = -xx * (rr < 64)
        YY = -yy * (rr < 64)
        g_cir = XX + 1j*YY

    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    CoM_plot(CoM, g_circle=g_cir, save_results = save_results)
    iCoM_plot(iCoM, loss, save_results = save_results)
    dCoM_plot(dCoM, save_results = save_results)

    




def run_msLSQML(file, data_4D, Voltage, alpha, scan_step, scan_rotation_angle, scan_flip,
        defocus,
        n_state,                                 # probe states number
        n_slice,                                 # object slices number
        slice_thickness,                         # object slice thickness in Å
        iter_max,                                # maximum iteration numner
        s_O,                                     # step size for updating Object Function
        s_P,                                     # step size for updating Probe Function
        n_block = None,                          # block number：The CBEDs are divided into n_block blocks for batch computing, must be divisible by the number of scanning positions
        BF_threshold = 0.5,                      # for calclulating aperture radius
        forced_aperture_radius = None,           # pixels, force the aperture radius to the given value, set to None if not needed
        e_f = 1e-9,                              # epsilon for reciprocal space updating
        e_g = 1e-1,                              # epsilin for real space updating
        e_LSQ = 5e-1,                            # eplison for LSQ step calculation
        position_shuffle = True,                 # shuffle the order of the scanning positions, can effectively suppress some periodic artifacts
        probe_orthog_constr = False,             # orthogonal constraint for mix-state probes
        sorting_probe = True,                    # then the probes are sorted by their energies
        POA = False,                             # phase object approximation constraint 
        ks_softThreshold = 0,                    # soft threshold for sparse constraint on object FFT
        kh_hardThreshold = 0,                    # hard threshold for sparse constraint on object FFT
        kz_regularization = 0,                   # kz constraint for the 'missing cone problem'
        rh_positive_phase = False,               # rh constraint: If True, all negative values in the phase result are clipped
        FFT_phase_offset = 0,
        S_position_correction = 0,
        Obj_pad = 10,
        plot_and_save_err = True,
        plot_and_save_LSQ_step = False,
        plot_and_save_object = True,
        plot_and_save_probe = True,
        show_probe_comparison = False, 
        show_proben_Phase = False, 
        save_results = True,
):
    print_and_log('')
    print_and_log(f'multi-slice LSQML')
    print_and_log(f'File: {file}')
    print_and_log('')
    print_and_log(f'################ Experimental Information ################')
    print_and_log(f'Voltage: {Voltage} kV')
    print_and_log(f'Alpha: {alpha} rad')
    print_and_log(f'scan step: {scan_step} Å/pixel')
    print_and_log(f'defocus: {defocus} Å')
    print_and_log(f'scan rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'scan flip x: {scan_flip}')
    
    data_shape = data_4D.shape

    paCBED, aperture_radius, *_ = PACBED_identify(data_4D, BF_threshold, forced_aperture_radius = forced_aperture_radius)
    reciprocalSpace_pixel_size = calculate_recipro_pixeSize(Voltage, alpha, aperture_radius)
    ptycho_move = calculate_iterPtycho_move(reciprocalSpace_pixel_size, scan_step, data_shape)
    posset = initialize_iterPtycho_patch_position(data_shape, scan_rotation_angle, scan_flip, ptycho_move, Obj_pad = Obj_pad)

    proben0, k_theta = initialize_iterPtycho_probe_mixstates(paCBED, reciprocalSpace_pixel_size, Voltage, defocus, n_state)
    objectn = inialize_iterPtycho_object_multislice(data_shape, posset, n_slice, Obj_pad = Obj_pad)
    propagators = initialize_iterPtycho_z_propagators(Voltage, k_theta, slice_thickness)


    msLSQML_results = msLSQML_pc_engine(iter_max, s_O, s_P, S_position_correction,
                data_4D, posset, proben0, objectn, propagators,
                n_block = n_block, 
                e_f = e_f,
                e_g = e_g,
                e_LSQ = e_LSQ,
                position_shuffle = position_shuffle,
                probe_orthog_constr = probe_orthog_constr,
                sorting_probe = sorting_probe ,
                POA = POA, 
                ks_softThreshold = ks_softThreshold,
                kh_hardThreshold = kh_hardThreshold,
                kz_regularization = kz_regularization,
                rh_positive_phase = rh_positive_phase,
                FFT_phase_offset = FFT_phase_offset,
    )
    msLSQML_Obj, msLSQML_Prb, msLSQML_err, Obj_LSQ_step, Prb_LSQ_step, pc_mean_shift, posset_pc = msLSQML_results

    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    if plot_and_save_err:    
        iterPtycho_error_plot(msLSQML_err, save_results=save_results)
    if plot_and_save_LSQ_step:
        iterPtycho_LSQ_step_plot(Obj_LSQ_step, Prb_LSQ_step, save_results=save_results)
    if plot_and_save_object:
        iterPtycho_objFunc_plot(msLSQML_Obj, scan_rotation_angle, scan_flip, ptycho_move, data_shape, slice_thickness, save_results=save_results)
    if plot_and_save_probe:
        iterPtycho_proben_plot(msLSQML_Prb, proben0, scan_rotation_angle, scan_flip, data_shape, show_probe_comparison=show_probe_comparison, show_proben_Phase=show_proben_Phase, save_results=save_results)
    if S_position_correction:
        position_correction_plot(pc_mean_shift, posset_pc, posset)

    '''
    msLSQML_results = msLSQML_engine(iter_max, s_O, s_P, 
                data_4D, posset, proben0, objectn, propagators,
                n_block = n_block, 
                e_f = e_f,
                e_g = e_g,
                e_LSQ = e_LSQ,
                position_shuffle = position_shuffle,
                probe_orthog_constr = probe_orthog_constr,
                sorting_probe = sorting_probe ,
                POA = POA, 
                ks_softThreshold = ks_softThreshold,
                kh_hardThreshold = kh_hardThreshold,
                kz_regularization = kz_regularization,
                rh_positive_phase = rh_positive_phase,
    )
    msLSQML_Obj, msLSQML_Prb, msLSQML_err, Obj_LSQ_step, Prb_LSQ_step = msLSQML_results

    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    if plot_and_save_err:    
        iterPtycho_error_plot(msLSQML_err, save_results=save_results)
    if plot_and_save_LSQ_step:
        iterPtycho_LSQ_step_plot(Obj_LSQ_step, Prb_LSQ_step, save_results=save_results)
    if plot_and_save_object:
        iterPtycho_objFunc_plot(msLSQML_Obj, scan_rotation_angle, scan_flip, ptycho_move, data_shape, slice_thickness, save_results=save_results)
    if plot_and_save_probe:
        iterPtycho_proben_plot(msLSQML_Prb, proben0, scan_rotation_angle, scan_flip, data_shape, show_probe_comparison=show_probe_comparison, show_proben_Phase=show_proben_Phase, save_results=save_results)
    #'''



def run_msePIE(file, data_4D, Voltage, alpha, scan_step, scan_rotation_angle, scan_flip,
        defocus,
        n_state,                                 # probe states number
        n_slice,                                 # object slices number 
        slice_thickness,                         # object slice thickness in Å
        iter_max,                                # maximum iteration numner
        s_O,                                     # step size for updating Object Function
        s_P,                                     # step size for updating Probe Function
        BF_threshold = 0.5,                      # for calclulating aperture radius
        forced_aperture_radius = None,           # pixels, force the aperture radius to the given value, set to None if not needed
        e_f = 1e-9,                              # epsilon for reciprocal space updating
        position_shuffle = True,                 # shuffle the order of the scanning positions, can effectively suppress some periodic artifacts
        probe_orthog_constr = False,             # orthogonal constraint for mix-state probes 
        sorting_probe = True,                    # then the probes are sorted by their energies 
        POA = False,                             # phase object approximation constraint
        ks_softThreshold = 0,                    # soft threshold for sparse constraint on object FFT
        kh_hardThreshold = 0,                    # hard threshold for sparse constraint on object FFT
        kz_regularization = 0,                   # kz constraint for the 'missing cone problem'
        rh_positive_phase = False,               # rh constraint: If True, all negative values in the phase result are clipped
        Obj_pad = 30,
        plot_and_save_err = True,                   
        plot_and_save_object = True,
        plot_and_save_probe = True,
        show_probe_comparison = False, 
        show_proben_Phase = False, 
        save_results = True,
):
    print_and_log('')
    print_and_log(f'multi-slice ePIE')
    print_and_log(f'File: {file}')
    print_and_log('')
    print_and_log(f'################ Experimental Information ################')
    print_and_log(f'Voltage: {Voltage} kV')
    print_and_log(f'Alpha: {alpha} rad')
    print_and_log(f'scan step: {scan_step} Å/pixel')
    print_and_log(f'defocus: {defocus} Å')
    print_and_log(f'scan rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'scan flip x: {scan_flip}')
    
    data_shape = data_4D.shape

    paCBED, aperture_radius, *_ = PACBED_identify(data_4D, BF_threshold, forced_aperture_radius = forced_aperture_radius)
    reciprocalSpace_pixel_size = calculate_recipro_pixeSize(Voltage, alpha, aperture_radius)
    ptycho_move = calculate_iterPtycho_move(reciprocalSpace_pixel_size, scan_step, data_shape)
    posset = initialize_iterPtycho_patch_position(data_shape, scan_rotation_angle, scan_flip, ptycho_move, Obj_pad=Obj_pad)

    proben0, k_theta = initialize_iterPtycho_probe_mixstates(paCBED, reciprocalSpace_pixel_size, Voltage, defocus, n_state)
    objectn = inialize_iterPtycho_object_multislice(data_shape, posset, n_slice, Obj_pad=Obj_pad)
    propagators = initialize_iterPtycho_z_propagators(Voltage, k_theta, slice_thickness)

    msePIE_results = msePIE_engine(iter_max, s_O, s_P, 
                data_4D, posset, proben0, objectn, propagators,
                e_f = e_f,
                position_shuffle = position_shuffle,
                probe_orthog_constr = probe_orthog_constr,
                sorting_probe = sorting_probe ,
                POA = POA, 
                ks_softThreshold = ks_softThreshold,
                kh_hardThreshold = kh_hardThreshold,
                kz_regularization = kz_regularization,
                rh_positive_phase = rh_positive_phase,
    )
    msePIE_Obj, msePIE_Prb, msePIE_err = msePIE_results

    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    if plot_and_save_err:    
        iterPtycho_error_plot(msePIE_err, save_results=save_results)
    if plot_and_save_object:
        iterPtycho_objFunc_plot(msePIE_Obj, scan_rotation_angle, scan_flip, ptycho_move, data_shape, slice_thickness, save_results=save_results)
    if plot_and_save_probe:
        iterPtycho_proben_plot(msePIE_Prb, proben0, scan_rotation_angle, scan_flip, data_shape, show_probe_comparison=show_probe_comparison, show_proben_Phase=show_proben_Phase, save_results=save_results)




def run_WDD(file, data_4D, Voltage, alpha, scan_step, scan_rotation_angle, scan_flip, 
        defocus,
        epsilon_WDD = 0.1,                       # epsilon for calculating WDD
        plot_FFT = False,
        save_results = True,
):
    print_and_log('')
    print_and_log(f'WDD')
    print_and_log(f'File: {file}')
    print_and_log('')
    print_and_log(f'################ Experimental Information ################')
    print_and_log(f'Voltage: {Voltage} kV')
    print_and_log(f'Alpha: {alpha} rad')
    print_and_log(f'scan step: {scan_step} Å/pixel')
    print_and_log(f'defocus: {defocus} Å')
    print_and_log(f'scan rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'scan flip x: {scan_flip}')
    
    data_shape = data_4D.shape

    *_ , aperture_radius, center_dx, center_dy = PACBED_identify(data_4D, BF_threshold=0.5)
    pPrime_pixel_size_dy, pPrime_pixel_size_dx = calculate_pPrime_calibration(scan_step, data_shape)
    reciprocalSpace_pixel_size = calculate_recipro_pixeSize(Voltage, alpha, aperture_radius)

    obj_fft, obj = WDD_engine(data_4D, 
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
               epsilon_WDD = epsilon_WDD,
               )
    
    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    WDD_plot(obj_fft, obj, plot_FFT=plot_FFT, save_results=save_results)




def run_SSB(file, data_4D, Voltage, alpha, scan_step, scan_rotation_angle, scan_flip,
        plot_FFT = False,
        save_results = True,
):
    print_and_log('')
    print_and_log(f'SSB')
    print_and_log(f'File: {file}')
    print_and_log('')
    print_and_log(f'################ Experimental Information ################')
    print_and_log(f'Voltage: {Voltage} kV')
    print_and_log(f'Alpha: {alpha} rad')
    print_and_log(f'scan step: {scan_step} Å/pixel')
    print_and_log(f'scan rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'scan flip x: {scan_flip}')
    
    data_shape = data_4D.shape

    *_ , aperture_radius, center_dx, center_dy = PACBED_identify(data_4D, BF_threshold=0.5)
    pPrime_pixel_size_dy, pPrime_pixel_size_dx = calculate_pPrime_calibration(scan_step, data_shape)
    reciprocalSpace_pixel_size = calculate_recipro_pixeSize(Voltage, alpha, aperture_radius)

    obj_fft, obj = SSB_engine(data_4D, 
               pPrime_pixel_size_dy, 
               pPrime_pixel_size_dx, 
               center_dy, 
               center_dx, 
               scan_rotation_angle, 
               scan_flip, 
               aperture_radius, 
               reciprocalSpace_pixel_size, 
               )
    
    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    SSB_plot(obj_fft, obj, plot_FFT=plot_FFT, save_results=save_results)




def run_tcBF(file, data_4D, alpha, scan_step, scan_rotation_angle, scan_flip,
        defocus,
        inner_radius = 0,                        # inner radius for the annular ROI on CBED
        outer_radius = 1,                        # outer radius for the annular ROI on CBED
        save_results = True,
):
    print_and_log('')
    print_and_log(f'tcBF')
    print_and_log(f'File: {file}')
    print_and_log('')
    print_and_log(f'################ Experimental Information ################')
    print_and_log(f'Alpha: {alpha} rad')
    print_and_log(f'scan step: {scan_step} Å/pixel')
    print_and_log(f'defocus: {defocus} Å')
    print_and_log(f'scan rotation angle: {scan_rotation_angle} degrees')
    print_and_log(f'scan flip x: {scan_flip}')

    *_ , aperture_radius, center_dx, center_dy = PACBED_identify(data_4D, BF_threshold=0.5)

    Aligned_BF, intensity_min = tcBF(data_4D, alpha, scan_step, scan_rotation_angle, scan_flip, 
                aperture_radius,
                center_dx, 
                center_dy,
                defocus = defocus,
                inner_radius = inner_radius, 
                outer_radius = outer_radius,
                )
    
    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    tcBF_plot(Aligned_BF, intensity_min, save_results=save_results)




def run_vSTEM(file, data_4D,
        radius_list = [ [0, 1],
                        [0, 0.5],
                        [0.5, 1],
                        [1, 2],
                        [1, 3],
                        [2, 3],
                       ],                        # list of inner of outer radiuses for the annular ROIs on CBED
        save_results = True,
):
    print_and_log('')
    print_and_log(f'vSTEM')
    print_and_log(f'File: {file}')
    

    *_ , aperture_radius, center_dx, center_dy = PACBED_identify(data_4D, BF_threshold=0.5)

    vSTEM_siries = vSTEM(data_4D,
        center_dy,
        center_dx,
        aperture_radius,
        radius_list = radius_list,
    )

    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    vSTEM_plot(vSTEM_siries, radius_list, save_results=save_results)
