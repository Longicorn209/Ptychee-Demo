"""
Author: Zeyu Wang
Date: October 2026
Description: "PtyChee run scripts"
"""

from .utils import *
from .findRotate import *
from .iterPtycho import *
from .analPtycho import *
from .iCoM import *
from .tcBF import *
from .vSTEM import *
from .dimReduction import *
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

    return CoM, iCoM, dCoM




def run_LSQML(file, data_4D, Voltage, alpha, scan_step, scan_rotation_angle, scan_flip,
        defocus,
        n_state,                                 # probe states number
        n_slice,                                 # object slices number
        slice_thickness,                         # Å, object slice thickness in Å
        iter_max,                                # maximum iteration number
        s_O,                                     # step size for updating Object Function
        s_P,                                     # step size for updating Probe Function
        n_block = None,                          # block number：The CBEDs are divided into n_block blocks for batch computing, must be divisible by the number of scanning positions
        position_clustering = False,             # cluster spatially adjacent scan positions; otherwise, positions are randomly assigned 
        plot_cluster = False,                    # plot position clustering
        BF_threshold = 0.5,                      # for calclulating aperture radius
        forced_aperture_radius = None,           # pixels, force the aperture radius to the given value, set to None if not needed
        e_f = 1e-9,                              # epsilon for reciprocal space updating
        e_g = 1e-1,                              # epsilin for real space updating
        e_LSQ = 5e-1,                            # eplison for LSQ step calculation

        probe_orthog_constr = False,             # orthogonal constraint for mix-state probes
        sorting_probe = True,                    # then the probes are sorted by their energies
        probe_fit = False,                       # fit probe into Zernike polynomials
        probe_support_mask = False,              # enable probe aperture support constraint or not
        s_position_correction = 0,               # step of position correction
        pc_start_iteration = None,               # position correction start from this iteration

        Obj_pad = 10,                            # pad zero to the object function in x and y dimensions
        POA = False,                             # phase object approximation constraint 
        l2_fft_lambdaTikhonov = 0,               # Tikhonov constraint on object (l2 constraint on object FFT)
        l1_fft_softThreshold = 0,                # soft threshold for sparse constraint on object FFT
        l0_fft_hardThreshold = 0,                # hard threshold for sparse constraint on object FFT
        kz_regularization = 0,                   # kz constraint for the 'missing cone problem'
        zblur_std = 0,                           # Å, z direction Gaussian blur standard deviation in Å
        rh_positive_phase = False,               # rh constraint: If True, all negative values in the phase result are clipped
        FFT_phase_offset = 0,                    # phase offset to remove donut shape

        detector_mask = None,                    # 2D NumPy mask matching the CBED size; 1 = included, 0 = excluded from the calculation

        device = None,                           # specify the GPU device or use "cpu"

        plot_and_save_err = True,
        plot_and_save_LSQ_step = False,
        plot_and_save_CBED_comparison = False,
        plot_and_save_object = True,
        plot_and_save_probe = True,
        show_probe_comparison = False, 
        show_proben_Phase = False, 
        save_results = True,
):
    print_and_log('')
    print_and_log(f'LSQML')
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

    LSQML_results = LSQML_engine(iter_max, s_O, s_P,
                data_4D, Voltage, alpha, aperture_radius, 
                posset, proben0, objectn, slice_thickness, k_theta,
                device = device,
                n_block = n_block, 
                position_clustering = position_clustering, 
                plot_cluster = plot_cluster,
                e_f = e_f,
                e_g = e_g,
                e_LSQ = e_LSQ,
                probe_orthog_constr = probe_orthog_constr,
                sorting_probe = sorting_probe,
                probe_fit = probe_fit,
                probe_support_mask = probe_support_mask,
                s_position_correction = s_position_correction,
                pc_start_iteration = pc_start_iteration,
                POA = POA, 
                l2_fft_lambdaTikhonov = l2_fft_lambdaTikhonov,
                l1_fft_softThreshold = l1_fft_softThreshold,
                l0_fft_hardThreshold = l0_fft_hardThreshold,
                kz_regularization = kz_regularization,
                zblur_std = zblur_std,
                rh_positive_phase = rh_positive_phase,
                FFT_phase_offset = FFT_phase_offset,
                detector_mask = detector_mask,
    )
    LSQML_Obj, LSQML_Prb, Prb_EW, LSQML_err, Obj_LSQ_step, Prb_LSQ_step, pc_mean_shift, posset_pc, cbed_comparison = LSQML_results

    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    if plot_and_save_err:    
        iter_error_plot(LSQML_err, save_results=save_results)
    if plot_and_save_LSQ_step:
        iterPtycho_LSQ_step_plot(Obj_LSQ_step, Prb_LSQ_step, save_results=save_results)
    if plot_and_save_CBED_comparison:
        iterPtycho_CBED_comparison_plot(cbed_comparison, save_results=save_results)
    if plot_and_save_object:
        iterPtycho_objFunc_plot(LSQML_Obj, scan_rotation_angle, scan_flip, ptycho_move, data_shape, slice_thickness, save_results=save_results)
    if plot_and_save_probe:
        iterPtycho_proben_plot(LSQML_Prb, proben0, Prb_EW, scan_rotation_angle, scan_flip, data_shape, show_probe_comparison=show_probe_comparison, show_proben_Phase=show_proben_Phase, save_results=save_results)
    if s_position_correction:
        position_correction_plot(pc_mean_shift, posset_pc, posset)

    return LSQML_Obj, LSQML_Prb




def run_ePIE(file, data_4D, Voltage, alpha, scan_step, scan_rotation_angle, scan_flip,
        defocus,
        n_state,                                 # probe states number
        n_slice,                                 # object slices number
        slice_thickness,                         # Å, object slice thickness in Å
        iter_max,                                # maximum iteration number
        s_O,                                     # step size for updating Object Function
        s_P,                                     # step size for updating Probe Function
        n_block = None,                          # block number：The CBEDs are divided into n_block blocks for batch computing, must be divisible by the number of scanning positions
        position_clustering = False,             # cluster spatially adjacent scan positions; otherwise, positions are randomly assigned 
        plot_cluster = False,                    # plot position clustering
        BF_threshold = 0.5,                      # for calclulating aperture radius
        forced_aperture_radius = None,           # pixels, force the aperture radius to the given value, set to None if not needed
        e_f = 1e-9,                              # epsilon for reciprocal space updating
        alpha_O = 0,                             # Alpha for rPIE Object; default is 0, which corresponds to ePIE
        alpha_P = 0,                             # Alpha for rPIE Probe; default is 0, which corresponds to ePIE
        
        probe_orthog_constr = False,             # orthogonal constraint for mix-state probes
        sorting_probe = True,                    # then the probes are sorted by their energies
        probe_fit = False,                       # fit probe into Zernike polynomials
        probe_support_mask = False,              # enable probe aperture support constraint or not
        s_position_correction = 0,               # step of position correction
        pc_start_iteration = None,               # position correction start from this iteration

        Obj_pad = 10,                            # pad zero to the object function in x and y dimensions
        POA = False,                             # phase object approximation constraint 
        l2_fft_lambdaTikhonov = 0,               # Tikhonov constraint on object (l2 constraint on object FFT)
        l1_fft_softThreshold = 0,                # soft threshold for sparse constraint on object FFT
        l0_fft_hardThreshold = 0,                # hard threshold for sparse constraint on object FFT
        kz_regularization = 0,                   # kz constraint for the 'missing cone problem'
        zblur_std = 0,                           # Å, z direction Gaussian blur standard deviation in Å
        rh_positive_phase = False,               # rh constraint: If True, all negative values in the phase result are clipped
        FFT_phase_offset = 0,                    # phase offset to remove donut shape

        detector_mask = None,                    # 2D NumPy mask matching the CBED size; 1 = included, 0 = excluded from the calculation

        device = None,                           # specify the GPU device or use "cpu"

        plot_and_save_err = True,
        plot_and_save_CBED_comparison = False,
        plot_and_save_object = True,
        plot_and_save_probe = True,
        show_probe_comparison = False, 
        show_proben_Phase = False, 
        save_results = True,
):
    print_and_log('')
    print_and_log(f'ePIE')
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

    ePIE_results = ePIE_engine(iter_max, s_O, s_P,
                data_4D, Voltage, alpha, aperture_radius, 
                posset, proben0, objectn, slice_thickness, k_theta,
                device = device,
                n_block = n_block, 
                position_clustering = position_clustering, 
                plot_cluster = plot_cluster,
                e_f = e_f,
                alpha_O=alpha_O,
                alpha_P=alpha_P,
                probe_orthog_constr = probe_orthog_constr,
                sorting_probe = sorting_probe,
                probe_fit = probe_fit,
                probe_support_mask = probe_support_mask,
                s_position_correction = s_position_correction,
                pc_start_iteration = pc_start_iteration,
                POA = POA, 
                l2_fft_lambdaTikhonov = l2_fft_lambdaTikhonov,
                l1_fft_softThreshold = l1_fft_softThreshold,
                l0_fft_hardThreshold = l0_fft_hardThreshold,
                kz_regularization = kz_regularization,
                zblur_std = zblur_std,
                rh_positive_phase = rh_positive_phase,
                FFT_phase_offset = FFT_phase_offset,
                detector_mask = detector_mask,
    )
    ePIE_Obj, ePIE_Prb, Prb_EW, ePIE_err, pc_mean_shift, posset_pc, cbed_comparison = ePIE_results

    print(Prb_EW.shape)

    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    if plot_and_save_err:    
        iter_error_plot(ePIE_err, save_results=save_results)
    if plot_and_save_CBED_comparison:
        iterPtycho_CBED_comparison_plot(cbed_comparison, save_results=save_results)
    if plot_and_save_object:
        iterPtycho_objFunc_plot(ePIE_Obj, scan_rotation_angle, scan_flip, ptycho_move, data_shape, slice_thickness, save_results=save_results)
    if plot_and_save_probe:
        iterPtycho_proben_plot(ePIE_Prb, proben0, Prb_EW, scan_rotation_angle, scan_flip, data_shape, show_probe_comparison=show_probe_comparison, show_proben_Phase=show_proben_Phase, save_results=save_results)
    if s_position_correction:
        position_correction_plot(pc_mean_shift, posset_pc, posset)

    return ePIE_Obj, ePIE_Prb



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

    return obj




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

    return obj




def run_tcBF(file, data_4D, alpha, scan_step, scan_rotation_angle, scan_flip,
        defocus,
        inner_radius = 0,                        # inner radius for the annular ROI on CBED
        outer_radius = 1,                        # outer radius for the annular ROI on CBED
        contrast_reverse = False,                # reverse contrast or not
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
                contrast_reverse = contrast_reverse,
                )
    
    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    tcBF_plot(Aligned_BF, intensity_min, save_results=save_results)

    return Aligned_BF




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

    return vSTEM_siries




def run_NMF(file, data_4D, 
        n_components, 
        iter_max = 1000,                         # maximum iteration number
        tol = 1e-7,                              # convergence tolerance
        n_block = None,                          # block number for saving memory, default is 1
        alpha_W = 0.0,                           # regularization strength for W
        alpha_H = 0.0,                           # regularization strength for H
        l1_ratio = 0.0,                          # L1/L2 regularization balance (0: L2, 1: L1)

        use_log_scale = False,                   # apply logarithmic scaling to the input data or not

        device = None,                           # specify the GPU device or use "cpu"

        show_W = True,
        show_H = True,
        save_results = True,
):
    print_and_log('')
    print_and_log(f'NMF')
    print_and_log(f'File: {file}')

    PACBED_identify(data_4D, BF_threshold=0.5)
    print_and_log(f'Log Scale: {use_log_scale}')

    if use_log_scale:
        data_4D[data_4D<0] = 0
        data_4D = np.log(data_4D+1)
        
    if n_block is None:
        n_block = 1

    W, H, err_list = fast_hals_NMF(data_4D, n_components, 
        iter_max = iter_max,
        tol = tol, 
        n_block = n_block, 
        alpha_W = alpha_W , 
        alpha_H = alpha_H, 
        l1_ratio = l1_ratio, 
        device = device,
    )

    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    iter_error_plot(err_list, save_results=save_results)
    dimReduction_plot(W, H, show_W=show_W, show_H=show_H, save_results=save_results)

    return W, H




def run_PCA(file, data_4D, 
        n_components,
        n_block = None,                          # block number for saving memory, default is 1

        use_log_scale = False,                   # apply logarithmic scaling to the input data or not

        device = None,                           # specify the GPU device or use "cpu"

        show_W = True, 
        show_H = True,
        save_results = True,
):
    print_and_log('')
    print_and_log(f'PCA')
    print_and_log(f'File: {file}')

    PACBED_identify(data_4D, BF_threshold=0.5)
    print_and_log(f'Log Scale: {use_log_scale}')

    if use_log_scale:
        data_4D[data_4D<0] = 0
        data_4D = np.log(data_4D+1)
        
    if n_block is None:
        n_block = 1

    W, H, Lambda = PCA(data_4D, n_components, 
        n_block = n_block, 
        device = device,
    )

    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    dimReduction_plot(W, H, Lambda=Lambda, show_W=show_W, show_H=show_H, save_results=save_results)

    return W, H


def run_ICA(file, data_4D, 
        n_components, 
        iter_max = 1000,                         # maximum iteration number
        tol = 1e-7,                              # convergence tolerance
        n_block = None,                          # block number for saving memory, default is 1

        use_log_scale = False,                   # apply logarithmic scaling to the input data or not

        device = None,                           # specify the GPU device or use "cpu"

        show_W = True, 
        show_H = True,
        save_results = True,
):
    print_and_log('')
    print_and_log(f'ICA')
    print_and_log(f'File: {file}')

    PACBED_identify(data_4D, BF_threshold=0.5)
    print_and_log(f'Log Scale: {use_log_scale}')

    if use_log_scale:
        data_4D[data_4D<0] = 0
        data_4D = np.log(data_4D+1)
        
    if n_block is None:
        n_block = 1

    W, H = fast_ICA(data_4D, n_components, 
        iter_max = iter_max,
        tol = tol, 
        n_block = n_block, 
        device = device,
    )

    print_and_log('')
    print_and_log(f'################### Saving and Ploting ###################')
    dimReduction_plot(W, H, show_W=show_W, show_H=show_H, save_results=save_results)

    return W, H




def run_INMF(file, data_4D, 
        n_components, 
        iter_max = 1000,                          # maximum iteration number
        tol = 1e-7,                               # convergence tolerance
        n_block = None,                           # block number for saving memory, default is 1
        alpha_W = 0.0,                           # regularization strength for W
        alpha_H = 0.0,                           # regularization strength for H
        l1_ratio = 0.0,                          # L1/L2 regularization balance (0: L2, 1: L1)

        independent_target = 'both',
        independent_components = None,
        independent_iter_max = 1000,
        independent_lr = 1e-2,
        lambda_ind = 0.1,

        use_log_scale = False,                   # apply logarithmic scaling to the input data or not

        device = None,                           # specify the GPU device or use "cpu"

        show_W = True, 
        show_H = True,
        save_results = True,
):
    print_and_log('')
    print_and_log('INMF')
    print_and_log(f'File: {file}')
    print_and_log(f'NMF Components: {n_components}')
    print_and_log(f'Independent Components: {independent_components if independent_components is not None else n_components}')
    print_and_log(f'Independent Target: {independent_target}')
    print_and_log(f'Independent Iterations: {independent_iter_max}')
    print_and_log(f'Independent Learning Rate: {independent_lr}')
    print_and_log(f'Independent Regularization: {lambda_ind}')

    PACBED_identify(data_4D, BF_threshold=0.5)
    print_and_log(f'Log Scale: {use_log_scale}')

    if use_log_scale:
        data_4D[data_4D < 0] = 0
        data_4D = np.log(data_4D + 1)

    if n_block is None:
        n_block = 1

    if independent_components is None:
        independent_components = n_components

    if device is None or device == 'None':
        device = 'cpu'

    W, H, err_list = fast_hals_NMF_independent(
        data_4D = data_4D,
        n_components = n_components,
        iter_max = iter_max,
        tol = tol,
        n_block = n_block,
        alpha_W = alpha_W,
        alpha_H = alpha_H,
        l1_ratio = l1_ratio,
        device = device,
        independent_target = independent_target,
        independent_components = independent_components,
        independent_iter_max = independent_iter_max,
        independent_lr = independent_lr,
        lambda_ind = lambda_ind,
        independent_tol = tol,
    )

    print_and_log('')
    print_and_log('################### Saving and Ploting ###################')
    iter_error_plot(err_list, save_results=save_results)
    dimReduction_plot(W, H, show_W=show_W, show_H=show_H, save_results=save_results)

    return W, H