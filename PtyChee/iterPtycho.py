"""
Author: Zeyu Wang & Xiaoyan Wu
Date: September 2025
Description: "iterative ptychogrphay"
"""

import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
from torch.fft import fft2, ifft2, fftn, ifftn, fftshift, ifftshift, fftfreq
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from .utils import print_and_log, log_to_file, get_subplot_grid, RGB_Complex_Plot, calculate_wavelength, Chi_defocus





def calculate_iterPtycho_move(reciprocalSpace_pixel_size, scan_step, data_shape):
    
    dx = data_shape[3]

    realSpace_pixel_size = 1/reciprocalSpace_pixel_size/dx
    ptycho_move = scan_step/realSpace_pixel_size

    print_and_log(f'real space pixel size: {realSpace_pixel_size} Å/pixel')
    print_and_log(f'ptycho move: {ptycho_move} pixels')

    return ptycho_move



def initialize_iterPtycho_patch_position(data_shape, scan_rotation_angle, scan_flip, ptycho_move, Obj_pad):

    sy, sx, *_ = data_shape

    positions = np.zeros((sy*sx,2))
    positions[:, 0] = np.repeat(np.arange(sy), sx)
    positions[:, 1] = np.tile(np.arange(sx), sy)

    if scan_flip:  
        scan_rotation_angle *= -1
    theta_r = np.pi*scan_rotation_angle/180
    #R_cw_array = np.array([[np.cos(theta_r), -np.sin(theta_r)], [np.sin(theta_r), np.cos(theta_r)]])
    R_ccw_array = np.array([[np.cos(theta_r), np.sin(theta_r)], [-np.sin(theta_r), np.cos(theta_r)]])
    positions_spin = np.dot(R_ccw_array, positions.T)

    positions_spin[0] -= np.min(positions_spin[0])
    positions_spin[1] -= np.min(positions_spin[1])

    posset = np.zeros((sy*sx,5))
    posset[:, 0] = positions[:, 0]
    posset[:, 1] = positions[:, 1]
    posset[:, 2] = positions_spin[0]*ptycho_move + Obj_pad
    posset[:, 3] = positions_spin[1]*ptycho_move + Obj_pad
    posset[:, 4] = np.arange(sy*sx)

    if scan_flip:
        posset[:, 3] = np.max(posset[:, 3]) - posset[:, 3] + Obj_pad

    return posset



def initialize_iterPtycho_probe(paCBED, reciprocalSpace_pixel_size, Voltage, defocus, show_probe0=False):

    dy, dx = paCBED.shape

    chi, k_theta = Chi_defocus(dy, dx, reciprocalSpace_pixel_size, Voltage, defocus)

    Ronchi_norm = (paCBED - np.amin(paCBED)) / np.ptp(paCBED)
    BFdisk = np.ones(Ronchi_norm.shape) * (Ronchi_norm > 0.5)

    #ApeFunc = np.sqrt(paCBED)*np.exp(-1j*chi) #/np.mean(np.sqrt(paCBED))
    ApeFunc = BFdisk*np.exp(-1j*chi)
    probe0 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ApeFunc)))

    if show_probe0:

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        im1 = axes[0].imshow(np.abs(probe0), cmap='gray')
        axes[0].set_title('Amplitude Prb0')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        divider1 = make_axes_locatable(axes[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.2)
        fig.colorbar(im1, cax=cax1)

        im2 = axes[1].imshow(np.angle(probe0), cmap='jet')
        axes[1].set_title('Phase Prb0')
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        divider2 = make_axes_locatable(axes[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.2)
        fig.colorbar(im2, cax=cax2)

        plt.tight_layout()
        plt.show()

    return probe0, k_theta



def initialize_iterPtycho_probe_mixstates(paCBED, reciprocalSpace_pixel_size, Voltage, defocus, n_state, gene_mode='Hermite', show_probe_mixstates=False):
    print_and_log('')
    print_and_log(f'############### Probe Mixstates Parameters ###############')

    probe0, k_theta = initialize_iterPtycho_probe(paCBED, reciprocalSpace_pixel_size, Voltage, defocus)

    dy, dx = probe0.shape

    intensity = np.abs(probe0)**2
    total_energy = intensity.sum()

    xx = (np.arange(dx)-dx/2)
    yy = (np.arange(dy)-dy/2)

    xx_grid, yy_grid = np.meshgrid(xx, yy)
    rr = np.sqrt(xx_grid**2 + yy_grid**2)

    sort_idx = np.argsort(rr.flatten())
    cum_energy = np.cumsum(intensity.flatten()[sort_idx])
    w = rr.flatten()[sort_idx][cum_energy >= 0.9*total_energy][0]

    h_list = [[0,0],[0,1],[1,0],[2,0],[1,1],
            [0,2],[0,3],[1,2],[2,1],[3,0],
            [4,0],[3,1],[2,2],[1,3],[0,4],
            [1,4],[2,3],[3,2],[4,1],[4,2],
            [3,3],[2,4],[3,4],[4,3],[4,4]]
    
    proben = np.zeros((n_state, dy, dx), dtype=np.complex64)

    if gene_mode == 'linear':
        for i in range(n_state):
            proben[i] = (1-i/n_state)*probe0

    elif gene_mode == 'Hermite':
        X = xx/w
        Y = yy/w

        def hermiteH(n, x):
            if n == 0:
                return np.ones(len(x))
            elif n == 1:
                return 2*x
            else:
                return 2*x*hermiteH(n-1, x) - 2*(n-1)*hermiteH(n-2, x)

        for i in range(n_state):
            m, n = h_list[i]
            Hx, Hy = np.meshgrid(hermiteH(m, X)*np.exp(-X**2/2), hermiteH(n, Y)*np.exp(-Y**2/2))
            proben[i] = Hx*Hy*probe0/(m+1)**2/(n+1)**2

        intensities = np.sum(np.abs(proben)**2, axis=(-2, -1))
        intensities_order = np.argsort(intensities)[::-1]
        proben = proben[intensities_order]

    elif gene_mode == 'Laguerre':
        X = xx/w
        Y = yy/w
        X_grid, Y_grid = np.meshgrid(X, Y)
        
        R = np.sqrt(X_grid**2 + Y_grid**2)
        theta = np.arctan2(Y_grid, X_grid)

        def laguerre(n, k, x):
            if n == 0:
                return np.ones_like(x)
            elif n == 1:
                return (k + 1 - x)
            else:
                return ( (2*(n-1) + k + 1 - x) * laguerre(n-1, k, x) 
                    - (n-1 + k) * laguerre(n-2, k, x)) / n 
            
        for i in range(n_state):
            p, l = h_list[i]
            L = laguerre(p, l, 2*R**2)
            radial = (R**l)*L*np.exp(-R**2)
            angular = np.cos(l*theta)
            proben[i] = radial*angular*probe0 / ((p+1)**2*(l+1)**2)

            intensities = np.sum(np.abs(proben)**2, axis=(-2, -1))
            intensities_order = np.argsort(intensities)[::-1]
            proben = proben[intensities_order]


    if show_probe_mixstates == True:

        probe_shown_rows, probe_shown_cols = get_subplot_grid(n_state)

        energy_n = np.sum(np.abs(proben)**2, axis=(1,2))
        for i in range(n_state):
            print('  energy of '+str(i+1)+'th probe mode after: '+str(energy_n[i]))
            energy_pp = energy_n[i]/np.sum(energy_n)*100
            plt.subplot(probe_shown_rows, probe_shown_cols, i+1)
            RGB_Complex_Plot(proben[i])
            plt.title('energy proportion: {:^3.0f}%'.format(energy_pp))
        plt.show()

        for i in range(n_state):
            energy_pp = energy_n[i]/np.sum(energy_n)*100
            plt.subplot(probe_shown_rows, probe_shown_cols, i+1)
            plt.imshow(np.abs(proben[i]), cmap='gray')
            plt.title('Amplitude of '+str(i+1)+'th probe\nenergy proportion: {:^3.0f}%'.format(energy_pp))
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
        plt.show()

        for i in range(n_state):
            energy_pp = energy_n[i]/np.sum(energy_n)*100
            plt.subplot(probe_shown_rows, probe_shown_cols, i+1)
            plt.imshow(np.angle(proben[i]), cmap='jet')
            plt.title('Phase of '+str(i+1)+'th probe\nenergy proportion: {:^3.0f}%'.format(energy_pp))
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
        plt.show()

    print_and_log(f'Probe States: {n_state}')
    print_and_log(f'Probe Generation Mode: {gene_mode}')
    print_and_log(f'Probe Radius: {w} pixels')

    return proben, k_theta



def inialize_iterPtycho_object_multislice(data_shape, posset, n_slice, Obj_pad):
    *_ , dy, dx = data_shape

    print_and_log('')
    print_and_log(f'############# Object Multi-slice Parameters ##############')
    print_and_log(f'Object Slices: {n_slice}')

    syptp = np.max(posset[:, 2]) - np.min(posset[:, 2])
    sxptp = np.max(posset[:, 3]) - np.min(posset[:, 3])

    objectn = np.ones((n_slice, int(syptp+1)+dy+2*Obj_pad, int(sxptp+1)+dx+2*Obj_pad))
    print_and_log(f"recovered objFunc shape: {objectn.shape}")
    return objectn



def initialize_iterPtycho_z_propagators(Voltage, k_theta, slice_thickness):

    print_and_log(f'Slice Thickness: {slice_thickness} Å')

    wavelength = calculate_wavelength(Voltage)
    propagators = np.fft.fftshift(np.exp(-1j*2*np.pi/wavelength*0.5*slice_thickness*k_theta**2))
    
    return propagators



def probe_orthogonalization(proben, n_state):
    # compute upper half of P* @ P
    pairwise_dot_product = torch.zeros((n_state, n_state), dtype=torch.complex64, device=device)

    for i in range(n_state):
        for j in range(i,n_state):
            pairwise_dot_product[i, j] = (proben[i].conj() * proben[j]).sum()
    # compute eigenvectors (effectively cheaper way of computing V* from SVD)
    _, evecs = torch.linalg.eigh(pairwise_dot_product, UPLO="U")
    proben = torch.tensordot(evecs.T, proben, dims=1)
    return proben



def ks_constraint(object_GPU, n_slice, ks_softThreshold):
    for sn in range(n_slice):
        objSlice = object_GPU[sn]
        objfft = fft2(objSlice)
        objfft_abs = objfft.abs()

        objfft_abs -= ks_softThreshold*objfft_abs.mean()
        objfft_abs[objfft_abs<=0] = 0

        objfft = objfft_abs*torch.exp(1j*objfft.angle())
        object_GPU[sn] = ifft2(objfft)

    return object_GPU



def kh_constraint(object_GPU, n_slice, kh_hardThreshold):
    for sn in range(n_slice):
        objSlice = object_GPU[sn]
        objfft = fft2(objSlice)
        objfft_abs = objfft.abs()

        hardThreshold = kh_hardThreshold*objfft_abs.mean()
        objfft_abs[objfft_abs<=hardThreshold] = 0

        objfft = objfft_abs*torch.exp(1j*objfft.angle())
        object_GPU[sn] = ifft2(objfft)

    return object_GPU



def kz_constraint(object_GPU, Wz):
    return ifftn(fftn(object_GPU)*Wz)



def rh_constraint(object_GPU, n_slice, rh_hardThreshold=0):
    for sn in range(n_slice):
        objSlice = object_GPU[sn]
        objSlice_phase = objSlice.angle()
        objSlice_abs = objSlice.abs()
                
        objSlice_phase[objSlice_phase<rh_hardThreshold] = 0
        object_GPU[sn] = objSlice_abs*torch.exp(1j*objSlice_phase)

    return object_GPU



def POA_constraint(object_GPU):
    return torch.exp(1j*object_GPU.angle())


def FFT_phase0_offset(object_GPU, n_slice, FFT_phase_offset=0):
    for sn in range(n_slice):
        objSlice = object_GPU[sn]
        objfft = torch.fft.fft2(objSlice)
        objfft_phase = torch.angle(objfft)

        objfft_phase[0][0] += FFT_phase_offset

        objfft = torch.abs(objfft)*torch.exp(1j*objfft_phase)
        object_GPU[sn] = torch.fft.ifft2(objfft)

    return object_GPU



'''
def msLSQML_engine(iter_max, s_O, s_P,
                data_4D, posset, proben0, objectn, propagators,
                n_block = None, 
                e_f = 1e-9,
                e_g = 1e-1,
                e_LSQ = 5e-1,
                #save_gap = None,
                position_shuffle = True,
                probe_orthog_constr = False,
                sorting_probe = True,
                POA = False, 
                ks_softThreshold = 0,
                kh_hardThreshold = 0,
                kz_regularization = 0,
                rh_positive_phase = False,
                FFT_phase_offset = 0,
                ):
    
    #if save_gap is None:
    #    save_gap = iter_max

    sy, sx, dy, dx = data_4D.shape
    n_state = proben0.shape[0]
    n_slice = objectn.shape[0]

    if n_block is None:
        n_block = sy

    print_and_log('')
    print_and_log(f'############## msLSQML Iteration Parameters ##############')
    print_and_log(f"Device Using: {device}")
    print_and_log(f'Number of Iterations: {iter_max}')
    #print_and_log(f'Save Gap: {save_gap}', logfile)
    print_and_log(f'Step for Updating Object: {s_O}')
    print_and_log(f'Step for Updating Probe: {s_P}')
    print_and_log(f'Number of Blocks: {n_block}')
    print_and_log(f'Block Size: {int(sy*sx/n_block)}')
    print_and_log(f'Epsilon for Recipro Optim: {e_f}')
    print_and_log(f'Epsilon for Real Optim: {e_g}')
    print_and_log(f'Epsilon for LSQ: {e_LSQ}')
    print_and_log(f'Scanning Position Shuffle: {position_shuffle}')
    print_and_log(f'Probe Orthog Constraint: {probe_orthog_constr}')
    print_and_log(f'Sorting Probe by Energy: {sorting_probe}')
    print_and_log(f'Phase Object Approximation: {POA}')
    print_and_log(f'Object Positive Phase: {rh_positive_phase}')
    print_and_log(f'Object SoftThreshold: {ks_softThreshold}')
    print_and_log(f'Object HardThreshold: {kh_hardThreshold}')
    print_and_log(f'Object kz Regularization: {kz_regularization}')
    print_and_log(f'Object FFT phase offset: {FFT_phase_offset}')

    data_4D[data_4D<0] = 0
    data_4D_sqrt_GPU = torch.from_numpy(np.reshape(np.sqrt(data_4D), (sy*sx,dy,dx))).to(device)
    proben_GPU = torch.from_numpy(proben0).to(device).to(dtype=torch.complex64)
    object_GPU = torch.from_numpy(objectn).to(device).to(dtype=torch.complex64)
    posset_GPU = torch.from_numpy(posset).to(device)
    propagators_GPU = torch.from_numpy(propagators).to(device).to(dtype=torch.complex64)

    _, Oy, Ox = object_GPU.shape
    y_ind = torch.arange(dy,device=device,dtype=torch.int32)
    x_ind = torch.arange(dx,device=device,dtype=torch.int32)

    #object_siries = torch.zeros((int(iter_max/save_gap), object_GPU.shape), dtype=torch.complex64, device=device)
    #probe_siries = torch.zeros((int(iter_max/save_gap), proben_GPU.shape),dtype=torch.complex64, device=device)
    err = torch.zeros(iter_max, device=device)
    Prb_LSQ_step = torch.zeros((iter_max,n_state), device=device)
    Obj_LSQ_step = torch.zeros((iter_max,n_slice), device=device)
  
    qy = fftfreq(Oy, device=device)
    qx = fftfreq(Ox, device=device)
    qz = fftfreq(n_slice, device=device)
    qza, qya, qxa = torch.meshgrid(qz, qy, qx, indexing="ij")
    qz2 = (qza*kz_regularization)**2
    qr2 = qxa**2+qya**2
    Wz = 1 - 2/torch.pi*torch.arctan2(qz2, qr2)

    start_time = time.time()
    print_and_log('')
    print_and_log(f'#################### msLSQML Process #####################')
    print("\rLSQ-3ML progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0,"*"*0,"."*10,0,0,0),end = "")
    
    for i in range(iter_max):
        time_i0 = time.time()

        err_u = 0
        err_d = 0
        
        if position_shuffle:
            idx = torch.randperm(posset_GPU.shape[0], device=device)
            posset_GPU = posset_GPU[idx]

        b_size = int(sy*sx/n_block)
        possetn_GPU = posset_GPU.reshape((n_block,int(sy*sx/n_block),5))
        poss_1D = possetn_GPU[:,:,4].to(dtype=torch.int64)

        if probe_orthog_constr:
            proben_GPU = probe_orthogonalization(proben_GPU, n_state)

        if sorting_probe:
            intensities = proben_GPU.abs().pow(2).sum(dim=(-2, -1))
            intensities_order = torch.argsort(intensities, descending=True)
            proben_GPU = proben_GPU[intensities_order]

        if i == 0 or (i+1)%10 == 0:
            gObj_d = torch.zeros((Oy,Ox), dtype=torch.complex64, device=device)
            for ii in range(n_block):
                ind_y = (possetn_GPU[ii,:,2][:,None,None]+y_ind[None,:,None]).to(dtype=torch.int32)
                ind_x = (possetn_GPU[ii,:,3][:,None,None]+x_ind[None,None,:]).to(dtype=torch.int32)
                indice = ind_y*Ox + ind_x

                proben_i = proben_GPU[0].abs().pow(2).unsqueeze(0).expand(b_size,dy,dx).to(torch.complex64)
                gObj_d = gObj_d.ravel().index_add_(0,indice.ravel(),proben_i.ravel()).reshape((Oy,Ox))


        for ii in range(n_block):
            ind_y = (possetn_GPU[ii,:,2][:,None,None]+y_ind[None,:,None]).to(dtype=torch.int32).long()
            ind_x = (possetn_GPU[ii,:,3][:,None,None]+x_ind[None,None,:]).to(dtype=torch.int32).long()
            indice = ind_y*Ox + ind_x

            Illuminated_patches = object_GPU[:,ind_y,ind_x]
            amplitudes_patches = data_4D_sqrt_GPU[poss_1D[ii]]

            Prb_ms = []
            Prb_ms.append(proben_GPU.unsqueeze(1))
            for s in range(n_slice):
                psi = Prb_ms[s]*Illuminated_patches[s].unsqueeze(0)
                if s != n_slice-1:
                    psifft = fft2(psi, dim=(2,3))
                    psifft *= propagators_GPU[None,None,:,:]
                    Prb_ms.append(ifft2(psifft, dim=(2,3)))

            psi_fft = fftshift(fft2(psi, dim=(2,3)), dim=(2,3))
            a_psi_fft = psi_fft.abs().pow(2).sum(dim=0).sqrt()
            chi_fft_R = amplitudes_patches/(a_psi_fft+e_f) - 1
            chi_fft = psi_fft*chi_fft_R.unsqueeze(0).to(torch.complex64)
            chi = ifft2(ifftshift(chi_fft, dim=(2,3)), dim=(2,3))
            
            for sr in range(n_slice-1,-1,-1):
                if sr != n_slice-1:
                    chi_fft = fft2(chi, dim=(2,3))
                    chi_fft /= propagators_GPU[None,None,:,:]
                    chi = ifft2(chi_fft, dim=(2,3))

                gPrb_i = chi*Illuminated_patches[sr].unsqueeze(0).conj()
                gPrb_u = gPrb_i.sum(dim=1)
                gPrb_d = Illuminated_patches[sr].abs().pow(2).sum(dim=0).unsqueeze(0)
                gPrb = gPrb_u/(gPrb_d+e_g*gPrb_d.max())

                gObj_i = chi[0]*Prb_ms[sr][0].conj()
                gObj_u = torch.zeros((Oy,Ox), dtype=torch.complex64, device=device)
                gObj_u = gObj_u.ravel().index_add_(0,indice.ravel(),gObj_i.ravel()).reshape((Oy,Ox))
                gObj = gObj_u/(gObj_d**2+(e_g*gObj_d.abs().max())**2).sqrt()

                gObj_patches = gObj[ind_y,ind_x]
                gOP = gObj_patches*Prb_ms[sr].sum(dim=0)
                Mr_0 = (gOP.conj()*chi.sum(dim=0)).real.sum(dim=(1,2))
                Ml_00 = gOP.abs().pow(2).sum(dim=(1,2))
                Ml_00 += e_LSQ*Ml_00.mean()
                a_O = (Mr_0/Ml_00*s_O).mean()/n_slice
                Obj_LSQ_step[i][sr] += a_O/n_block
                object_GPU[sr] += a_O*gObj

                if sr == 0:
                    gPO = gPrb.unsqueeze(1)*Illuminated_patches[sr].unsqueeze(0)
                    Mr_1 = (gPO.conj()*chi).real.sum(dim=(2,3))
                    Ml_11 = gPO.abs().pow(2).sum(dim=(2,3))
                    Ml_11 += e_LSQ*Ml_11.mean()
                    a_P = (Mr_1/Ml_11).mean(dim=1)*s_P
                    Prb_LSQ_step[i] += a_P/n_block
                    proben_GPU += a_P[:,None,None]*gPrb

                chi = gPrb_i

            err_u += (amplitudes_patches-(psi_fft.abs().pow(2).sum(dim=0)).sqrt()).pow(2).sum()
            err_d += amplitudes_patches.to(torch.float32).pow(2).sum()
        err[i] = err_u/err_d

        if kz_regularization > 0:
            object_GPU = kz_constraint(object_GPU, Wz)

        if ks_softThreshold > 0:
            object_GPU = ks_constraint(object_GPU, n_slice, ks_softThreshold)

        if kh_hardThreshold > 0:
            object_GPU = kh_constraint(object_GPU, n_slice, kh_hardThreshold)

        if rh_positive_phase:
            object_GPU = rh_constraint(object_GPU, n_slice, rh_hardThreshold=0)

        if POA:
            object_GPU = POA_constraint(object_GPU)

        if FFT_phase_offset > 0 and i>iter_max*1/2:
            object_GPU = FFT_phase0_offset(object_GPU, n_slice, FFT_phase_offset)

        #if (i+1)%save_gap == 0:
        #    object_siries[int((i+1)/save_gap-1)] = object_GPU
        #    probe_siries[int((i+1)/save_gap-1)] = proben_GPU

        time_is = time.time()-time_i0
        speed = 1/time_is
        process = (i+1)/iter_max*100
        aa = "*" * int(process/10)
        bb = "." * (10-int(process/10))
        dur = int(time.time() - start_time)
        time_remain = int((iter_max-i-1)*time_is)
        dur_h = dur//3600
        dur_m = (dur-dur_h*3600)//60
        dur_s = dur-dur_h*3600-dur_m*60
        time_remain_h = time_remain//3600
        time_remain_m = (time_remain-time_remain_h*3600)//60
        time_remain_s = time_remain-time_remain_h*3600-time_remain_m*60
        print("\rLSQ-3ML progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s),end = "")
    print('')
    gpu_time = time.time() - start_time
    
    msLSQML_Obj = np.array(object_GPU.cpu())
    msLSQML_Prb = np.array(proben_GPU.cpu())
    msLSQML_err = np.array(err.cpu())
    Obj_LSQ_step = np.array(Obj_LSQ_step.cpu())
    Prb_LSQ_step = np.array(Prb_LSQ_step.cpu())

    log_to_file("LSQ-3ML progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s))
    print_and_log(f'LSQ-3ML process finished in {gpu_time} s')

    return msLSQML_Obj, msLSQML_Prb, msLSQML_err, Obj_LSQ_step, Prb_LSQ_step
#'''




def msLSQML_pc_engine(iter_max, s_O, s_P, s_PC,
                data_4D, posset, proben0, objectn, propagators,
                n_block = None, 
                e_f = 1e-9,
                e_g = 1e-1,
                e_LSQ = 5e-1,
                #save_gap = None,
                position_shuffle = True,
                probe_orthog_constr = False,
                sorting_probe = True,
                POA = False, 
                ks_softThreshold = 0,
                kh_hardThreshold = 0,
                kz_regularization = 0,
                rh_positive_phase = False,
                FFT_phase_offset = 0,
                ):
    
    #if save_gap is None:
    #    save_gap = iter_max

    sy, sx, dy, dx = data_4D.shape
    n_state = proben0.shape[0]
    n_slice = objectn.shape[0]

    if n_block is None:
        n_block = sy

    print_and_log('')
    print_and_log(f'############## msLSQML Iteration Parameters ##############')
    print_and_log(f"Device Using: {device}")
    print_and_log(f'Number of Iterations: {iter_max}')
    #print_and_log(f'Save Gap: {save_gap}', logfile)
    print_and_log(f'Step for Position Correction: {s_PC}')
    print_and_log(f'Step for Updating Object: {s_O}')
    print_and_log(f'Step for Updating Probe: {s_P}')
    print_and_log(f'Number of Blocks: {n_block}')
    print_and_log(f'Block Size: {int(sy*sx/n_block)}')
    print_and_log(f'Epsilon for Recipro Optim: {e_f}')
    print_and_log(f'Epsilon for Real Optim: {e_g}')
    print_and_log(f'Epsilon for LSQ: {e_LSQ}')
    print_and_log(f'Scanning Position Shuffle: {position_shuffle}')
    print_and_log(f'Probe Orthog Constraint: {probe_orthog_constr}')
    print_and_log(f'Sorting Probe by Energy: {sorting_probe}')
    print_and_log(f'Phase Object Approximation: {POA}')
    print_and_log(f'Object Positive Phase: {rh_positive_phase}')
    print_and_log(f'Object SoftThreshold: {ks_softThreshold}')
    print_and_log(f'Object HardThreshold: {kh_hardThreshold}')
    print_and_log(f'Object kz Regularization: {kz_regularization}')
    print_and_log(f'Object FFT phase offset: {FFT_phase_offset}')

    data_4D[data_4D<0] = 0
    data_4D_sqrt_GPU = torch.from_numpy(np.reshape(np.sqrt(data_4D), (sy*sx,dy,dx))).to(device)
    proben_GPU = torch.from_numpy(proben0).to(device).to(dtype=torch.complex64)
    object_GPU = torch.from_numpy(objectn).to(device).to(dtype=torch.complex64)
    posset_GPU = torch.from_numpy(posset).to(device)
    propagators_GPU = torch.from_numpy(propagators).to(device).to(dtype=torch.complex64)

    _, Oy, Ox = object_GPU.shape
    y_ind = torch.arange(dy,device=device,dtype=torch.int32)
    x_ind = torch.arange(dx,device=device,dtype=torch.int32)

    #object_siries = torch.zeros((int(iter_max/save_gap), object_GPU.shape), dtype=torch.complex64, device=device)
    #probe_siries = torch.zeros((int(iter_max/save_gap), proben_GPU.shape),dtype=torch.complex64, device=device)
    err = torch.zeros(iter_max, device=device)
    Prb_LSQ_step = torch.zeros((iter_max,n_state), device=device)
    Obj_LSQ_step = torch.zeros((iter_max,n_slice), device=device)
    pc_mean_shift = torch.zeros(iter_max, device=device)

    qmy, qmx = torch.meshgrid(fftfreq(dx,device=device), fftfreq(dy,device=device),indexing='ij')
  
    qy = fftfreq(Oy, device=device)
    qx = fftfreq(Ox, device=device)
    qz = fftfreq(n_slice, device=device)
    qza, qya, qxa = torch.meshgrid(qz, qy, qx, indexing="ij")
    qz2 = (qza*kz_regularization)**2
    qr2 = qxa**2+qya**2
    Wz = 1 - 2/torch.pi*torch.arctan2(qz2, qr2)

    start_time = time.time()
    print_and_log('')
    print_and_log(f'#################### msLSQML Process #####################')
    print("\rLSQ-3ML progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0,"*"*0,"."*10,0,0,0),end = "")
    
    for i in range(iter_max):
        time_i0 = time.time()

        err_u = 0
        err_d = 0
        
        if position_shuffle:
            idx = torch.randperm(posset_GPU.shape[0], device=device)
            posset_GPU = posset_GPU[idx]

        b_size = int(sy*sx/n_block)
        possetn_GPU = posset_GPU.reshape((n_block,int(sy*sx/n_block),5))
        poss_1D = possetn_GPU[:,:,4].to(dtype=torch.int64)

        if probe_orthog_constr:
            proben_GPU = probe_orthogonalization(proben_GPU, n_state)

        if sorting_probe:
            intensities = proben_GPU.abs().pow(2).sum(dim=(-2, -1))
            intensities_order = torch.argsort(intensities, descending=True)
            proben_GPU = proben_GPU[intensities_order]

        if i == 0 or (i+1)%10 == 0:
            gObj_d = torch.zeros((Oy,Ox), dtype=torch.complex64, device=device)
            for ii in range(n_block):
                ind_y = (possetn_GPU[ii,:,2][:,None,None]+y_ind[None,:,None]).to(dtype=torch.int32)
                ind_x = (possetn_GPU[ii,:,3][:,None,None]+x_ind[None,None,:]).to(dtype=torch.int32)
                indice = ind_y*Ox + ind_x

                proben_i = proben_GPU[0].abs().pow(2).unsqueeze(0).expand(b_size,dy,dx).to(torch.complex64)
                gObj_d = gObj_d.ravel().index_add_(0,indice.ravel(),proben_i.ravel()).reshape((Oy,Ox))


        for ii in range(n_block):
            ind_y = (possetn_GPU[ii,:,2][:,None,None]+y_ind[None,:,None]).to(dtype=torch.int32).long()
            ind_x = (possetn_GPU[ii,:,3][:,None,None]+x_ind[None,None,:]).to(dtype=torch.int32).long()
            indice = ind_y*Ox + ind_x

            Illuminated_patches = object_GPU[:,ind_y,ind_x]
            amplitudes_patches = data_4D_sqrt_GPU[poss_1D[ii]]

            Prb_ms = []
            Prb_ms.append(proben_GPU.unsqueeze(1))
            for s in range(n_slice):
                psi = Prb_ms[s]*Illuminated_patches[s].unsqueeze(0)
                if s != n_slice-1:
                    psifft = fft2(psi, dim=(2,3))
                    psifft *= propagators_GPU[None,None,:,:]
                    Prb_ms.append(ifft2(psifft, dim=(2,3)))

            psi_fft = fftshift(fft2(psi, dim=(2,3)), dim=(2,3))
            a_psi_fft = psi_fft.abs().pow(2).sum(dim=0).sqrt()
            chi_fft_R = amplitudes_patches/(a_psi_fft+e_f) - 1
            chi_fft = psi_fft*chi_fft_R.unsqueeze(0).to(torch.complex64)
            chi = ifft2(ifftshift(chi_fft, dim=(2,3)), dim=(2,3))
            
            for sr in range(n_slice-1,-1,-1):
                if sr != n_slice-1:
                    chi_fft = fft2(chi, dim=(2,3))
                    chi_fft /= propagators_GPU[None,None,:,:]
                    chi = ifft2(chi_fft, dim=(2,3))

                gPrb_i = chi*Illuminated_patches[sr].unsqueeze(0).conj()
                gPrb_u = gPrb_i.sum(dim=1)
                gPrb_d = Illuminated_patches[sr].abs().pow(2).sum(dim=0).unsqueeze(0)
                gPrb = gPrb_u/(gPrb_d+e_g*gPrb_d.max())

                gObj_i = chi[0]*Prb_ms[sr][0].conj()
                gObj_u = torch.zeros((Oy,Ox), dtype=torch.complex64, device=device)
                gObj_u = gObj_u.ravel().index_add_(0,indice.ravel(),gObj_i.ravel()).reshape((Oy,Ox))
                gObj = gObj_u/(gObj_d**2+(e_g*gObj_d.abs().max())**2).sqrt()

                gObj_patches = gObj[ind_y,ind_x]
                gOP = gObj_patches*Prb_ms[sr].sum(dim=0)
                Mr_0 = (gOP.conj()*chi.sum(dim=0)).real.sum(dim=(1,2))
                Ml_00 = gOP.abs().pow(2).sum(dim=(1,2))
                Ml_00 += e_LSQ*Ml_00.mean()
                a_O = (Mr_0/Ml_00*s_O).mean()/n_slice
                Obj_LSQ_step[i][sr] += a_O/n_block
                object_GPU[sr] += a_O*gObj

                if sr == 0:
                    gPO = gPrb.unsqueeze(1)*Illuminated_patches[sr].unsqueeze(0)
                    Mr_1 = (gPO.conj()*chi).real.sum(dim=(2,3))
                    Ml_11 = gPO.abs().pow(2).sum(dim=(2,3))
                    Ml_11 += e_LSQ*Ml_11.mean()
                    a_P = (Mr_1/Ml_11).mean(dim=1)*s_P
                    Prb_LSQ_step[i] += a_P/n_block
                    proben_GPU += a_P[:,None,None]*gPrb

                chi = gPrb_i


                if i>=iter_max//4 and sr == n_slice//2 and s_PC:
                    I_patches_fft = fft2(Illuminated_patches[sr], dim=(1,2))
                    I_patches_gx = ifft2(2j*torch.pi*I_patches_fft*qmx[None,:,:],dim=(1,2))#.to(torch.complex64)
                    I_patches_gy = ifft2(2j*torch.pi*I_patches_fft*qmy[None,:,:],dim=(1,2))#.to(torch.complex64)
                    
                    #gxP = I_patches_gx*Prb_ms[sr].sum(dim=0)
                    #Mr_x = (gxP.conj()*chi.sum(dim=0)).real.sum(dim=(1,2))
                    gxP = I_patches_gx*Prb_ms[sr][0]
                    Mr_x = (gxP.conj()*chi[0]).real.sum(dim=(1,2))
                    Ml_x = gxP.abs().pow(2).sum(dim=(1,2))
                    gx = s_PC*(Mr_x/Ml_x).real

                    #gyP = I_patches_gy*Prb_ms[sr].sum(dim=0)
                    #Mr_y = (gyP.conj()*chi.sum(dim=0)).real.sum(dim=(1,2))
                    gyP = I_patches_gy*Prb_ms[sr][0]
                    Mr_y = (gyP.conj()*chi[0]).real.sum(dim=(1,2))
                    Ml_y = gyP.abs().pow(2).sum(dim=(1,2))
                    gy = s_PC*(Mr_y/Ml_y).real

                    possetn_GPU[ii,:,2] += gy
                    possetn_GPU[ii,:,3] += gx

                    pc_mean_shift[i] += (gy**2+gx**2).sqrt().mean()


            err_u += (amplitudes_patches-(psi_fft.abs().pow(2).sum(dim=0)).sqrt()).pow(2).sum()
            err_d += amplitudes_patches.to(torch.float32).pow(2).sum()
        err[i] = err_u/err_d
        posset_GPU = possetn_GPU.reshape(-1, 5)

        if kz_regularization > 0:
            object_GPU = kz_constraint(object_GPU, Wz)

        if ks_softThreshold > 0:
            object_GPU = ks_constraint(object_GPU, n_slice, ks_softThreshold)

        if kh_hardThreshold > 0:
            object_GPU = kh_constraint(object_GPU, n_slice, kh_hardThreshold)

        if rh_positive_phase:
            object_GPU = rh_constraint(object_GPU, n_slice, rh_hardThreshold=0)

        if POA:
            object_GPU = POA_constraint(object_GPU)

        if FFT_phase_offset > 0 and i>iter_max*2/3:
            object_GPU = FFT_phase0_offset(object_GPU, n_slice, FFT_phase_offset)

        #if (i+1)%save_gap == 0:
        #    object_siries[int((i+1)/save_gap-1)] = object_GPU
        #    probe_siries[int((i+1)/save_gap-1)] = proben_GPU

        time_is = time.time()-time_i0
        speed = 1/time_is
        process = (i+1)/iter_max*100
        aa = "*" * int(process/10)
        bb = "." * (10-int(process/10))
        dur = int(time.time() - start_time)
        time_remain = int((iter_max-i-1)*time_is)
        dur_h = dur//3600
        dur_m = (dur-dur_h*3600)//60
        dur_s = dur-dur_h*3600-dur_m*60
        time_remain_h = time_remain//3600
        time_remain_m = (time_remain-time_remain_h*3600)//60
        time_remain_s = time_remain-time_remain_h*3600-time_remain_m*60
        print("\rLSQ-3ML progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s),end = "")
    print('')
    gpu_time = time.time() - start_time
    
    msLSQML_Obj = np.array(object_GPU.cpu())
    msLSQML_Prb = np.array(proben_GPU.cpu())
    msLSQML_err = np.array(err.cpu())
    Obj_LSQ_step = np.array(Obj_LSQ_step.cpu())
    Prb_LSQ_step = np.array(Prb_LSQ_step.cpu())
    pc_mean_shift = np.array(pc_mean_shift.cpu())
    posset = np.array(posset_GPU.cpu())

    log_to_file("LSQ-3ML progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s))
    print_and_log(f'LSQ-3ML process finished in {gpu_time} s')

    return msLSQML_Obj, msLSQML_Prb, msLSQML_err, Obj_LSQ_step, Prb_LSQ_step, pc_mean_shift, posset





def msePIE_engine(iter_max, s_O, s_P, 
                data_4D, posset, proben0, objectn, propagators,
                e_f = 1e-9,
                #save_gap = None,
                position_shuffle = True,
                probe_orthog_constr = False,
                sorting_probe = True,
                POA = False, 
                ks_softThreshold = 0,
                kh_hardThreshold = 0,
                kz_regularization = 0,
                rh_positive_phase = False,
                ):
    
    #if save_gap is None:
    #    save_gap = iter_max

    print_and_log('')
    print_and_log(f'############## msePIE Iteration Parameters ###############')
    print_and_log(f"Device Using: {device}")
    print_and_log(f'Number of Iterations: {iter_max}')
    #print_and_log(f'Save Gap: {save_gap}')
    print_and_log(f'Step for Updating Object: {s_O}')
    print_and_log(f'Step for Updating Probe: {s_P}')
    print_and_log(f'Epsilon for Recipro Optim: {e_f}')
    print_and_log(f'Scanning Position Shuffle: {position_shuffle}')
    print_and_log(f'Probe Orthog Constraint: {probe_orthog_constr}')
    print_and_log(f'Sorting Probe by Energy: {sorting_probe}')
    print_and_log(f'Phase Object Approximation: {POA}')
    print_and_log(f'Object Positive Phase: {rh_positive_phase}')
    print_and_log(f'Object SoftThreshold: {ks_softThreshold}')
    print_and_log(f'Object HardThreshold: {kh_hardThreshold}')
    print_and_log(f'Object kz Regularization: {kz_regularization}')


    sy, sx, dy, dx = data_4D.shape
    n_state = proben0.shape[0]
    n_slice = objectn.shape[0]

    data_4D[data_4D<0] = 0
    data_4D_sqrt_GPU = torch.from_numpy(np.sqrt(data_4D)).to(device)
    proben_GPU = torch.from_numpy(proben0).to(device).to(dtype=torch.complex64)
    object_GPU = torch.from_numpy(objectn).to(device).to(dtype=torch.complex64)
    posset_GPU = torch.from_numpy(posset).to(device)
    propagators_GPU = torch.from_numpy(propagators).to(device).to(dtype=torch.complex64)

    _, Oy, Ox = object_GPU.shape
    err = torch.zeros(iter_max, device=device)
    #object_siries = torch.zeros((int(iter_max/save_gap), object_GPU.shape), dtype=torch.complex64, device=device)
    #probe_siries = torch.zeros((int(iter_max/save_gap), proben_GPU.shape),dtype=torch.complex64, device=device)
    
    qy = fftfreq(Oy, device=device)
    qx = fftfreq(Ox, device=device)
    qz = fftfreq(n_slice, device=device)
    qza, qya, qxa = torch.meshgrid(qz, qy, qx, indexing="ij")
    qz2 = (qza*kz_regularization)**2
    qr2 = qxa**2+qya**2
    Wz = 1 - 2/torch.pi*torch.arctan2(qz2, qr2)

    start_time = time.time()
    print_and_log('')
    print_and_log(f'##################### msePIE Process #####################')
    print("\r3PIE progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0,"*"*0,"."*10,0,0,0),end = "")
    for i in range(iter_max):
        time_i0 = time.time()

        err_u = 0
        err_d = 0
        
        if position_shuffle == 1:
            idx = torch.randperm(posset_GPU.shape[0], device=device)
            posset_GPU = posset_GPU[idx]

        if probe_orthog_constr  == 1:
            proben_GPU = probe_orthogonalization(proben_GPU, n_state)

        if sorting_probe == 1:
            intensities = proben_GPU.abs().pow(2).sum(dim=(-2, -1))
            intensities_order = torch.argsort(intensities, descending=True)
            proben_GPU = proben_GPU[intensities_order]

        for iy, ix, iyr, ixr, ii in posset_GPU:
            y0, x0 = int(iyr), int(ixr)
            y1, x1 = y0 + dy, x0 + dx

            objectIlluminated = object_GPU[:, y0:y1, x0:x1]
            objectObservedSqrt = data_4D_sqrt_GPU[int(iy), int(ix)]

            psi_i = torch.ones((n_slice, n_state, dy, dx), dtype=torch.complex64, device=device)
            psi_i_p = torch.ones((n_slice, n_state, dy, dx), dtype=torch.complex64, device=device)
            psi_e = torch.ones((n_slice, n_state, dy, dx), dtype=torch.complex64, device=device)
            psi_e_p = torch.ones((n_slice, n_state, dy, dx), dtype=torch.complex64, device=device)
            oPrime = torch.ones((n_slice, dy, dx), dtype=torch.complex64, device=device)

            psi_i[0] = proben_GPU

            for s in range(n_slice):
                psi_e[s] = psi_i[s]*objectIlluminated[s][:None]
                if s != n_slice-1:
                    psifft = fft2(psi_e[s], dim=(1,2))
                    psifft *= propagators_GPU[:None]
                    psi_i[s+1] = ifft2(psifft, dim=(1,2))

            psi_ew = psi_e[-1]
            psi_ew_fft = fftshift(fft2(psi_ew, dim=(1,2)), dim=(1,2))
            psi_ew_O_fft = objectObservedSqrt[:None]*psi_ew_fft
            psi_ew_O_fft /= (psi_ew_fft.abs().pow(2).sum(dim=0).sqrt()+e_f)[:None]
            psi_ew_O =  ifft2(ifftshift(psi_ew_O_fft, dim=(1,2)), dim=(1,2))
            psi_e_p[-1] = psi_ew_O

            for sr in range(n_slice-1,-1,-1):

                if sr == n_slice-1:
                    oPrime[sr] = objectIlluminated[sr] + s_O*psi_i[sr][0].conj()*(psi_e_p[sr][0]-psi_e[sr][0])/psi_i[sr][0].abs().pow(2).max()
                    psi_i_p[sr] = psi_i[sr] + s_P*(psi_e_p[sr]-psi_e[sr])*objectIlluminated[sr][:None].conj()/objectIlluminated[sr].abs().pow(2).max()

                else:
                    psiPrimefft = fft2(psi_i_p[sr+1], dim=(1,2))
                    psiPrimefft /= propagators_GPU[:None]
                    psi_e_p[sr] = ifft2(psiPrimefft, dim=(1,2))

                    oPrime[sr] = objectIlluminated[sr] + psi_i[sr][0].conj()*(psi_e_p[sr][0]-psi_e[sr][0])/psi_i[sr][0].abs().pow(2).max()
                    psi_i_p[sr] = psi_i[sr] + (psi_e_p[sr]-psi_e[sr])*objectIlluminated[sr][:None].conj()/objectIlluminated[sr].abs().pow(2).max()

            proben_GPU = psi_i_p[0]
            object_GPU[:, y0:y1, x0:x1] = oPrime

            err_u += (objectObservedSqrt-psi_ew_fft.abs().pow(2).sum(dim=0).sqrt()).pow(2).sum()
            err_d += objectObservedSqrt.to(torch.float32).pow(2).sum()
        err[i] = err_u/err_d

        if kz_regularization > 0:
            object_GPU = kz_constraint(object_GPU, Wz)

        if ks_softThreshold > 0:
            object_GPU = ks_constraint(object_GPU, n_slice, ks_softThreshold)

        if kh_hardThreshold > 0:
            object_GPU = kh_constraint(object_GPU, n_slice, kh_hardThreshold)

        if rh_positive_phase:
            object_GPU = rh_constraint(object_GPU, n_slice, rh_hardThreshold=0)

        if POA:
            object_GPU = POA_constraint(object_GPU)

        #if (i+1)%save_gap == 0:
        #    object_siries[int((i+1)/save_gap-1)] = object_GPU
        #    probe_siries[int((i+1)/save_gap-1)] = proben_GPU

        time_is = time.time()-time_i0
        speed = 1/time_is
        process = (i+1)/iter_max*100
        aa = "*" * int(process/10)
        bb = "." * (10-int(process/10))
        dur = int(time.time() - start_time)
        time_remain = int((iter_max-i-1)*time_is)
        dur_h = dur//3600
        dur_m = (dur-dur_h*3600)//60
        dur_s = dur-dur_h*3600-dur_m*60
        time_remain_h = time_remain//3600
        time_remain_m = (time_remain-time_remain_h*3600)//60
        time_remain_s = time_remain-time_remain_h*3600-time_remain_m*60
        print("\r3PIE progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s),end = "")
    print('')
    total_time = time.time() - start_time

    msePIE_Obj = np.array(object_GPU.cpu())
    msePIE_Prb = np.array(proben_GPU.cpu())
    msePIE_err = np.array(err.cpu())

    log_to_file("3PIE progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process,aa,bb,speed,dur_h,dur_m,dur_s,time_remain_h,time_remain_m,time_remain_s))
    print_and_log(f'3PIE process finished in {total_time} s')

    return msePIE_Obj, msePIE_Prb, msePIE_err
