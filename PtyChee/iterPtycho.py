"""
Author: Zeyu Wang & Xiaoyan Wu
Date: September 2025
Description: "iterative ptychogrphay"
"""

import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.restoration import unwrap_phase

import torch
from torch.fft import fft2, ifft2, fftn, ifftn, fftshift, ifftshift, fftfreq



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

    # Ronchi_norm = (paCBED - np.amin(paCBED)) / np.ptp(paCBED)
    # BFdisk = np.ones(Ronchi_norm.shape) * (Ronchi_norm > 0.5)
    # ApeFunc = BFdisk*np.exp(-1j*chi) * np.mean(np.sqrt(paCBED))

    ApeFunc = np.sqrt(paCBED)*np.exp(-1j*chi) #/ np.mean(np.sqrt(paCBED))
    
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



def probe_orthogonalization(proben, n_state, device):
    # compute upper half of P* @ P
    pairwise_dot_product = torch.zeros((n_state, n_state), dtype=torch.complex64, device=device)

    for i in range(n_state):
        for j in range(i,n_state):
            pairwise_dot_product[i, j] = (proben[i].conj() * proben[j]).sum()
    # compute eigenvectors (effectively cheaper way of computing V* from SVD)
    _, evecs = torch.linalg.eigh(pairwise_dot_product, UPLO="U")
    proben = torch.tensordot(evecs.T, proben, dims=1)
    return proben



def build_pupil_phase_basis(N, r_mask, NA, device):
    y, x = torch.meshgrid(
        torch.linspace(-N//2, N//2-1, N, device=device),
        torch.linspace(-N//2, N//2-1, N, device=device),
        indexing='ij')

    rho_pixels = torch.sqrt(x**2 + y**2)
    mask = rho_pixels <= r_mask
    rho = rho_pixels * NA / r_mask
    theta = torch.atan2(y, x)

    basis = []

    basis.append(torch.ones_like(rho))  # Piston

    basis.append( rho * torch.cos(theta))  # Tilt X
    basis.append( rho * torch.sin(theta))  # Tilt Y

    basis.append(rho**2 * torch.cos(2*theta))      # Astigmatism 0°
    basis.append(0.5 * rho**2)                     # Defocus 1/2
    basis.append(rho**2 * torch.sin(2*theta))      # Astigmatism 45°

    basis.append((1/3) * rho**3 * torch.sin(3*theta)) # Trefoil X
    basis.append((1/3) * rho**3 * torch.sin(theta))   # Coma X
    basis.append((1/3) * rho**3 * torch.cos(theta))   # Coma Y
    basis.append((1/3) * rho**3 * torch.cos(3*theta)) # Trefoil Y

    basis.append((1/4) * rho**4 * torch.sin(4*theta))  # Quadrafoil 0°
    basis.append((1/4) * rho**4 * torch.sin(2*theta))  # Secondary Astig 45°
    basis.append((1/4) * rho**4)                        # Primary Spherical
    basis.append((1/4) * rho**4 * torch.cos(2*theta))  # Secondary Astig 0°
    basis.append((1/4) * rho**4 * torch.cos(4*theta))  # Quadrafoil 45°

    basis.append((1/5) * rho**5 * torch.sin(5*theta))  # Pentafoil 45°
    basis.append((1/5) * rho**5 * torch.sin(3*theta))  # Secondary Trefoil Y
    basis.append((1/5) * rho**5 * torch.sin(theta))    # Secondary Coma Y
    basis.append((1/5) * rho**5 * torch.cos(theta))    # Secondary Coma X
    basis.append((1/5) * rho**5 * torch.cos(3*theta))  # Secondary Trefoil X
    basis.append((1/5) * rho**5 * torch.cos(5*theta))  # Pentafoil 0°

    basis = torch.stack(basis, dim=0)
    return basis, mask



def fit_probe_with_aberration(probe, wavelength, NA, r_mask, device):

    N = probe.shape[-1]

    pupil = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(probe)))
    amp = torch.abs(pupil)
    phase = torch.angle(pupil)


    basis, mask = build_pupil_phase_basis(N, r_mask, NA, device)

    phase_unwrap = torch.tensor(
        unwrap_phase((phase*mask).cpu().numpy()),
        dtype=torch.float32, device=device)


    B = basis.view(basis.shape[0], -1).T
    y = phase_unwrap.flatten()
    M = mask.flatten()
    B_mask = B[M]
    y_mask = y[M]

    coeff_phase = -torch.linalg.lstsq(B_mask, y_mask).solution  # rad

    phi_fit = torch.sum(coeff_phase[:, None, None] * basis, dim=0)
    phi_fit[~mask] = 0.0
    pupil_fit = amp * torch.exp(-1j * phi_fit)
    # pupil_fit[~mask] = 0.0
    probe_fit = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(pupil_fit)))


    aberrations_m = (wavelength / (2 * torch.pi)) * coeff_phase

    aberration_names = [
        "Piston","Tilt X","Tilt Y",
        "Astigmatism 0°","Defocus","Astigmatism 45°",
        "Trefoil X","Coma X","Coma Y","Trefoil Y",
        "Quadrafoil 0°","Secondary Astig 45°","Primary Spherical",
        "Secondary Astig 0°","Quadrafoil 45°",
        "Pentafoil 45°","Secondary Trefoil Y","Secondary Coma Y",
        "Secondary Coma X","Secondary Trefoil X","Pentafoil 0°"
    ]

    aberration_symbols = [
        "C0","T_x", "T_y",
        "A1_0", "C1", "A1_45",
        "B2_X", "C3_1X", "C3_1Y", "B2_Y",
        "A3_0", "A3_45s", "C3", "A3_45", "A4_0",
        "A5_45", "B5_Y", "C5_1Y", "C5_1X", "B5_X", "A6_0"
    ]   

    return {
        "probe_fit": probe_fit,
        "phi_fit": phi_fit,
        "coeff_phase_rad": coeff_phase,          # rad
        "coeff_wavefront_m": aberrations_m,     # 米
        "mask": mask,
        "aberration_names": aberration_names,
        "aberration_symbols": aberration_symbols,
    }



def scan_position_clustering(posset_GPU, n_block, device, plot_cluster=False, mode='BKD'):

    if mode == 'Morton':        
        # Morton order spatial cluster
        pos_y = posset_GPU[:, 2].to(torch.int64)
        pos_x = posset_GPU[:, 3].to(torch.int64)
        max_coord = max(pos_y.max().item(), pos_x.max().item())
        n_bits = max(1, int(max_coord).bit_length())
        morton_code = torch.zeros_like(pos_x)
        for bit in range(n_bits):
            morton_code |= ((pos_x >> bit) & 1) << (2 * bit)
            morton_code |= ((pos_y >> bit) & 1) << (2 * bit + 1)
        position_order = torch.argsort(morton_code)
        posset_sorted = posset_GPU[position_order]
        spatial_blocks = list(torch.tensor_split(posset_sorted, n_block, dim=0))

    if mode == 'kmean':
        # K-mean clustering for scan positions
        time_0 = time.time()
        print_and_log('')
        print('\rK-mean clustering for scan positions started', end="")
        pos_xy = posset_GPU[:, 2:4].to(torch.float32)
        pos_min = pos_xy.min(dim=0).values
        pos_max = pos_xy.max(dim=0).values
        pos_scale = (pos_max - pos_min).clamp_min(1e-8)
        pos_xy_scaled = (pos_xy - pos_min) / pos_scale
        n_block_y = max(1, int(np.sqrt(n_block)))
        n_block_x = int(np.ceil(n_block / n_block_y))
        while n_block_y * n_block_x < n_block:
            n_block_y += 1
        y_centers = torch.linspace(0.5 / n_block_y, 1 - 0.5 / n_block_y, n_block_y, device=device)
        x_centers = torch.linspace(0.5 / n_block_x, 1 - 0.5 / n_block_x, n_block_x, device=device)
        cy, cx = torch.meshgrid(y_centers, x_centers, indexing='ij')
        centroids = torch.stack((cy.flatten(), cx.flatten()), dim=1)[:n_block]
        for _ in range(50):
            dist = torch.cdist(pos_xy_scaled, centroids)
            labels = torch.argmin(dist, dim=1)
            counts = torch.bincount(labels, minlength=n_block)
            empty = torch.where(counts == 0)[0]
            if empty.numel() > 0:
                largest = torch.argsort(counts, descending=True)
                for empty_id, largest_id in zip(empty.tolist(), largest.tolist()):
                    if counts[largest_id] > 1:
                        candidate = torch.where(labels == largest_id)[0]
                        farthest = candidate[torch.argmax(dist[candidate, largest_id])]
                        centroids[empty_id] = pos_xy_scaled[farthest]
                        labels[farthest] = empty_id
                        counts[largest_id] -= 1
                        counts[empty_id] += 1
            new_centroids = torch.zeros_like(centroids)
            new_centroids.index_add_(0, labels, pos_xy_scaled)
            new_centroids = new_centroids / counts.clamp_min(1).to(torch.float32).unsqueeze(1)
            if torch.allclose(centroids, new_centroids, atol=1e-5):
                break
            centroids = new_centroids
        spatial_blocks = [posset_GPU[labels == k] for k in range(n_block) if torch.any(labels == k)]
        print_and_log(f'\rK-mean clustering for scan positions finished in {time.time()-time_0:.2f} s')

    if mode == 'BKD':
        #Balanced KD partition 
        pos_xy = posset_GPU[:, 2:4].to(torch.float32) 
        N_pos = pos_xy.shape[0] 
        n_block = min(n_block, N_pos) 
        spatial_blocks = [] 
        def split_block(indices, n_parts): 
            n_points = indices.numel() 
            if n_parts == 1: 
                spatial_blocks.append(posset_GPU[indices]) 
                return 
            pos = pos_xy[indices] 
            ranges = pos.max(dim=0).values - pos.min(dim=0).values 
            axis = torch.argmax(ranges).item() 
            order = torch.argsort(pos[:, axis]) 
            indices_sorted = indices[order] 
            n_left_parts = n_parts // 2 
            n_right_parts = n_parts - n_left_parts 
            n_left = (n_points * n_left_parts) // n_parts 
            n_left = max(1, min(n_points - 1, n_left)) 
            n_right = n_points - n_left 
            split_block(indices_sorted[:n_left], n_left_parts) 
            split_block(indices_sorted[n_left:], n_right_parts) 
        indices = torch.arange(N_pos, device=posset_GPU.device) 
        split_block(indices, n_block)


    if plot_cluster:
        # plot scan positions clustering
        print('real n_block: '+str(len(spatial_blocks)))
        plt.figure(figsize=(8, 8))
        for cluster_id, block in enumerate(spatial_blocks):
            block_cpu = block[:, 2:4].detach().cpu().numpy()
            plt.scatter(block_cpu[:, 1], block_cpu[:, 0], s=3, label=f'Cluster {cluster_id}')
        plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Spatial Clustering: {n_block} Blocks')
        plt.tight_layout()
        plt.show()

    return spatial_blocks



def l2_fft_constraint(object_GPU, n_slice, l2_lambdaTikhonov):
    sy, sx = object_GPU.shape[-2:]
    ky = torch.fft.fftfreq(sy, d=1.0, device=object_GPU.device)
    kx = torch.fft.fftfreq(sx, d=1.0, device=object_GPU.device)
    ky, kx = torch.meshgrid(ky, kx, indexing='ij')
    k2 = 4*torch.sin(torch.pi*ky)**2 + 4*torch.sin(torch.pi*kx)**2

    for sn in range(n_slice):
        objSlice = object_GPU[sn]
        objfft = fft2(objSlice)
        objfft_abs = objfft.abs()

        objfft_abs = objfft_abs / (1 + l2_lambdaTikhonov*k2)

        objfft = objfft_abs*torch.exp(1j*objfft.angle())
        object_GPU[sn] = ifft2(objfft)

    return object_GPU



def l1_fft_constraint(object_GPU, n_slice, l1_softThreshold):
    for sn in range(n_slice):
        objSlice = object_GPU[sn]
        objfft = fft2(objSlice)
        objfft_abs = objfft.abs()

        objfft_abs_shift = torch.fft.fftshift(objfft_abs)
        objfft_abs_shift[0,0] = 0

        objfft_abs -= l1_softThreshold*objfft_abs_shift.mean()
        objfft_abs[objfft_abs<=0] = 0

        objfft = objfft_abs*torch.exp(1j*objfft.angle())
        object_GPU[sn] = ifft2(objfft)

    return object_GPU



def l0_fft_constraint(object_GPU, n_slice, l0_hardThreshold):
    for sn in range(n_slice):
        objSlice = object_GPU[sn]
        objfft = fft2(objSlice)
        objfft_abs = objfft.abs()

        hardThreshold = l0_hardThreshold*objfft_abs.mean()
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



def LSQML_engine(iter_max, s_O, s_P,
                    data_4D, Voltage, alpha, aperture_radius, 
                    posset, proben0, objectn, slice_thickness, k_theta,
                    device=None,
                    n_block=None, 
                    position_clustering=False, 
                    plot_cluster = False,
                    e_f=1e-9, 
                    e_g=1e-1, 
                    e_LSQ=5e-1,
                    probe_orthog_constr=False, 
                    sorting_probe=True, 
                    probe_fit=False,
                    s_position_correction=0,
                    pc_start_iteration=None,
                    pc_mode='intensity',         #'intensity' or 'object'
                    POA=False,
                    l2_fft_lambdaTikhonov=0, 
                    l1_fft_softThreshold=0, 
                    l0_fft_hardThreshold=0, 
                    kz_regularization=0, 
                    rh_positive_phase=False, 
                    FFT_phase_offset=0,
                    ):

    sy, sx, dy, dx = data_4D.shape
    n_state = proben0.shape[0]
    n_slice = objectn.shape[0]
    N_pos = sy * sx

    if n_block is None:
        n_block = sy

    n_block = min(n_block, N_pos)

    if pc_start_iteration is None:
        pc_start_iteration = 0.25*iter_max

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print_and_log(f'Slice Thickness: {slice_thickness} Å')
    print_and_log('')
    print_and_log(f'############### LSQML Iteration Parameters ###############')
    print_and_log(f"Device Using: {device}")
    print_and_log(f'Number of Iterations: {iter_max}')
    print_and_log(f'Step for Position Correction: {s_position_correction}')
    if s_position_correction>0:
        print_and_log(f'Mode for Position Correction: {pc_mode}')
        print_and_log(f'Start for Position Correction: {pc_start_iteration}th iteration')
    print_and_log(f'Step for Updating Object: {s_O}')
    print_and_log(f'Step for Updating Probe: {s_P}')
    print_and_log(f'Number of Blocks: {n_block}')
    print_and_log(f'Average Block Size: {N_pos / n_block:.2f} positions')
    print_and_log(f'Spatial Position Clustering: {position_clustering}')
    print_and_log(f'Epsilon for Recipro Optim: {e_f}')
    print_and_log(f'Epsilon for Real Optim: {e_g}')
    print_and_log(f'Epsilon for LSQ: {e_LSQ}')
    print_and_log(f'Probe Orthog Constraint: {probe_orthog_constr}')
    print_and_log(f'Sorting Probe by Energy: {sorting_probe}')
    print_and_log(f'Phase Object Approximation: {POA}')
    print_and_log(f'Object Positive Phase: {rh_positive_phase}')
    print_and_log(f'Object FFT LambdaTikhonov (L2): {l2_fft_lambdaTikhonov}')
    print_and_log(f'Object FFT SoftThreshold (L1): {l1_fft_softThreshold}')
    print_and_log(f'Object FFT HardThreshold (L0): {l0_fft_hardThreshold}')
    print_and_log(f'Object kz Regularization: {kz_regularization}')
    print_and_log(f'Object FFT phase offset: {FFT_phase_offset}')
    
    wavelength = calculate_wavelength(Voltage)
    propagators = np.fft.fftshift(np.exp(-1j*2*np.pi/wavelength*0.5*slice_thickness*k_theta**2))

    data_4D[data_4D < 0] = 0
    data_4D_sqrt_GPU = torch.from_numpy(np.reshape(np.sqrt(data_4D), (N_pos, dy, dx))).to(device)
    proben_GPU = torch.from_numpy(proben0).to(device).to(dtype=torch.complex64)
    object_GPU = torch.from_numpy(objectn).to(device).to(dtype=torch.complex64)
    posset_GPU = torch.from_numpy(posset).to(device)
    propagators_GPU = torch.from_numpy(propagators).to(device).to(dtype=torch.complex64)

    _, Oy, Ox = object_GPU.shape
    y_ind = torch.arange(dy, device=device, dtype=torch.int32)
    x_ind = torch.arange(dx, device=device, dtype=torch.int32)

    err = torch.zeros(iter_max, device=device)
    Prb_LSQ_step = torch.zeros((iter_max, n_state), device=device)
    Obj_LSQ_step = torch.zeros((iter_max, n_slice), device=device)
    pc_shift = torch.zeros(iter_max, device=device)

    qmy, qmx = torch.meshgrid(fftfreq(dx, device=device), fftfreq(dy, device=device), indexing='ij')
    qy = fftfreq(Oy, device=device)
    qx = fftfreq(Ox, device=device)
    qz = fftfreq(n_slice, device=device)
    qza, qya, qxa = torch.meshgrid(qz, qy, qx, indexing="ij")
    qz2 = (qza * kz_regularization) ** 2
    qr2 = qxa ** 2 + qya ** 2
    Wz = 1 - 2 / torch.pi * torch.arctan2(qz2, qr2)

    if position_clustering:
        spatial_blocks = scan_position_clustering(posset_GPU, n_block, device=device, plot_cluster=plot_cluster)


    start_time = time.time()
    print_and_log('')
    print_and_log(f'###################### LSQML Process #####################')
    print("\rLSQML progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0, "*" * 0, "." * 10, 0, 0, 0), end="")

    for i in range(iter_max):
        time_i0 = time.time()
        err_u = 0
        err_d = 0

        # if i == iter_max//2:
        #     s_P = 0
        #     object_GPU = torch.from_numpy(objectn).to(device).to(dtype=torch.complex64)

        if position_clustering:
            block_order = torch.randperm(n_block).tolist()
            posset_blocks = [spatial_blocks[j] for j in block_order]
        else:
            idx = torch.randperm(posset_GPU.shape[0], device=device)
            posset_shuffled = posset_GPU[idx]
            posset_blocks = list(torch.tensor_split(posset_shuffled, n_block, dim=0))

        if probe_orthog_constr:
            proben_GPU = probe_orthogonalization(proben_GPU, n_state, device=device)

        if sorting_probe:
            intensities = proben_GPU.abs().pow(2).sum(dim=(-2, -1))
            intensities_order = torch.argsort(intensities, descending=True)
            proben_GPU = proben_GPU[intensities_order]

        if i == 0 or (i + 1) % 10 == 0:
            gObj_d = torch.zeros((Oy, Ox), dtype=torch.complex64, device=device)

            for pos_block in posset_blocks:
                b_size = pos_block.shape[0]
                ind_y = (pos_block[:, 2][:, None, None] + y_ind[None, :, None]).to(dtype=torch.int32)#.long()
                ind_x = (pos_block[:, 3][:, None, None] + x_ind[None, None, :]).to(dtype=torch.int32)#.long()
                indice = ind_y * Ox + ind_x
                proben_i = proben_GPU[0].abs().pow(2).unsqueeze(0).expand(b_size, dy, dx).to(torch.complex64)
                gObj_d = gObj_d.ravel().index_add_(0, indice.ravel(), proben_i.ravel()).reshape((Oy, Ox))

        for pos_block in posset_blocks:
            b_size = pos_block.shape[0]
            poss_1D = pos_block[:, 4].to(dtype=torch.int64)
            ind_y = (pos_block[:, 2][:, None, None] + y_ind[None, :, None]).to(dtype=torch.int32)#.long()
            ind_x = (pos_block[:, 3][:, None, None] + x_ind[None, None, :]).to(dtype=torch.int32)#.long()
            indice = ind_y * Ox + ind_x
            Illuminated_patches = object_GPU[:, ind_y, ind_x]
            amplitudes_patches = data_4D_sqrt_GPU[poss_1D]

            Prb_ms = []
            Prb_ms.append(proben_GPU.unsqueeze(1))

            for s in range(n_slice):
                psi = Prb_ms[s] * Illuminated_patches[s].unsqueeze(0)

                if s != n_slice - 1:
                    psifft = fft2(psi, dim=(2, 3))
                    psifft *= propagators_GPU[None, None, :, :]
                    Prb_ms.append(ifft2(psifft, dim=(2, 3)))

            psi_fft = fftshift(fft2(psi, dim=(2, 3)), dim=(2, 3))
            a_psi_fft = psi_fft.abs().pow(2).sum(dim=0).sqrt()  

            # Euclidean
            chi_fft_R = amplitudes_patches / (a_psi_fft + e_f) - 1

            # Gaussian
            # chi_fft_R = amplitudes_patches.pow(2) / (a_psi_fft.pow(2) + e_f) - 1

            # Poisson
            # chi_fft_R = 1 - a_psi_fft.pow(2) / (amplitudes_patches.pow(2) + e_f)

            chi_fft = psi_fft * chi_fft_R.unsqueeze(0).to(torch.complex64)
            chi = ifft2(ifftshift(chi_fft, dim=(2, 3)), dim=(2, 3))
                        
            for sr in range(n_slice - 1, -1, -1):
                if sr != n_slice - 1:
                    chi_fft = fft2(chi, dim=(2, 3))
                    chi_fft /= propagators_GPU[None, None, :, :]
                    chi = ifft2(chi_fft, dim=(2, 3))

                gObj_i = chi[0] * Prb_ms[sr][0].conj()
                gObj_u = torch.zeros((Oy, Ox), dtype=torch.complex64, device=device)
                gObj_u = gObj_u.ravel().index_add_(0, indice.ravel(), gObj_i.ravel()).reshape((Oy, Ox))
                gObj = gObj_u / (gObj_d**2 + (e_g * gObj_d.abs().max())**2).sqrt()
                #gObj = gObj_u / (gObj_d  + e_g * gObj_d.abs().max())
                #gObj = gObj_u / (gObj_d_i**2 + (e_g * gObj_d_i.abs().max())**2).sqrt()

                gObj_patches = gObj[ind_y, ind_x]
                gOP = gObj_patches * Prb_ms[sr].sum(dim=0)
                Mr_0 = (gOP.conj() * chi.sum(dim=0)).real.sum(dim=(1, 2))
                Ml_00 = gOP.abs().pow(2).sum(dim=(1, 2))
                Ml_00 += e_LSQ * Ml_00.mean()
                a_O = (Mr_0 / Ml_00 * s_O).mean() / n_slice
                Obj_LSQ_step[i][sr] += a_O / n_block
                object_GPU[sr] += a_O * gObj

                gPrb_i = chi * Illuminated_patches[sr].unsqueeze(0).conj()

                if sr == 0:
                    gPrb_u = gPrb_i.sum(dim=1)
                    gPrb_d = Illuminated_patches[sr].abs().pow(2).sum(dim=0).unsqueeze(0)
                    #gPrb = gPrb_u / (gPrb_d**2 + (e_g * gPrb_d.max())**2).sqrt()
                    #gPrb = gPrb_u / (gPrb_d + e_g * gPrb_d.max())
                    gPrb = gPrb_u / gPrb_d

                    gPO = gPrb.unsqueeze(1) * Illuminated_patches[sr].unsqueeze(0)
                    Mr_1 = (gPO.conj() * chi).real.sum(dim=(2, 3))
                    Ml_11 = gPO.abs().pow(2).sum(dim=(2, 3))
                    Ml_11 += e_LSQ * Ml_11.mean()
                    a_P = (Mr_1 / Ml_11).mean(dim=1) * s_P
                    Prb_LSQ_step[i] += a_P / n_block
                    proben_GPU += a_P[:, None, None] * gPrb

                chi = gPrb_i


                if i >= pc_start_iteration and sr == n_slice // 2 and s_position_correction and pc_mode == 'object':

                    I_patches_fft = fft2(Illuminated_patches[sr], dim=(1, 2))
                    I_patches_gx = ifft2(2j * torch.pi * I_patches_fft * qmx[None, :, :], dim=(1, 2))
                    I_patches_gy = ifft2(2j * torch.pi * I_patches_fft * qmy[None, :, :], dim=(1, 2))
                    gxP = I_patches_gx * Prb_ms[sr][0]
                    Mr_x = (gxP.conj() * chi[0]).real.sum(dim=(1, 2))
                    Ml_x = gxP.abs().pow(2).sum(dim=(1, 2))
                    gx = s_position_correction * (Mr_x / Ml_x).real
                    gyP = I_patches_gy * Prb_ms[sr][0]
                    Mr_y = (gyP.conj() * chi[0]).real.sum(dim=(1, 2))
                    Ml_y = gyP.abs().pow(2).sum(dim=(1, 2))
                    gy = s_position_correction * (Mr_y / Ml_y).real
                    pos_block[:, 2] += gy
                    pos_block[:, 3] += gx
                    pc_shift[i] += (gy ** 2 + gx ** 2).sqrt().sum()


            if i >= pc_start_iteration and s_position_correction and pc_mode == 'intensity':

                I_cal = psi_fft.abs().pow(2).sum(dim=0)
                I_exp = amplitudes_patches.pow(2)
                dI = I_exp - I_cal
                I_patches_fft = fft2(Illuminated_patches[-1], dim=(1, 2))

                I_patches_gx = ifft2(2j * torch.pi * I_patches_fft * qmx[None, :, :], dim=(1, 2))
                I_patches_gy = ifft2(2j * torch.pi * I_patches_fft * qmy[None, :, :], dim=(1, 2))

                gxP = I_patches_gx * Prb_ms[-1]
                gyP = I_patches_gy * Prb_ms[-1]

                gxP_fft = fftshift(fft2(gxP, dim=(2, 3)), dim=(2, 3))
                gyP_fft = fftshift(fft2(gyP, dim=(2, 3)), dim=(2, 3))

                dI_dx = 2 * (psi_fft.conj() * gxP_fft).real.sum(dim=0)
                dI_dy = 2 * (psi_fft.conj() * gyP_fft).real.sum(dim=0)

                Mr_x = (dI * dI_dx).sum(dim=(1, 2))
                Ml_x = dI_dx.pow(2).sum(dim=(1, 2))
                gx = s_position_correction * Mr_x / (Ml_x + 1e-12)

                Mr_y = (dI * dI_dy).sum(dim=(1, 2))
                Ml_y = dI_dy.pow(2).sum(dim=(1, 2))
                gy = s_position_correction * Mr_y / (Ml_y + 1e-12)

                pos_block[:, 2] += gy
                pos_block[:, 3] += gx
                pc_shift[i] += (gy.pow(2) + gx.pow(2)).sqrt().sum()

            err_u += (amplitudes_patches - (psi_fft.abs().pow(2).sum(dim=0)).sqrt()).pow(2).sum()
            err_d += amplitudes_patches.to(torch.float32).pow(2).sum()

        if torch.isnan(err_u / err_d):
            if i > 1:
                plt.figure(figsize=(6,4))
                plt.semilogy(range(len(err[1:i].cpu())), err[1:i].cpu())
                plt.xlabel("Iteration")
                plt.ylabel("Error")
                plt.title("Convergence Curve")
                plt.tight_layout()

            raise ValueError("err is NaN")
        
        err[i] = err_u / err_d
        posset_GPU = torch.cat(posset_blocks, dim=0)

        if probe_fit:
            for ns in range(n_state):
                proben_GPU[ns] = fit_probe_with_aberration(proben_GPU[ns], wavelength, alpha, aperture_radius, device=device)["probe_fit"]

        if l2_fft_lambdaTikhonov > 0:
            object_GPU = l2_fft_constraint(object_GPU, n_slice, l2_fft_lambdaTikhonov)

        if l1_fft_softThreshold > 0:
            object_GPU = l1_fft_constraint(object_GPU, n_slice, l1_fft_softThreshold)

        if l0_fft_hardThreshold > 0:
            object_GPU = l0_fft_constraint(object_GPU, n_slice, l0_fft_hardThreshold)

        if kz_regularization > 0:
            object_GPU = kz_constraint(object_GPU, Wz)

        if FFT_phase_offset > 0 and i > iter_max * 2 / 3:
            object_GPU = FFT_phase0_offset(object_GPU, n_slice, FFT_phase_offset)

        if rh_positive_phase:
            object_GPU = rh_constraint(object_GPU, n_slice, rh_hardThreshold=0)

        if POA:
            object_GPU = POA_constraint(object_GPU)


        time_is = time.time() - time_i0
        speed = 1 / time_is
        process = (i + 1) / iter_max * 100
        aa = "*" * int(process / 10)
        bb = "." * (10 - int(process / 10))
        dur = int(time.time() - start_time)
        time_remain = int((iter_max - i - 1) * time_is)
        dur_h = dur // 3600
        dur_m = (dur - dur_h * 3600) // 60
        dur_s = dur - dur_h * 3600 - dur_m * 60
        time_remain_h = time_remain // 3600
        time_remain_m = (time_remain - time_remain_h * 3600) // 60
        time_remain_s = time_remain - time_remain_h * 3600 - time_remain_m * 60

        print("\rLSQ-3ML progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process, aa, bb, speed, dur_h, dur_m, dur_s, time_remain_h, time_remain_m, time_remain_s), end="")

    print('')
    gpu_time = time.time() - start_time

    msLSQML_Obj = np.array(object_GPU.cpu())
    msLSQML_Prb = np.array(proben_GPU.cpu())
    Prb_EW = np.array(psi[0][0].cpu())
    msLSQML_err = np.array(err.cpu())
    Obj_LSQ_step = np.array(Obj_LSQ_step.cpu())
    Prb_LSQ_step = np.array(Prb_LSQ_step.cpu())
    pc_mean_shift = np.array((pc_shift/sy/sx).cpu())
    posset = np.array(posset_GPU.cpu())

    log_to_file("LSQML progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process, aa, bb, speed, dur_h, dur_m, dur_s, time_remain_h, time_remain_m, time_remain_s))
    print_and_log(f'LSQML process finished in {gpu_time} s')

    if probe_fit:
    # if 1:
        print_and_log(f'')
        print_and_log(f'################# probe fitting results ##################')
        probe_fit_results = fit_probe_with_aberration(proben_GPU[0], wavelength, alpha, aperture_radius, device=device)
        print_and_log("    Fitted Aberration:")
        for i_aberr, (symbol, name, coeff_m) in enumerate(zip(probe_fit_results["aberration_symbols"],probe_fit_results["aberration_names"],probe_fit_results["coeff_wavefront_m"])):
            print_and_log(f"    {(i_aberr):02d}. {symbol:8s} | {name:28s} = {(coeff_m).item():+.3e} A")

    return msLSQML_Obj, msLSQML_Prb, Prb_EW, msLSQML_err, Obj_LSQ_step, Prb_LSQ_step, pc_mean_shift, posset




def ePIE_engine(iter_max, s_O, s_P,
                    data_4D, Voltage, alpha, aperture_radius, 
                    posset, proben0, objectn, slice_thickness, k_theta,
                    device=None,
                    n_block=None, 
                    position_clustering=False, 
                    plot_cluster = False,
                    e_f=1e-9,
                    alpha_O=1,
                    alpha_P=1,
                    probe_orthog_constr=False, 
                    sorting_probe=True, 
                    probe_fit=False,
                    s_position_correction=0,
                    pc_start_iteration=None,
                    pc_mode='intensity',         #'intensity' or 'object'
                    POA=False,
                    l2_fft_lambdaTikhonov=0, 
                    l1_fft_softThreshold=0, 
                    l0_fft_hardThreshold=0, 
                    kz_regularization=0, 
                    rh_positive_phase=False, 
                    FFT_phase_offset=0,
                    ):

    sy, sx, dy, dx = data_4D.shape
    n_state = proben0.shape[0]
    n_slice = objectn.shape[0]
    N_pos = sy * sx

    if n_block is None:
        n_block = sy

    n_block = min(n_block, N_pos)

    if pc_start_iteration is None:
        pc_start_iteration = 0.25*iter_max

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print_and_log(f'Slice Thickness: {slice_thickness} Å')
    print_and_log('')
    print_and_log(f'############### ePIE Iteration Parameters ################')
    print_and_log(f"Device Using: {device}")
    print_and_log(f'Number of Iterations: {iter_max}')
    print_and_log(f'Step for Position Correction: {s_position_correction}')
    if s_position_correction>0:
        print_and_log(f'Mode for Position Correction: {pc_mode}')
        print_and_log(f'Start for Position Correction: {pc_start_iteration}th iteration')
    print_and_log(f'Step for Updating Object: {s_O}')
    print_and_log(f'Step for Updating Probe: {s_P}')
    print_and_log(f'Number of Blocks: {n_block}')
    print_and_log(f'Average Block Size: {N_pos / n_block:.2f} positions')
    print_and_log(f'Spatial Position Clustering: {position_clustering}')
    print_and_log(f'Epsilon for Recipro Optim: {e_f}')
    print_and_log(f'rPIE-alpha for Object: {alpha_O }')
    print_and_log(f'rPIE-alpha for Probe: {alpha_P}')
    print_and_log(f'Probe Orthog Constraint: {probe_orthog_constr}')
    print_and_log(f'Sorting Probe by Energy: {sorting_probe}')
    print_and_log(f'Phase Object Approximation: {POA}')
    print_and_log(f'Object Positive Phase: {rh_positive_phase}')
    print_and_log(f'Object FFT LambdaTikhonov (L2): {l2_fft_lambdaTikhonov}')
    print_and_log(f'Object FFT SoftThreshold (L1): {l1_fft_softThreshold}')
    print_and_log(f'Object FFT HardThreshold (L0): {l0_fft_hardThreshold}')
    print_and_log(f'Object kz Regularization: {kz_regularization}')
    print_and_log(f'Object FFT phase offset: {FFT_phase_offset}')
    
    wavelength = calculate_wavelength(Voltage)
    propagators = np.fft.fftshift(np.exp(-1j*2*np.pi/wavelength*0.5*slice_thickness*k_theta**2))

    data_4D[data_4D < 0] = 0
    data_4D_sqrt_GPU = torch.from_numpy(np.reshape(np.sqrt(data_4D), (N_pos, dy, dx))).to(device)
    proben_GPU = torch.from_numpy(proben0).to(device).to(dtype=torch.complex64)
    object_GPU = torch.from_numpy(objectn).to(device).to(dtype=torch.complex64)
    posset_GPU = torch.from_numpy(posset).to(device)
    propagators_GPU = torch.from_numpy(propagators).to(device).to(dtype=torch.complex64)

    _, Oy, Ox = object_GPU.shape
    y_ind = torch.arange(dy, device=device, dtype=torch.int32)
    x_ind = torch.arange(dx, device=device, dtype=torch.int32)

    err = torch.zeros(iter_max, device=device)
    pc_shift = torch.zeros(iter_max, device=device)

    qmy, qmx = torch.meshgrid(fftfreq(dx, device=device), fftfreq(dy, device=device), indexing='ij')
    qy = fftfreq(Oy, device=device)
    qx = fftfreq(Ox, device=device)
    qz = fftfreq(n_slice, device=device)
    qza, qya, qxa = torch.meshgrid(qz, qy, qx, indexing="ij")
    qz2 = (qza * kz_regularization) ** 2
    qr2 = qxa ** 2 + qya ** 2
    Wz = 1 - 2 / torch.pi * torch.arctan2(qz2, qr2)

    if position_clustering:
        spatial_blocks = scan_position_clustering(posset_GPU, n_block, device=device, plot_cluster=plot_cluster)


    start_time = time.time()
    print_and_log('')
    print_and_log(f'###################### ePIE Process ######################')
    print("\rePIE progressing: {:^3.0f}%[{}->{}] ?iter/s ({:0>2}:{:0>2}:{:0>2}<??:??:??)".format(0, "*" * 0, "." * 10, 0, 0, 0), end="")

    for i in range(iter_max):
        time_i0 = time.time()
        err_u = 0
        err_d = 0

        if position_clustering:
            block_order = torch.randperm(n_block).tolist()
            posset_blocks = [spatial_blocks[j] for j in block_order]
        else:
            idx = torch.randperm(posset_GPU.shape[0], device=device)
            posset_shuffled = posset_GPU[idx]
            posset_blocks = list(torch.tensor_split(posset_shuffled, n_block, dim=0))

        if probe_orthog_constr:
            proben_GPU = probe_orthogonalization(proben_GPU, n_state, device=device)

        if sorting_probe:
            intensities = proben_GPU.abs().pow(2).sum(dim=(-2, -1))
            intensities_order = torch.argsort(intensities, descending=True)
            proben_GPU = proben_GPU[intensities_order]

        for pos_block in posset_blocks:
            b_size = pos_block.shape[0]

            poss_1D = pos_block[:, 4].to(dtype=torch.int64)

            ind_y = (pos_block[:, 2][:, None, None] + y_ind[None, :, None]).to(dtype=torch.int32)
            ind_x = (pos_block[:, 3][:, None, None] + x_ind[None, None, :]).to(dtype=torch.int32)
            indice = ind_y * Ox + ind_x

            Illuminated_patches = object_GPU[:, ind_y, ind_x]
            amplitudes_patches = data_4D_sqrt_GPU[poss_1D]

            Prb_ms = []
            Prb_ms.append(proben_GPU.unsqueeze(1).expand(-1, b_size, -1, -1))

            for s in range(n_slice):
                psi = Prb_ms[s] * Illuminated_patches[s].unsqueeze(0)

                if s != n_slice - 1:
                    psifft = fft2(psi, dim=(2, 3))
                    psifft *= propagators_GPU[None, None, :, :]
                    Prb_ms.append(ifft2(psifft, dim=(2, 3)))

            psi_fft = fftshift(fft2(psi, dim=(2, 3)), dim=(2, 3))
            a_psi_fft = psi_fft.abs().pow(2).sum(dim=0).sqrt()
            chi_fft_R = amplitudes_patches / (a_psi_fft + e_f) - 1
            chi_fft = psi_fft * chi_fft_R.unsqueeze(0).to(torch.complex64)
            chi = ifft2(ifftshift(chi_fft, dim=(2, 3)), dim=(2, 3))

            for sr in range(n_slice - 1, -1, -1):
                if sr != n_slice - 1:
                    chi_fft = fft2(chi, dim=(2, 3))
                    chi_fft /= propagators_GPU[None, None, :, :]
                    chi = ifft2(chi_fft, dim=(2, 3))

                P_abs2 = Prb_ms[sr][0].abs().pow(2)
                P_max = P_abs2.amax(dim=(-2, -1), keepdim=True)
                P_norm = (1 - alpha_O) * P_abs2 + alpha_O * P_max
                dObj_i = chi[0] * Prb_ms[sr][0].conj() / P_norm
                dObj = torch.zeros((Oy, Ox), dtype=torch.complex64, device=device)
                dObj = dObj.ravel().index_add_(0, indice.ravel(), dObj_i.ravel()).reshape((Oy, Ox))
                object_GPU[sr] += s_O * dObj / b_size

                gPrb_i = chi * Illuminated_patches[sr].unsqueeze(0).conj()

                if sr == 0:
                    O_abs2 = Illuminated_patches[sr].abs().pow(2)
                    O_max = O_abs2.amax(dim=(-2, -1), keepdim=True)
                    O_norm = (1 - alpha_P) * O_abs2 + alpha_P * O_max
                    dPrb_i = gPrb_i / O_norm.unsqueeze(0)
                    dPrb = dPrb_i.sum(dim=1)
                    proben_GPU += s_P * dPrb / b_size

                chi = gPrb_i

                if i >= pc_start_iteration and sr == n_slice // 2 and s_position_correction and pc_mode == 'object':

                    I_patches_fft = fft2(Illuminated_patches[sr], dim=(1, 2))
                    I_patches_gx = ifft2(2j * torch.pi * I_patches_fft * qmx[None, :, :], dim=(1, 2))
                    I_patches_gy = ifft2(2j * torch.pi * I_patches_fft * qmy[None, :, :], dim=(1, 2))

                    gxP = I_patches_gx * Prb_ms[sr][0]
                    Mr_x = (gxP.conj() * chi[0]).real.sum(dim=(1, 2))
                    Ml_x = gxP.abs().pow(2).sum(dim=(1, 2))
                    gx = s_position_correction * (Mr_x / Ml_x).real

                    gyP = I_patches_gy * Prb_ms[sr][0]
                    Mr_y = (gyP.conj() * chi[0]).real.sum(dim=(1, 2))
                    Ml_y = gyP.abs().pow(2).sum(dim=(1, 2))
                    gy = s_position_correction * (Mr_y / Ml_y).real

                    pos_block[:, 2] += gy
                    pos_block[:, 3] += gx
                    pc_shift[i] += (gy ** 2 + gx ** 2).sqrt().sum()

            if i >= pc_start_iteration and s_position_correction and pc_mode == 'intensity':

                I_cal = psi_fft.abs().pow(2).sum(dim=0)
                I_exp = amplitudes_patches.pow(2)
                dI = I_exp - I_cal

                I_patches_fft = fft2(Illuminated_patches[-1], dim=(1, 2))

                I_patches_gx = ifft2(2j * torch.pi * I_patches_fft * qmx[None, :, :], dim=(1, 2))
                I_patches_gy = ifft2(2j * torch.pi * I_patches_fft * qmy[None, :, :], dim=(1, 2))

                gxP = I_patches_gx * Prb_ms[-1]
                gyP = I_patches_gy * Prb_ms[-1]

                gxP_fft = fftshift(fft2(gxP, dim=(2, 3)), dim=(2, 3))
                gyP_fft = fftshift(fft2(gyP, dim=(2, 3)), dim=(2, 3))

                dI_dx = 2 * (psi_fft.conj() * gxP_fft).real.sum(dim=0)
                dI_dy = 2 * (psi_fft.conj() * gyP_fft).real.sum(dim=0)

                Mr_x = (dI * dI_dx).sum(dim=(1, 2))
                Ml_x = dI_dx.pow(2).sum(dim=(1, 2))
                gx = s_position_correction * Mr_x / (Ml_x + 1e-12)

                Mr_y = (dI * dI_dy).sum(dim=(1, 2))
                Ml_y = dI_dy.pow(2).sum(dim=(1, 2))
                gy = s_position_correction * Mr_y / (Ml_y + 1e-12)

                pos_block[:, 2] += gy
                pos_block[:, 3] += gx
                pc_shift[i] += (gy.pow(2) + gx.pow(2)).sqrt().sum()

            err_u += (amplitudes_patches - (psi_fft.abs().pow(2).sum(dim=0)).sqrt()).pow(2).sum()
            err_d += amplitudes_patches.to(torch.float32).pow(2).sum()

        if torch.isnan(err_u / err_d):
            if i > 1:
                plt.figure(figsize=(6,4))
                plt.semilogy(range(len(err[1:i].cpu())), err[1:i].cpu())
                plt.xlabel("Iteration")
                plt.ylabel("Error")
                plt.title("Convergence Curve")
                plt.tight_layout()

            raise ValueError("err is NaN")

        err[i] = err_u / err_d
        posset_GPU = torch.cat(posset_blocks, dim=0)

        if probe_fit:
            for ns in range(n_state):
                proben_GPU[ns] = fit_probe_with_aberration(proben_GPU[ns], wavelength, alpha, aperture_radius, device=device)["probe_fit"]

        if l2_fft_lambdaTikhonov > 0:
            object_GPU = l2_fft_constraint(object_GPU, n_slice, l2_fft_lambdaTikhonov)

        if l1_fft_softThreshold > 0:
            object_GPU = l1_fft_constraint(object_GPU, n_slice, l1_fft_softThreshold)

        if l0_fft_hardThreshold > 0:
            object_GPU = l0_fft_constraint(object_GPU, n_slice, l0_fft_hardThreshold)

        if kz_regularization > 0:
            object_GPU = kz_constraint(object_GPU, Wz)
            
        if FFT_phase_offset > 0 and i > iter_max * 2 / 3:
            object_GPU = FFT_phase0_offset(object_GPU, n_slice, FFT_phase_offset)

        if rh_positive_phase:
            object_GPU = rh_constraint(object_GPU, n_slice, rh_hardThreshold=0)

        if POA:
            object_GPU = POA_constraint(object_GPU)


        time_is = time.time() - time_i0
        speed = 1 / time_is
        process = (i + 1) / iter_max * 100
        aa = "*" * int(process / 10)
        bb = "." * (10 - int(process / 10))
        dur = int(time.time() - start_time)
        time_remain = int((iter_max - i - 1) * time_is)
        dur_h = dur // 3600
        dur_m = (dur - dur_h * 3600) // 60
        dur_s = dur - dur_h * 3600 - dur_m * 60
        time_remain_h = time_remain // 3600
        time_remain_m = (time_remain - time_remain_h * 3600) // 60
        time_remain_s = time_remain - time_remain_h * 3600 - time_remain_m * 60

        print("\rePIE progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process, aa, bb, speed, dur_h, dur_m, dur_s, time_remain_h, time_remain_m, time_remain_s), end="")

    print('')
    gpu_time = time.time() - start_time

    msLSQML_Obj = np.array(object_GPU.cpu())
    msLSQML_Prb = np.array(proben_GPU.cpu())
    # Prb_EW = np.array(psi[0][0].cpu())
    Prb_EW = np.array(psi[0].mean(axis=0).cpu())
    msLSQML_err = np.array(err.cpu())
    pc_mean_shift = np.array((pc_shift/sy/sx).cpu())
    posset = np.array(posset_GPU.cpu())

    log_to_file("ePIE progressing: {:^3.0f}%[{}->{}] {:.2f}iter/s ({:0>2}:{:0>2}:{:0>2}<{:0>2}:{:0>2}:{:0>2})".format(process, aa, bb, speed, dur_h, dur_m, dur_s, time_remain_h, time_remain_m, time_remain_s))
    print_and_log(f'ePIE process finished in {gpu_time} s')

    if probe_fit:
        print_and_log(f'')
        print_and_log(f'################# probe fitting results ##################')
        probe_fit_results = fit_probe_with_aberration(proben_GPU[0], wavelength, alpha, aperture_radius, device=device)
        print_and_log("    Fitted Aberration:")
        for i_aberr, (symbol, name, coeff_m) in enumerate(zip(probe_fit_results["aberration_symbols"],probe_fit_results["aberration_names"],probe_fit_results["coeff_wavefront_m"])):
            print_and_log(f"    {(i_aberr):02d}. {symbol:8s} | {name:28s} = {(coeff_m).item():+.3e} A")

    return msLSQML_Obj, msLSQML_Prb, Prb_EW, msLSQML_err, pc_mean_shift, posset
