"""
Author: Zeyu Wang
Email: zywang209@gmail.com
Date: September 2025
Description: "results plotting and saving"
"""

import os
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tifffile

from .utils import get_subplot_grid, RGB_Complex_Plot, print_and_log, Power_Spectrum_Log




def curl_curve_plot(test_angle_range, mean_curl, mean_curl_flip, save_results=True):
    
    _LOG_DIR = print_and_log("curl curve saving")

    plt.plot(test_angle_range, mean_curl, label='still')
    plt.plot(test_angle_range, mean_curl_flip, label='flip x')
    plt.title('Curl along Rotate Angle')
    plt.legend()

    if save_results:
        outfile = os.path.join(_LOG_DIR, "Curl_Curve.png")
        plt.savefig(outfile, dpi=300)

    plt.show()



def iterPtycho_error_plot(err, save_results=True):

    plt.figure(figsize=(6,4))
    plt.semilogy(range(len(err[1:])), err[1:])

    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Convergence Curve")

    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()

    _LOG_DIR = print_and_log("Error saving")

    if save_results:
        np.save(os.path.join(_LOG_DIR, "convergence_curve.npy"), err)
        outfile = os.path.join(_LOG_DIR, "convergence_curve.png")
        plt.savefig(outfile, dpi=300)

    plt.show()


def iterPtycho_LSQ_step_plot(Obj_LSQ_step, Prb_LSQ_step, save_results=True):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(Obj_LSQ_step)
    axes[0].set_title('object LSQ step')
    axes[0].legend(['1st slice step', '2nd slice step', '3rd slice step', '...th Slice step'])

    axes[1].plot(Prb_LSQ_step)
    axes[1].set_title('probe LSQ step')
    axes[1].legend(['1st Prb step', '2nd Prb step', '3rd Prb step', '...th Prb step'])

    plt.tight_layout()

    _LOG_DIR = print_and_log("LSQ step saving")
    if save_results:
        #np.save(os.path.join(_LOG_DIR, "Object_LSQ_step.npy"), Obj_LSQ_step)
        #np.save(os.path.join(_LOG_DIR, "Probe_LSQ_step.npy"), Prb_LSQ_step)
        outfile = os.path.join(_LOG_DIR, "LSQ_step.png")
        fig.savefig(outfile, dpi=300)  

    plt.show()



def iterPtycho_proben_plot(proben, proben0, scan_rotation_angle, scan_flip, data_shape,
                           show_probe_comparison = False,
                           show_proben_Amp = True,
                           show_proben_Phase = False,
                           save_results = True,
                           ):

    *_, dy, dx = data_shape
    n_state = proben.shape[0]
    probe_shown_rows, probe_shown_cols = get_subplot_grid(n_state)

    _LOG_DIR = print_and_log("Probe saving")
    if save_results:
        np.save(os.path.join(_LOG_DIR, "Probe_Complex.npy"), proben)


    if scan_flip:
        proben = np.flip(proben, axis=2)
        scan_rotation_angle *= -1
    proben = rotate(proben, angle=scan_rotation_angle, axes=(1,2))
    dy_r, dx_r = proben[0].shape
    proben_shown = proben[:, int(dy_r/2)-int(dy/2):int(dy_r/2)+int(dy/2),int(dx_r/2)-int(dx/2):int(dx_r/2)+int(dx/2)]

    energy_n = np.sum(np.abs(proben_shown)**2, axis=(1,2))
    energy_total = np.sum(energy_n)

    proben_phase = np.angle(proben_shown)
    Pmin, Pmax = proben_phase.min(), proben_phase.max()
    P_norm = plt.Normalize(vmin=Pmin, vmax=Pmax)

    proben_abs = np.abs(proben_shown)
    Amin, Amax = proben_abs.min(), proben_abs.max()
    A_norm = plt.Normalize(vmin=Amin, vmax=Amax)

    phase_hues = np.linspace(0, 1, 256)[:, None]
    s = np.ones_like(phase_hues)
    v = np.ones_like(phase_hues)
    hsv = np.dstack([phase_hues, s, v])
    rgb = hsv_to_rgb(hsv) 
    P_cmap = plt.cm.colors.ListedColormap(rgb[:, 0, :])


    if show_probe_comparison:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        plt.sca(axes[0])
        RGB_Complex_Plot(proben0[0])
        axes[0].set_title('Initial Probe')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        plt.sca(axes[1])
        RGB_Complex_Plot(proben_shown[0])
        axes[1].set_title('Final Probe')
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        fig.subplots_adjust(wspace=0.1, hspace=0.2, left=0.1, right=0.85, top=0.9, bottom=0.06)  
        cbar_ax = fig.add_axes([0.88, 0.06, 0.02, 0.84]) # [left, bottom, width, height]

        sm = plt.cm.ScalarMappable(cmap=P_cmap, norm=P_norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Phase [rad]")

        if save_results:
            outfile = os.path.join(_LOG_DIR, "Probe_Comparison.png")
            plt.savefig(outfile, dpi=300)

        plt.show()
    

    fig, axes = plt.subplots(
                probe_shown_rows, 
                probe_shown_cols, 
                figsize=(probe_shown_cols*4.7, probe_shown_rows*4),
                )
    axes = np.array([axes]).flatten()

    for i, ax in enumerate(axes[:n_state]):
        plt.sca(ax)
        RGB_Complex_Plot(proben_shown[i])
        ax.set_xticks([])
        ax.set_yticks([])
        energy_pp = energy_n[i] / energy_total * 100
        ax.set_title(f'Energy: {energy_pp:.0f}%', fontsize=10)

    for ax in axes[n_state:]:
        ax.axis("off")

    fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.03, right=0.85, top=0.97, bottom=0.03)  
    cbar_ax = fig.add_axes([0.88, 0.03, 0.02, 0.94]) # [left, bottom, width, height]

    sm = plt.cm.ScalarMappable(cmap=P_cmap, norm=P_norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Phase [rad]")

    if save_results:
        outfile = os.path.join(_LOG_DIR, "Probe_Complex.png")
        plt.savefig(outfile, dpi=300)

    plt.show()


    if show_proben_Amp:
        fig, axes = plt.subplots(
                    probe_shown_rows, 
                    probe_shown_cols, 
                    figsize=(probe_shown_cols*4.7, probe_shown_rows*4),
                    )
        axes = np.array([axes]).flatten()

        for ax in axes[n_state:]:
            ax.axis("off")

        fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.03, right=0.85, top=0.97, bottom=0.03)  
        cbar_ax = fig.add_axes([0.88, 0.03, 0.02, 0.94])  # [left, bottom, width, height]

        sm = plt.cm.ScalarMappable(cmap='gray', norm=A_norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Amplitude")

        for i in range(n_state):
            energy_pp = energy_n[i]/np.sum(energy_n)*100
            axes[i].imshow(proben_abs[i], cmap='gray', vmin=Amin, vmax=Amax)
            axes[i].set_title(f'Amp. of {i+1}th probe\nEnergy: {energy_pp:.0f}%', fontsize=10)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        #outfile = os.path.join(_LOG_DIR, "Probe_Amp.png")
        #plt.savefig(outfile, dpi=300)
        plt.show()


    if show_proben_Phase:
        fig, axes = plt.subplots(
                    probe_shown_rows, 
                    probe_shown_cols, 
                    figsize=(probe_shown_cols*4.7, probe_shown_rows*4),
                    )
        axes = np.array([axes]).flatten()

        for ax in axes[n_state:]:
            ax.axis("off")

        fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.03, right=0.85, top=0.97, bottom=0.03)  
        cbar_ax = fig.add_axes([0.88, 0.03, 0.02, 0.94])  # [left, bottom, width, height]

        sm = plt.cm.ScalarMappable(cmap=P_cmap, norm=P_norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Phase [rad]")

        for i in range(n_state):
            energy_pp = energy_n[i]/np.sum(energy_n)*100
            axes[i].imshow(proben_phase[i], cmap=P_cmap, vmin=Pmin, vmax=Pmax)
            axes[i].set_title(f'Phase of {i+1}th probe\nEnergy: {energy_pp:.0f}%', fontsize=10)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        #outfile = os.path.join(_LOG_DIR, "Probe_Phase.png")
        #plt.savefig(outfile, dpi=300)
        plt.show()
 




def iterPtycho_objFunc_plot(objectn, scan_rotation_angle, scan_flip, ptycho_move, data_shape, slice_thickness,
                            show_Amp_objectn  = False,
                            show_Phase_objectn = True,
                            show_Phase_sum_PS = True,
                            save_results = True,
                            ):

    sy, sx, *_ = data_shape
    n_slice = objectn.shape[0]
    object_shown_rows, object_shown_cols = get_subplot_grid(n_slice)

    _LOG_DIR = print_and_log("Object saving")
    if save_results:
        np.save(os.path.join(_LOG_DIR, "Object_Complex.npy"), objectn)

    if scan_flip:
        objectn = np.flip(objectn, axis=2)
        scan_rotation_angle *= -1
    objectn = rotate(objectn, angle=scan_rotation_angle, axes=(1,2))
    sy_r, sx_r = objectn[0].shape
    objectn_shown = objectn[:, int(sy_r/2)-int(sy*ptycho_move/2):int(sy_r/2)+int(sy*ptycho_move/2),int(sx_r/2)-int(sx*ptycho_move/2):int(sx_r/2)+int(sx*ptycho_move/2)]

    if save_results:
        tifffile.imwrite(os.path.join(_LOG_DIR, "Object_Phase.tif"), np.angle(objectn_shown), photometric='minisblack')


    objectn_abs = np.abs(objectn_shown)
    Amin, Amax = objectn_abs.min(), objectn_abs.max()
    A_norm = plt.Normalize(vmin=Amin, vmax=Amax)

    objectn_phase = np.angle(objectn_shown)
    Pmin, Pmax = objectn_phase.min(), objectn_phase.max()
    P_norm = plt.Normalize(vmin=Pmin, vmax=Pmax)


    if show_Amp_objectn:
        fig, axes = plt.subplots(
                    object_shown_rows,
                    object_shown_cols,
                    figsize=(object_shown_cols*4.7, object_shown_rows*4),
                    constrained_layout=False
                    )
        axes = np.array([axes]).flatten()

        for ax in axes[n_slice:]:
            ax.axis("off")

        fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.03, right=0.85, top=0.97, bottom=0.03)  
        cbar_ax = fig.add_axes([0.88, 0.03, 0.02, 0.94])  # [left, bottom, width, height]

        sm = plt.cm.ScalarMappable(cmap='gray', norm=A_norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Amplitude")

        for i in range(n_slice):
            axes[i].imshow(objectn_abs[i], cmap='gray', vmin=Amin, vmax=Amax)
            axes[i].set_title(f'Amplitude {(i+1)*slice_thickness} Å slice', fontsize=10)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        if save_results:
            outfile = os.path.join(_LOG_DIR, "Object_Amplitude.png")
            plt.savefig(outfile, dpi=300)

        plt.show()


    if show_Phase_objectn:
        fig, axes = plt.subplots(
                    object_shown_rows,
                    object_shown_cols,
                    figsize=(object_shown_cols*4.7, object_shown_rows*4),
                    constrained_layout=False
                    )
        axes = np.array([axes]).flatten()

        for ax in axes[n_slice:]:
            ax.axis("off")

        fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.03, right=0.85, top=0.97, bottom=0.03)  
        cbar_ax = fig.add_axes([0.88, 0.03, 0.02, 0.94])  # [left, bottom, width, height]

        sm = plt.cm.ScalarMappable(cmap='gray', norm=P_norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Phase [rad]")

        for i in range(n_slice):
            axes[i].imshow(np.angle(objectn_shown[i]), cmap='gray', vmin=Pmin, vmax=Pmax)
            axes[i].set_title(f'Phase {(i+1)*slice_thickness} Å slice', fontsize=10)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        if save_results:
            outfile = os.path.join(_LOG_DIR, "Object_Phase.png")
            plt.savefig(outfile, dpi=300)

        plt.show()


    print('Object Function Summed')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(np.sum(np.abs(objectn_shown), axis=0), cmap='gray')
    axes[0].set_title('Summed Amplitude Obj')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(np.sum(np.angle(objectn_shown), axis=0), cmap='gray')
    axes[1].set_title('Summed Phase Obj')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()

    if save_results:
        tifffile.imwrite(os.path.join(_LOG_DIR, "Object_Phase_Sum.tif"), np.sum(np.angle(objectn_shown), axis=0))
        tifffile.imwrite(os.path.join(_LOG_DIR, "Object_Amplitude_Sum.tif"), np.sum(np.abs(objectn_shown), axis=0))
        outfile = os.path.join(_LOG_DIR, "Object_Summed.png")
        plt.savefig(outfile, dpi=300)

    plt.show()
        

    if show_Phase_sum_PS:
        #obj_fft = np.fft.fftshift(np.fft.fft2())
        obj_PS = Power_Spectrum_Log(np.sum(np.angle(objectn_shown), axis=0))
        plt.imshow(obj_PS, cmap='gray')
        plt.title('PS of sumed object phase (log)')
        plt.xticks([])
        plt.yticks([])

        if save_results:
            outfile = os.path.join(_LOG_DIR, "Power_Spectrum_of_Sumed_Object_Phase(log).png")
            plt.savefig(outfile, dpi=300)

        plt.show()



def position_correction_plot(pc_mean_shift, posset_pc, posset, save_results=True):

    _LOG_DIR = print_and_log("PC saving")

    plt.figure(figsize=(6,4))
    plt.plot(range(len(pc_mean_shift[1:])), pc_mean_shift[1:])

    plt.xlabel("Iteration")
    plt.ylabel("pixels")
    plt.title("scanning position mean shift")

    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()

    if save_results:
        #np.save(os.path.join(_LOG_DIR, "pc_mean_shift.npy"), pc_mean_shift)
        outfile = os.path.join(_LOG_DIR, "pc_mean_shift.png")
        plt.savefig(outfile, dpi=300)

    plt.show()


    plt.scatter(posset[:,3], -posset[:,2], color='red', label='original')
    plt.scatter(posset_pc[:,3], -posset_pc[:,2], color='black', label='corrected')
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()

    if save_results:
        outfile = os.path.join(_LOG_DIR, "Scan Position Comparison.png")
        plt.savefig(outfile, dpi=300)

    plt.show()




def CoM_plot(g_CoM, g_circle=None, save_results=True):

    _LOG_DIR = print_and_log("CoM saving")
    #np.save(os.path.join(_LOG_DIR, "CoM_Complex.npy"), g_CoM)

    P_norm = plt.Normalize(vmin=0, vmax=360)

    phase_hues = np.linspace(0, 1, 256)[:, None]
    s = np.ones_like(phase_hues)
    v = np.ones_like(phase_hues)
    hsv = np.dstack([phase_hues, s, v])
    rgb = hsv_to_rgb(hsv)
    P_cmap = plt.cm.colors.ListedColormap(rgb[:, 0, :]).reversed()
    

    fig, ax = plt.subplots(1, 1)

    plt.sca(ax)
    RGB_Complex_Plot(g_CoM)
    plt.title('CoM Complex')
    ax.set_xticks([])
    ax.set_yticks([])

    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.2)

    sm = plt.cm.ScalarMappable(cmap=P_cmap, norm=P_norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Orientation [degree]")

    if save_results:
        outfile = os.path.join(_LOG_DIR, "CoM_Complex.png")
        plt.savefig(outfile, dpi=300)

    plt.show()


    if g_circle is not None:
        RGB_Complex_Plot(g_circle)
        plt.xticks([])
        plt.yticks([])
        outfile = os.path.join(_LOG_DIR, "Circle_Complex.png")
        plt.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()



def iCoM_plot(iCoM, loss=None, save_results=True):

    _LOG_DIR = print_and_log("iCoM saving")
    if save_results:
        tifffile.imwrite(os.path.join(_LOG_DIR, "iCoM.tif"), iCoM.astype(dtype=np.float32))

    if loss is not None:
        plt.figure(figsize=(6,4))
        plt.semilogy(range(len(loss[1:])), loss[1:])

        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.title("Convergence Curve")

        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.tight_layout()

        if save_results:
            #np.save(os.path.join(_LOG_DIR, "convergence_curve.npy"), loss)
            outfile = os.path.join(_LOG_DIR, "convergence_curve.png")
            plt.savefig(outfile, dpi=300)

        plt.show()


    #iCoM_fft_shown = FFT_Log(iCoM)
    iCoM_PS = Power_Spectrum_Log(iCoM)
    plt.imshow(iCoM_PS, cmap='gray')
    plt.title('iCoM Power Spectrum (log)')
    plt.xticks([])
    plt.yticks([])
    plt.clim([0,np.max(iCoM_PS)])
    plt.colorbar()

    if save_results:
        outfile = os.path.join(_LOG_DIR, "iCoM_Power_Spectrum(log).png")
        plt.savefig(outfile, dpi=300)

    plt.show()


    plt.imshow(iCoM, cmap='gray')
    plt.title('iCoM')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    if save_results:
        outfile = os.path.join(_LOG_DIR, "iCoM.png")
        plt.savefig(outfile, dpi=300)

    plt.show()



def dCoM_plot(dCoM, save_results=True):

    _LOG_DIR = print_and_log("dCoM saving")
    if save_results:
        tifffile.imwrite(os.path.join(_LOG_DIR, "dCoM.tif"), dCoM.astype(dtype=np.float32))

    plt.imshow(dCoM, cmap='gray')
    plt.title('dCoM')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    if save_results:
        outfile = os.path.join(_LOG_DIR, "dCoM.png")
        plt.savefig(outfile, dpi=300)

    plt.show()



def WDD_plot(obj_fft, obj, plot_FFT=False, save_results=True):

    _LOG_DIR = print_and_log("WDD saving")
    if save_results:
        tifffile.imwrite(os.path.join(_LOG_DIR, "WDD_phase.tif"), np.angle(obj).astype(dtype=np.float32))
        np.save(os.path.join(_LOG_DIR, "WDD_Obj_Complex.npy"), obj)

    if plot_FFT:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        im1 = axes[0].imshow(np.log(np.abs(obj_fft)+1e-5), cmap='gray')
        axes[0].set_title('Amplitude Obj_FFT (log)')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        divider1 = make_axes_locatable(axes[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.2)
        fig.colorbar(im1, cax=cax1)

        im2 = axes[1].imshow(np.angle(obj_fft), cmap='gray')
        axes[1].set_title('Phase Obj_FFT')
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        divider2 = make_axes_locatable(axes[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.2)
        fig.colorbar(im2, cax=cax2)

        plt.tight_layout()

        if save_results:
            outfile = os.path.join(_LOG_DIR, "WDD_Object_FFT.png")
            plt.savefig(outfile, dpi=300)

        plt.show()


    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im1 = axes[0].imshow(np.abs(obj), cmap='gray')
    axes[0].set_title('Amplitude Obj')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(im1, cax=cax1)

    im2 = axes[1].imshow(np.angle(obj), cmap='gray')
    axes[1].set_title('Phase Obj')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(im2, cax=cax2)

    plt.tight_layout()

    if save_results:
        outfile = os.path.join(_LOG_DIR, "WDD_Object.png")
        plt.savefig(outfile, dpi=300)

    plt.show()



def SSB_plot(obj_fft, obj, plot_FFT=False, save_results=True):

    _LOG_DIR = print_and_log("SSB saving")
    if save_results:
        tifffile.imwrite(os.path.join(_LOG_DIR, "SSB_real.tif"), np.real(obj).astype(dtype=np.float32))


    if plot_FFT: 
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        im1 = axes[0].imshow(np.log(np.abs(obj_fft)+1e-5), cmap='gray')
        axes[0].set_title('Amplitude Obj_FFT (log)')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        divider1 = make_axes_locatable(axes[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.2)
        fig.colorbar(im1, cax=cax1)

        im2 = axes[1].imshow(np.angle(obj_fft), cmap='gray')
        axes[1].set_title('Phase Obj_FFT')
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        divider2 = make_axes_locatable(axes[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.2)
        fig.colorbar(im2, cax=cax2)

        plt.tight_layout()

        if save_results:
            outfile = os.path.join(_LOG_DIR, "SSB_Object_FFT.png")
            plt.savefig(outfile, dpi=300)

        plt.show()


    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im1 = axes[0].imshow(np.real(obj), cmap='gray')
    axes[0].set_title('Real Obj')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(im1, cax=cax1)

    im2 = axes[1].imshow(np.imag(obj), cmap='gray')
    axes[1].set_title('Imaginary Obj')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(im2, cax=cax2)

    plt.tight_layout()

    if save_results:
        outfile = os.path.join(_LOG_DIR, "SSB_Object.png")
        plt.savefig(outfile, dpi=300)

    plt.show()



def tcBF_plot(Aligned_BF, intensity_min, save_results=True):

    _LOG_DIR = print_and_log("tcBF saving")
    if save_results:
        tifffile.imwrite(os.path.join(_LOG_DIR, "tcBF.tif"), Aligned_BF.astype(dtype=np.float32))

    tcBF_PS = Power_Spectrum_Log(Aligned_BF)


    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im1 = axes[0].imshow(Aligned_BF, cmap='gray')
    im1.set_clim([intensity_min, np.max(Aligned_BF)])
    axes[0].set_title('tcBF')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(im1, cax=cax1)

    im2 = axes[1].imshow(tcBF_PS, cmap='gray')
    axes[1].set_title('tcBF Power Spectrum (log)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(im2, cax=cax2)

    plt.tight_layout()

    if save_results:
        outfile = os.path.join(_LOG_DIR, "tcBF.png")
        plt.savefig(outfile, dpi=300)

    plt.show()




def vSTEM_plot(vSTEM_siries, radius_list, save_results=True):

    _LOG_DIR = print_and_log("vSTEM saving")

    n_pic = len(radius_list)

    shown_rows, shown_cols = get_subplot_grid(n_pic)


    fig, axes = plt.subplots(
                shown_rows, 
                shown_cols, 
                figsize=(shown_cols*4, shown_rows*4),
                )
    axes = np.array([axes]).flatten()

    for i in range(n_pic):
        axes[i].imshow(vSTEM_siries[i], cmap='gray')
        axes[i].set_title(f'vSTEM ({radius_list[i][0]} - {radius_list[i][1]}) Alpha')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

        if save_results:
            tifffile.imwrite(os.path.join(_LOG_DIR, f"vSTEM_{radius_list[i][0]}-{radius_list[i][1]}_Alpha.tif"), vSTEM_siries[i].astype(dtype=np.float32))

    if save_results:
        outfile = os.path.join(_LOG_DIR, "vSTEM.png")
        plt.savefig(outfile, dpi=300)

    plt.show()


    fig, axes = plt.subplots(
                shown_rows, 
                shown_cols, 
                figsize=(shown_cols*4, shown_rows*4),
                )
    axes = np.array([axes]).flatten()

    for i in range(n_pic):
        axes[i].imshow(Power_Spectrum_Log(vSTEM_siries[i]), cmap='gray')
        axes[i].set_title(f'vSTEM PS (log) of ({radius_list[i][0]} - {radius_list[i][1]}) Alpha')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    if save_results:
        outfile = os.path.join(_LOG_DIR, "vSTEM_Power_Spectrum.png")
        plt.savefig(outfile, dpi=300)

    plt.show()



