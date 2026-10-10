import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
plt.rcParams.update({
    'font.size': 16,          # General font size
    'axes.titlesize': 18,     # Title font size
    'axes.labelsize': 16,     # X and Y axis label font size
    'xtick.labelsize': 14,    # X tick label font size
    'ytick.labelsize': 14,    # Y tick label font size
    'legend.fontsize': 14     # Legend font size
})
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
HARTREE_TO_EV = 27.2114
r = 4
p_idx = 0
nocc = 114
k_label = r"$k=(0,0,0)$" if p_idx == 0 else r"$k=(2\pi/L,0,0)$"
title = rf"114e/485o - {k_label} at $r_s={r}$"
def inspect_hdf5(filename):
    """Utility to print the structure of an HDF5 file for debugging."""
    with h5py.File(filename, "r") as f:
        print("Top-level keys:", list(f.keys()))
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                print("DATASET:", name, obj.shape, obj.dtype)
            elif isinstance(obj, h5py.Group):
                print("GROUP:  ", name)
        f.visititems(walk)
#fig2 refs
ref_data_2 = np.genfromtxt('fig2.csv', delimiter=',', skip_header=2, filling_values=np.nan)
ccsd_x_2 = ref_data_2[:, 0]
ccsd_y_2 = ref_data_2[:, 1]
hf_gw_x_2 = ref_data_2[:, 2]
hf_gw_y_2 = ref_data_2[:, 3]
lda_gw_x_2 = ref_data_2[:, 4]
lda_gw_y_2 = ref_data_2[:, 5]
lda_gw_c_x_2 = ref_data_2[:, 6]
lda_gw_c_y_2 = ref_data_2[:, 7]

cpp_gw_cache = "ne114_rs4_gw_cache_cpp.h5"
inspect_hdf5(cpp_gw_cache)
'''
DATASET: hf_g0w0_spec_ne114 (1, 7997) float64
DATASET: hf_g0w0pC_mfepep_ne114 (1, 7997) float64
DATASET: hf_g0w0pC_qpepep_ne114 (1, 7997) float64
DATASET: hf_g0wpC_mfepep_ne114 (1, 7997) float64
DATASET: hf_g0wpC_qpepep_ne114 (1, 7997) float64
DATASET: hf_gw0pC_mfepep_ne114 (1, 7997) float64
DATASET: hf_gw0pC_qpepep_ne114 (1, 7997) float64
DATASET: hf_gwpC_mfepep_ne114 (1, 7997) float64
DATASET: hf_gwpC_qpepep_ne114 (1, 7997) float64
DATASET: hf_omegas_ne114 (1, 7997) float64
DATASET: hf_qpes_g0w0_ne114 (1, 19) float64
DATASET: hf_qpes_g0w_ne114 (1, 19) float64
DATASET: hf_qpes_gw0_ne114 (1, 19) float64
DATASET: hf_qpes_gw_ne114 (1, 19) float64
DATASET: lda_g0w0_spec_ne114 (1, 7997) float64
DATASET: lda_g0w0pC_mfepep_ne114 (1, 7997) float64
DATASET: lda_g0w0pC_qpepep_ne114 (1, 7997) float64
DATASET: lda_g0wpC_mfepep_ne114 (1, 7997) float64
DATASET: lda_g0wpC_qpepep_ne114 (1, 7997) float64
DATASET: lda_gw0pC_mfepep_ne114 (1, 7997) float64
DATASET: lda_gw0pC_qpepep_ne114 (1, 7997) float64
DATASET: lda_gwpC_mfepep_ne114 (1, 7997) float64
DATASET: lda_gwpC_qpepep_ne114 (1, 7997) float64
DATASET: lda_omegas_ne114 (1, 7997) float64
DATASET: lda_qpes_g0w0_ne114 (1, 19) float64
DATASET: lda_qpes_g0w_ne114 (1, 19) float64
DATASET: lda_qpes_gw0_ne114 (1, 19) float64
DATASET: lda_qpes_gw_ne114 (1, 19) float64
'''
with h5py.File(cpp_gw_cache, "r") as f:
    cpp_omegas_hf = f['hf_omegas_ne114'][:].flatten()
    cpp_g0w0_spec_hf = f['hf_g0w0_spec_ne114'][:].flatten()
    cpp_g0w0pC_qpepep_hf = f['hf_g0w0pC_qpepep_ne114'][:].flatten()
    cpp_g0wpC_qpepep_hf = f['hf_g0wpC_qpepep_ne114'][:].flatten()
    cpp_gw0pC_qpepep_hf = f['hf_gw0pC_qpepep_ne114'][:].flatten()
    cpp_gwpC_qpepep_hf = f['hf_gwpC_qpepep_ne114'][:].flatten()
    cpp_g0w0pC_mfepep_hf = f['hf_g0w0pC_mfepep_ne114'][:].flatten()
    cpp_g0wpC_mfepep_hf = f['hf_g0wpC_mfepep_ne114'][:].flatten()
    cpp_gw0pC_mfepep_hf = f['hf_gw0pC_mfepep_ne114'][:].flatten()
    cpp_gwpC_mfepep_hf = f['hf_gwpC_mfepep_ne114'][:].flatten()
    cpp_omegas_lda = f['lda_omegas_ne114'][:].flatten()
    cpp_g0w0_spec_lda = f['lda_g0w0_spec_ne114'][:].flatten()
    cpp_g0w0pC_qpepep_lda = f['lda_g0w0pC_qpepep_ne114'][:].flatten()
    cpp_g0wpC_qpepep_lda = f['lda_g0wpC_qpepep_ne114'][:].flatten()
    cpp_gw0pC_qpepep_lda = f['lda_gw0pC_qpepep_ne114'][:].flatten()
    cpp_gwpC_qpepep_lda = f['lda_gwpC_qpepep_ne114'][:].flatten()
    cpp_g0w0pC_mfepep_lda = f['lda_g0w0pC_mfepep_ne114'][:].flatten()
    cpp_g0wpC_mfepep_lda = f['lda_g0wpC_mfepep_ne114'][:].flatten()
    cpp_gw0pC_mfepep_lda = f['lda_gw0pC_mfepep_ne114'][:].flatten()
    cpp_gwpC_mfepep_lda = f['lda_gwpC_mfepep_ne114'][:].flatten()
    cpp_qpes_g0w0_lda = f['lda_qpes_g0w0_ne114'][:].flatten()
    cpp_qpes_g0w_lda = f['lda_qpes_g0w_ne114'][:].flatten()
    cpp_qpes_gw0_lda = f['lda_qpes_gw0_ne114'][:].flatten()
    cpp_qpes_gw_lda = f['lda_qpes_gw_ne114'][:].flatten()

cpp_hfgw='/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/114e485o_nwcd400_HF_cpp_to.h5'
inspect_hdf5(cpp_hfgw)
'''
DATASET: hf_g0w0_spec_ne114 (1, 280) float64
DATASET: hf_g0w0pC_mfepep_ne114 (1, 797) float64
DATASET: hf_g0w0pC_qpepep_ne114 (1, 797) float64
DATASET: hf_omegas_mC_ne114 (1, 280) float64
DATASET: hf_omegas_pC_ne114 (1, 797) float64
DATASET: hf_qpes_g0w0_ne114 (1, 1) float64
'''
with h5py.File(cpp_hfgw, "r") as f:
    cpp_hfgw_g0w0_spec = f['hf_g0w0_spec_ne114'][:].flatten()
    cpp_hfgw_g0w0pC_mfepep = f['hf_g0w0pC_mfepep_ne114'][:].flatten()
    cpp_hfgw_g0w0pC_qpepep = f['hf_g0w0pC_qpepep_ne114'][:].flatten()
    cpp_hfgw_omegas_mC = f['hf_omegas_mC_ne114'][:].flatten()
    cpp_hfgw_omegas_pC = f['hf_omegas_pC_ne114'][:].flatten()
    cpp_hfgw_qpe_g0w0 = f['hf_qpes_g0w0_ne114'][()].squeeze()
cpp_ldagw='/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/114e485o_nwcd400_LDA_cpp_to.h5'
inspect_hdf5(cpp_ldagw)
with h5py.File(cpp_ldagw, "r") as f:
    cpp_ldagw_g0w0_spec = f['lda_g0w0_spec_ne114'][:].flatten()
    cpp_ldagw_omegas_mC = f['lda_omegas_mC_ne114'][:].flatten()
    cpp_ldagw_omegas_pC = f['lda_omegas_pC_ne114'][:].flatten()
    cpp_ldagw_qpe_g0w0 = f['lda_qpes_g0w0_ne114'][()].squeeze()
    cpp_ldagw_g0w0pC_qpepep = f['lda_g0w0pC_qpepep_ne114'][:].flatten()
    cpp_ldagw_g0w0pC_mfepep = f['lda_g0w0pC_mfepep_ne114'][:].flatten()
cpp_hfgw_casida='/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/114e485o_nwcd400_HF_cpp_casida_1shot_24510785_small.h5'
inspect_hdf5(cpp_hfgw_casida)
'''
DATASET: hf_g0w0_spec_ne114 (1, 420) float64
DATASET: hf_g0w0pC_mfepep_ne114 (1, 797) float64
DATASET: hf_g0w0pC_qpepep_ne114 (1, 797) float64
DATASET: hf_omegas_mC_ne114 (1, 420) float64
DATASET: hf_omegas_pC_ne114 (1, 797) float64
DATASET: hf_qpes_g0w0_ne114 (1, 1) float64
'''
with h5py.File(cpp_hfgw_casida, "r") as f:
    cpp_hfgw_casida_g0w0_spec = f['hf_g0w0_spec_ne114'][:].flatten()
    cpp_hfgw_casida_g0w0pC_mfepep = f['hf_g0w0pC_mfepep_ne114'][:].flatten()
    cpp_hfgw_casida_g0w0pC_qpepep = f['hf_g0w0pC_qpepep_ne114'][:].flatten()
    cpp_hfgw_casida_omegas_mC = f['hf_omegas_mC_ne114'][:].flatten()
    cpp_hfgw_casida_omegas_pC = f['hf_omegas_pC_ne114'][:].flatten()
    cpp_hfgw_casida_qpe_g0w0 = f['hf_qpes_g0w0_ne114'][()].squeeze()
cpp_ldagw_casida='/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/114e485o_nwcd400_LDA_cpp_casida_1shot_24510745_small.h5'
inspect_hdf5(cpp_ldagw_casida)
'''
DATASET: lda_g0w0_spec_ne114 (1, 420) float64
DATASET: lda_g0w0pC_mfepep_ne114 (1, 797) float64
DATASET: lda_g0w0pC_qpepep_ne114 (1, 797) float64
DATASET: lda_omegas_mC_ne114 (1, 420) float64
DATASET: lda_omegas_pC_ne114 (1, 797) float64
DATASET: lda_qpes_g0w0_ne114 (1, 1) float64
'''
with h5py.File(cpp_ldagw_casida, "r") as f:
    cpp_ldagw_casida_g0w0_spec = f['lda_g0w0_spec_ne114'][:].flatten()
    cpp_ldagw_casida_omegas_mC = f['lda_omegas_mC_ne114'][:].flatten()
    cpp_ldagw_casida_omegas_pC = f['lda_omegas_pC_ne114'][:].flatten()
    cpp_ldagw_casida_qpe_g0w0 = f['lda_qpes_g0w0_ne114'][()].squeeze()
    cpp_ldagw_casida_g0w0pC_qpepep = f['lda_g0w0pC_qpepep_ne114'][:].flatten()
    cpp_ldagw_casida_g0w0pC_mfepep = f['lda_g0w0pC_mfepep_ne114'][:].flatten()
cpp_hfgw_anal = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/114e485o_nwcd400_HF_cpp_casida_1shot_24510785_small.h5'
inspect_hdf5(cpp_hfgw_anal)
with h5py.File(cpp_hfgw_anal, "r") as f:
    cpp_hfgw_anal_g0w0_spec = f['hf_g0w0_spec_ne114'][:].flatten()
    cpp_hfgw_anal_g0w0pC_mfepep = f['hf_g0w0pC_mfepep_ne114'][:].flatten()
    cpp_hfgw_anal_g0w0pC_qpepep = f['hf_g0w0pC_qpepep_ne114'][:].flatten()
    cpp_hfgw_anal_omegas_mC = f['hf_omegas_mC_ne114'][:].flatten()
    cpp_hfgw_anal_omegas_pC = f['hf_omegas_pC_ne114'][:].flatten()
    cpp_hfgw_anal_qpe_g0w0 = f['hf_qpes_g0w0_ne114'][()].squeeze()
cpp_ldagw_anal = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/114e485o_nwcd400_LDA_cpp_casida_1shot_24510745_small.h5'
inspect_hdf5(cpp_ldagw_anal)
with h5py.File(cpp_ldagw_anal, "r") as f:
    cpp_ldagw_anal_g0w0_spec = f['lda_g0w0_spec_ne114'][:].flatten()
    cpp_ldagw_anal_omegas_mC = f['lda_omegas_mC_ne114'][:].flatten()
    cpp_ldagw_anal_omegas_pC = f['lda_omegas_pC_ne114'][:].flatten()
    cpp_ldagw_anal_qpe_g0w0 = f['lda_qpes_g0w0_ne114'][()].squeeze()
    cpp_ldagw_anal_g0w0pC_qpepep = f['lda_g0w0pC_qpepep_ne114'][:].flatten()
    cpp_ldagw_anal_g0w0pC_mfepep = f['lda_g0w0pC_mfepep_ne114'][:].flatten()
cpp_ldag0w = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/114e485o_nwcd400_LDA_cpp_casida_analytic_g0w_24857640_spec.h5'
inspect_hdf5(cpp_ldag0w)
'''
DATASET: lda_g0w_spec_ne114 (1, 420) float64
DATASET: lda_g0wpC_mfepep_ne114 (1, 797) float64
DATASET: lda_g0wpC_qpepep_ne114 (1, 797) float64
DATASET: lda_omegas_mC_ne114 (1, 420) float64
DATASET: lda_omegas_pC_ne114 (1, 797) float64
DATASET: lda_qpes_g0w_ne114 (1, 485) float64
'''
with h5py.File(cpp_ldag0w, "r") as f:
    cpp_ldag0w_g0w_spec = f['lda_g0w_spec_ne114'][:].flatten()
    cpp_ldag0w_omegas_mC = f['lda_omegas_mC_ne114'][:].flatten()
    cpp_ldag0w_omegas_pC = f['lda_omegas_pC_ne114'][:].flatten()
    cpp_ldag0w_qpe_g0w = f['lda_qpes_g0w_ne114'][:].flatten()
    cpp_ldag0w_g0wpC_qpepep = f['lda_g0wpC_qpepep_ne114'][:].flatten()
    cpp_ldag0w_g0wpC_mfepep = f['lda_g0wpC_mfepep_ne114'][:].flatten()
cpp_ldagw0 = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/114e485o_nwcd400_LDA_cpp_casida_analytic_gw0_24857399_spec.h5'
inspect_hdf5(cpp_ldagw0)
'''
DATASET: lda_gw0_spec_ne114 (1, 420) float64
DATASET: lda_gw0pC_mfepep_ne114 (1, 797) float64
DATASET: lda_gw0pC_qpepep_ne114 (1, 797) float64
DATASET: lda_omegas_mC_ne114 (1, 420) float64
DATASET: lda_omegas_pC_ne114 (1, 797) float64
DATASET: lda_qpes_gw0_ne114 (1, 485) float64
'''
with h5py.File(cpp_ldagw0, "r") as f:
    cpp_ldagw0_gw0_spec = f['lda_gw0_spec_ne114'][:].flatten()
    cpp_ldagw0_omegas_mC = f['lda_omegas_mC_ne114'][:].flatten()
    cpp_ldagw0_omegas_pC = f['lda_omegas_pC_ne114'][:].flatten()
    cpp_ldagw0_qpe_gw0 = f['lda_qpes_gw0_ne114'][:].flatten()
    cpp_ldagw0_gwpC_qpepep = f['lda_gw0pC_qpepep_ne114'][:].flatten()
    cpp_ldagw0_gwpC_mfepep = f['lda_gw0pC_mfepep_ne114'][:].flatten()
cpp_ldascgw = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/114e485o_nwcd400_LDA_cpp_casida_analytic_scgw_24857850_spec.h5'
inspect_hdf5(cpp_ldascgw)
'''
DATASET: lda_gwpC_mfepep_ne114 (1, 797) float64
DATASET: lda_gwpC_qpepep_ne114 (1, 797) float64
DATASET: lda_omegas_mC_ne114 (1, 420) float64
DATASET: lda_omegas_pC_ne114 (1, 797) float64
DATASET: lda_qpes_gw_ne114 (1, 485) float64
DATASET: lda_scgw_spec_ne114 (1, 420) float64
'''
with h5py.File(cpp_ldascgw, "r") as f:
    cpp_ldascgw_scgw_spec = f['lda_scgw_spec_ne114'][:].flatten()
    cpp_ldascgw_omegas_mC = f['lda_omegas_mC_ne114'][:].flatten()
    cpp_ldascgw_omegas_pC = f['lda_omegas_pC_ne114'][:].flatten()
    cpp_ldascgw_qpe_gw = f['lda_qpes_gw_ne114'][:].flatten()
    cpp_ldascgw_gwpC_qpepep = f['lda_gwpC_qpepep_ne114'][:].flatten()
    cpp_ldascgw_gwpC_mfepep = f['lda_gwpC_mfepep_ne114'][:].flatten()
# cpp_lda_scce = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/114e485o_nwcd400_LDA_cpp_scce_27661417.h5'
cpp_lda_scce = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/114e485o_nwcd400_LDA_cpp_scce_27780781.h5'
inspect_hdf5(cpp_lda_scce)
'''
Top-level keys: ['lda_scce']
GROUP:   lda_scce
GROUP:   lda_scce/meanfield
GROUP:   lda_scce/meanfield/p0_ne114
DATASET: lda_scce/meanfield/p0_ne114/A_iter0 (1, 2701) float64
DATASET: lda_scce/meanfield/p0_ne114/A_iter1 (1, 2701) float64
DATASET: lda_scce/meanfield/p0_ne114/A_iter2 (1, 2701) float64
DATASET: lda_scce/meanfield/p0_ne114/A_iter3 (1, 2701) float64
DATASET: lda_scce/meanfield/p0_ne114/A_iter4 (1, 2701) float64
DATASET: lda_scce/meanfield/p0_ne114/omega_A (1, 2701) float64
'''
with h5py.File(cpp_lda_scce, "r") as f:
    A0_scce_mfepep_gw0 = f['lda_scce/meanfield/p0_ne114/A_iter0'][:].flatten()
    A4_scce_mfepep_gw0 = f['lda_scce/meanfield/p0_ne114/A_iter4'][:].flatten()
    omega_A = f['lda_scce/meanfield/p0_ne114/omega_A'][:].flatten()
    '''
    GROUP:   lda_scce/qp
GROUP:   lda_scce/qp/p0_ne114
DATASET: lda_scce/qp/p0_ne114/A_iter0 (1, 2701) float64
DATASET: lda_scce/qp/p0_ne114/A_iter1 (1, 2701) float64
DATASET: lda_scce/qp/p0_ne114/A_iter2 (1, 2701) float64
DATASET: lda_scce/qp/p0_ne114/A_iter3 (1, 2701) float64
DATASET: lda_scce/qp/p0_ne114/A_iter4 (1, 2701) float64
'''
    A0_scce_qpepep_gw0 = f['lda_scce/qp/p0_ne114/A_iter0'][:].flatten()
    A4_scce_qpepep_gw0 = f['lda_scce/qp/p0_ne114/A_iter4'][:].flatten()
lda_scce_v2 = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/114e485o_LDA_scce_30422508.h5'
'''
GROUP:   lda_scce
GROUP:   lda_scce/meanfield
GROUP:   lda_scce/meanfield/p0_ne114
DATASET: lda_scce/meanfield/p0_ne114/A_iter0 (1, 2701) float64
DATASET: lda_scce/meanfield/p0_ne114/A_iter1 (1, 2701) float64
DATASET: lda_scce/meanfield/p0_ne114/A_iter2 (1, 2701) float64
DATASET: lda_scce/meanfield/p0_ne114/A_iter3 (1, 2701) float64
DATASET: lda_scce/meanfield/p0_ne114/A_iter4 (1, 2701) float64
DATASET: lda_scce/meanfield/p0_ne114/A_iter5 (1, 2701) float64
DATASET: lda_scce/meanfield/p0_ne114/A_iter6 (1, 2701) float64
DATASET: lda_scce/meanfield/p0_ne114/deltas (1, 6) float64
DATASET: lda_scce/meanfield/p0_ne114/niter (1, 1) float64
DATASET: lda_scce/meanfield/p0_ne114/omega_A (1, 2701) float64
DATASET: lda_scce/meanfield/p0_ne114/omega_sigma (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_im_iter0 (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_im_iter1 (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_im_iter2 (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_im_iter3 (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_im_iter4 (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_im_iter5 (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_im_iter6 (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_re_iter0 (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_re_iter1 (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_re_iter2 (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_re_iter3 (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_re_iter4 (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_re_iter5 (1, 4001) float64
DATASET: lda_scce/meanfield/p0_ne114/sigma_re_iter6 (1, 4001) float64
'''
inspect_hdf5(lda_scce_v2)
with h5py.File(lda_scce_v2, "r") as f:
    A0_scce_mfepep_gw0_v2 = f['lda_scce/meanfield/p0_ne114/A_iter0'][:].flatten()
    A6_scce_mfepep_gw0_v2 = f['lda_scce/meanfield/p0_ne114/A_iter6'][:].flatten()
    omega_A_v2 = f['lda_scce/meanfield/p0_ne114/omega_A'][:].flatten()

cpp_hf_scce = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/114e485o_nwcd400_HF_cpp_scce_27661439.h5'
inspect_hdf5(cpp_hf_scce)
'''
Top-level keys: ['hf_scce']
GROUP:   hf_scce
GROUP:   hf_scce/meanfield
GROUP:   hf_scce/meanfield/p0_ne114
DATASET: hf_scce/meanfield/p0_ne114/A_iter0 (1, 4239) float64
DATASET: hf_scce/meanfield/p0_ne114/A_iter1 (1, 4239) float64
DATASET: hf_scce/meanfield/p0_ne114/A_iter2 (1, 4239) float64
DATASET: hf_scce/meanfield/p0_ne114/A_iter3 (1, 4239) float64
DATASET: hf_scce/meanfield/p0_ne114/A_iter4 (1, 4239) float64
DATASET: hf_scce/meanfield/p0_ne114/deltas (1, 4) float64
DATASET: hf_scce/meanfield/p0_ne114/niter (1, 1) float64
DATASET: hf_scce/meanfield/p0_ne114/omega_A (1, 4239) float64
'''
with h5py.File(cpp_hf_scce, "r") as f:
    A0_scce_mfepep_gw0_hf = f['hf_scce/meanfield/p0_ne114/A_iter0'][:].flatten()
    A4_scce_mfepep_gw0_hf = f['hf_scce/meanfield/p0_ne114/A_iter4'][:].flatten()
    omega_A_hf = f['hf_scce/meanfield/p0_ne114/omega_A'][:].flatten()
cvs_polaron_spec = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/cvs_polaron_spec.h5'
'''
[pkozlowski@holy8a24301 quasiboson]$ h5ls cvs_polaron_spec.h5 
114e485o                 Group
14e19o                   Group
14e57o                   Group
38e123o                  Group
54e305o                  Group
[pkozlowski@holy8a24301 quasiboson]$ h5ls cvs_polaron_spec.h5/114e485o 
hf_bws2_a1_4_a2_-1       Group
hf_bws2_a1_4_a2_-2       Group
hf_bws2_a1_4_a2_-3       Group
hf_rpa                   Group
lda_rpa                  Group
[pkozlowski@holy8a24301 quasiboson]$ h5ls cvs_polaron_spec.h5/114e485o/lda_rpa 
cut_full                 Group
cut_q25                  Group
cut_q50                  Group
cut_q75                  Group
cut_q90                  Group
[pkozlowski@holy8a24301 quasiboson]$ h5ls cvs_polaron_spec.h5/114e485o/lda_rpa/cut_full
A                        Dataset {15915}
omegas                   Dataset {15915}
[pkozlowski@holy8a24301 quasiboson]$ 
'''
with h5py.File(cvs_polaron_spec, "r") as f:
    cvs_polaron_ldagw_omegas = f['114e485o/lda_rpa/cut_full/omegas'][:].flatten()
    cvs_polaron_ldagw_A = f['114e485o/lda_rpa/cut_full/A'][:].flatten()
ldagw_chk = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/conv_114e485o_exact_rs4_nwcd200_eta0p8_32194013.h5'
inspect_hdf5(ldagw_chk)
with h5py.File(ldagw_chk, "r") as f:
    '''
    DATASET: p0/A_mfepep (1, 797) float64
DATASET: p0/A_qpepep (1, 797) float64
DATASET: p0/omega_A (1, 797) float64
'''
    A_mfepep_ldagw_chk = f['p0/A_mfepep'][:].flatten()
    omega_A_ldagw_chk = f['p0/omega_A'][:].flatten()
bws2a2_chk = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/conv_114e485o_bws2_rs4_a12_eta0p8_32196807.h5'
inspect_hdf5(bws2a2_chk)
with h5py.File(bws2a2_chk, "r") as f:
    '''
    DATASET: p0/A (1, 797) float64
DATASET: p0/omega_A (1, 797) float64
'''
    A_bws2a2_chk = f['p0/A'][:].flatten()
    omega_A_bws2a2_chk = f['p0/omega_A'][:].flatten()
plt.figure(figsize=(10, 6))
# plt.plot(omega_A_v2 * HARTREE_TO_EV, A0_scce_mfepep_gw0_v2/HARTREE_TO_EV, label=f'SCCE(GW0)@LDA; epep@mf (Initial iteration)', color='yellow', linestyle='-')
# plt.plot(omega_A_ldagw_chk * HARTREE_TO_EV, A_mfepep_ldagw_chk/HARTREE_TO_EV, label=f'Converged LDA+G0W0+C epep@mf', color='orange', linestyle='-')
plt.plot(omega_A_bws2a2_chk * HARTREE_TO_EV, A_bws2a2_chk/HARTREE_TO_EV, label=rf'BWs2+C; epep@mf ($\alpha_1=2$, $\alpha_2=0$)', color='purple', linestyle='-')
# plt.plot(omega_A_v2 * HARTREE_TO_EV, A6_scce_mfepep_gw0_v2/HARTREE_TO_EV, label=f'SCCE(GW0)@LDA; epep@mf (Converged to 1e-7 Ha in 6 iters)', color='red', linestyle='--')
# plot fig2 refs
# plt.plot(hf_gw_x_2, hf_gw_y_2, label='Garnet\'s HF+G0W0', color='blue', linestyle='-')
norm_lda_gw_2 = np.trapz(lda_gw_y_2, lda_gw_x_2)
# plt.plot(lda_gw_x_2, lda_gw_y_2, label=f'Garnet\'s LDA+G0W0', color='orange', linestyle='-')
plt.plot(lda_gw_c_x_2, lda_gw_c_y_2, label='Garnet\'s LDA+G0W0+C', color='black', linestyle='-')
plt.plot(ccsd_x_2, ccsd_y_2, label='Garnet\'s CCSD', color='green', linestyle='-')

# plt.plot(omega_A * HARTREE_TO_EV, A0_scce_mfepep_gw0/HARTREE_TO_EV, label=f'SCCE(GW0)@LDA;epep@mf (Initial iteration)', color='yellow', linestyle='-')
# plt.plot(omega_A * HARTREE_TO_EV, A4_scce_mfepep_gw0/HARTREE_TO_EV, label=f'SCCE(GW0)@LDA; epep@mf (Converged to 1e-7 Ha in 8 iters)', color='red', linestyle=':')  
# plt.plot(omega_A_hf * HARTREE_TO_EV, A0_scce_mfepep_gw0_hf/HARTREE_TO_EV, label=f'SCCE(GW0)@HF;epep@mf (Initial iteration)', color='blue', linestyle='-')
# plt.plot(omega_A_hf * HARTREE_TO_EV, A4_scce_mfepep_gw0_hf/HARTREE_TO_EV, label=f'SCCE(GW0)@HF;epep@mf (Converged at four iterations)', color='purple', linestyle=':')
# plot my GW data; I just want to plot an axvline at the G0 W0 qp pole, then the ref+GW spectral functions, and then the ref+G0W0+C spectral functions for both HF and LDA starting points
# plt.plot(cpp_hfgw_omegas_mC * HARTREE_TO_EV, cpp_hfgw_g0w0_spec/(HARTREE_TO_EV*1.5), label=f'My HF+G0W0 w/ CD', color='red', linestyle='--')
# plt.plot(cpp_hfgw_casida_omegas_mC * HARTREE_TO_EV, cpp_hfgw_casida_g0w0_spec/(HARTREE_TO_EV*1.5), label=f'My HF+G0W0 (Exact)', color='red', linestyle=':')
# plt.plot(cpp_hfgw_omegas_pC * HARTREE_TO_EV, cpp_hfgw_g0w0pC_qpepep/(HARTREE_TO_EV), label=f'My HF+G0W0+C epep@qp', color='gray', linestyle='-')
norm_cpp_lda_gw_2 = np.trapz(cpp_ldagw_g0w0_spec/HARTREE_TO_EV, cpp_ldagw_omegas_mC * HARTREE_TO_EV)
# plt.plot(cpp_ldagw_omegas_mC * HARTREE_TO_EV, cpp_ldagw_g0w0_spec/HARTREE_TO_EV, label=f'My LDA+G0W0 w/ CD', color='magenta', linestyle='--')
# plt.plot(cpp_ldagw_casida_omegas_mC * HARTREE_TO_EV, cpp_ldagw_casida_g0w0_spec/HARTREE_TO_EV, label=f'My LDA+G0W0', color='cyan', linestyle=':')
# plt.plot(cpp_ldagw_omegas_pC * HARTREE_TO_EV, cpp_ldagw_g0w0pC_qpepep/HARTREE_TO_EV, label=f'My LDA+G0W0+C epep@qp', color='magenta', linestyle='-')
# plt.plot(cpp_ldagw_omegas_pC * HARTREE_TO_EV, cpp_ldagw_g0w0pC_mfepep/HARTREE_TO_EV, label=f'My LDA+G0W0+C epep@mf', color='gray', linestyle='-')
# plt.plot(cpp_ldagw_anal_omegas_pC * HARTREE_TO_EV, cpp_ldagw_anal_g0w0pC_qpepep/HARTREE_TO_EV, label=f'My LDA+G0W0+C; epep@qp', color='orange', linestyle='--')
plt.plot(cpp_ldagw_anal_omegas_pC * HARTREE_TO_EV, cpp_ldagw_anal_g0w0pC_mfepep/HARTREE_TO_EV, label=f'LDA+G0W0+C; epep@mf', color='blue', linestyle='--')
# plt.plot(cvs_polaron_ldagw_omegas * HARTREE_TO_EV, cvs_polaron_ldagw_A/HARTREE_TO_EV, label=f'Cvs Polaron LDA+G0W0+C; epep@mf', color='green', linestyle='--')
# plt.axvline(cpp_ldagw_qpe_g0w0 * HARTREE_TO_EV, color='red', linestyle='-', label='LDA+G0W0 QP Pole')
# plt.plot(cpp_ldag0w_omegas_pC * HARTREE_TO_EV, cpp_ldag0w_g0wpC_qpepep/HARTREE_TO_EV, label=f'My LDA+G0W+C epep@qp', color='green', linestyle='--')
# plt.plot(cpp_ldag0w_omegas_pC * HARTREE_TO_EV, cpp_ldag0w_g0wpC_mfepep/HARTREE_TO_EV, label=f'My LDA+G0W+C epep@mf', color='purple', linestyle='--')
# plt.plot(cpp_ldagw0_omegas_pC * HARTREE_TO_EV, cpp_ldagw0_gwpC_qpepep/HARTREE_TO_EV, label=f'My LDA+GW0+C epep@qp', color='brown', linestyle='--')


# plt.plot(cpp_ldagw0_omegas_pC * HARTREE_TO_EV, cpp_ldagw0_gwpC_mfepep/HARTREE_TO_EV, label=f'M My LDA+GW0+C epep@mf', color='cyan', linestyle='--')
# plt.plot(cpp_ldascgw_omegas_pC * HARTREE_TO_EV, cpp_ldascgw_gwpC_qpepep/HARTREE_TO_EV, label=f'My LDA+GW+C epep@qp', color='black', linestyle='--')
# plt.plot(cpp_ldascgw_omegas_pC * HARTREE_TO_EV, cpp_ldascgw_gwpC_mfepep/HARTREE_TO_EV, label=f'My LDA+GW+C epep@mf', color='gray', linestyle='--')
# plt.axvline(cpp_ldagw_anal_qpe_g0w0 * HARTREE_TO_EV, color='green', linestyle='--', label='CPP LDA+G0W0 QP Pole')

# plt.plot(cpp_ldagw_omegas_pC * HARTREE_TO_EV, cpp_ldagw_g0w0pC_qpepep/HARTREE_TO_EV, label=f'My LDA+G0W0+C epep@qp', color='magenta', linestyle='-')
# plt.plot(cpp_ldagw_omegas_pC * HARTREE_TO_EV, cpp_ldagw_g0w0pC_mfepep/HARTREE_TO_EV, label=f'My LDA+G0W0+C epep@mf', color='gray', linestyle='--')
# plt.plot(cpp_omegas_lda * HARTREE_TO_EV, cpp_g0w0pC_qpepep_lda/HARTREE_TO_EV, label='LDA+G0W0+C epep@qp', color='blue', linestyle=':')
# plt.plot(cpp_omegas_lda * HARTREE_TO_EV, cpp_g0w0pC_mfepep_lda/HARTREE_TO_EV, label='LDA+G0W0+C epep@mf', color='blue', linestyle='-')
# # plt.plot(cpp_omegas_lda * HARTREE_TO_EV, cpp_gw0pC_qpepep_lda/HARTREE_TO_EV, label='LDA+GW0+C epep@qp', color='red', linestyle='-')
# plt.plot(cpp_omegas_lda * HARTREE_TO_EV, cpp_gw0pC_mfepep_lda/HARTREE_TO_EV, label='LDA+GW0+C epep@mf', color='red', linestyle='-')
# # plt.plot(cpp_omegas_lda * HARTREE_TO_EV, cpp_g0wpC_qpepep_lda/HARTREE_TO_EV, label='LDA+G0W+C epep@qp', color='orange', linestyle='--')
# plt.plot(cpp_omegas_lda * HARTREE_TO_EV, cpp_g0wpC_mfepep_lda/HARTREE_TO_EV, label='LDA+G0W+C epep@mf', color='orange', linestyle='--')
# # plt.plot(cpp_omegas_lda * HARTREE_TO_EV, cpp_gwpC_qpepep_lda/HARTREE_TO_EV, label='LDA+GW+C epep@qp', color='purple', linestyle='-')
# plt.plot(cpp_omegas_lda * HARTREE_TO_EV, cpp_gwpC_mfepep_lda/HARTREE_TO_EV, label='LDA+GW+C epep@mf', color='purple', linestyle='-')
# # plot axvline at the lda+gw qp poles for g0w0 and gw0
# plt.axvline(cpp_qpes_gw0_lda[p_idx] * HARTREE_TO_EV, color='green', linestyle='-', label='CPP LDA+GW0 QP Pole')
# plt.axvline(qpes_gw0_lda[p_idx] * HARTREE_TO_EV, color='orange', linestyle='--', label='Python LDA+GW0 QP Pole')
# plt.plot(cpp_omegas_lda * HARTREE_TO_EV, cpp_gw0pC_mfepep_lda/HARTREE_TO_EV, label='LDA+GW0+C epep@qp CPP', color='purple', linestyle='-')
# plt.plot(omegas * HARTREE_TO_EV, g0w0pC_mfepep_lda/HARTREE_TO_EV, label='LDA+G0W0+C epep@mf', color='black', linestyle='-')
# # plt.plot(cpp_omegas_lda * HARTREE_TO_EV, cpp_g0w0pC_mfepep_lda/HARTREE_TO_EV, label='LDA+G0W0+C epep@mf CPP', color='gray', linestyle=':')
# a1eq2_path = "ce_bws2_a1e2scan_rs4_npw485.h5"
# inspect_hdf5(a1eq2_path)
# with h5py.File(a1eq2_path, "r") as f:
#     a1eq2_alpha_pairs = np.array(f["alpha_pairs"])
#     a1eq2_a_fft = np.array(f["cumulant_spec_fn"])
#     a1eq2_a_sp_energies = np.array(f["bws2_sp_energies"])
#     a1eq2_omegas_fft = np.squeeze(f["omegas_fft"])
# omegas_485 = np.squeeze(a1eq2_omegas_fft)

# for (alpha1, alpha2) in a1eq2_alpha_pairs:
#     if alpha1 != 2 or alpha2 != 0:
#         continue
#     idx = np.where(
#         (a1eq2_alpha_pairs[:, 0] == alpha1) &
#         (a1eq2_alpha_pairs[:, 1] == alpha2)
#     )[0]
#     if len(idx) == 0:
#         continue
#     idx = idx[0]
#     a_fft_485 = a1eq2_a_fft[idx, :]
#     norm_485 = np.trapz(a_fft_485, omegas_485)

#     plt.plot(
#         omegas_485 * HARTREE_TO_EV,
#         a_fft_485 / HARTREE_TO_EV,
#         color="cyan",
#         linestyle="--",
#         label=rf'BWs2+C; epep@mf ($\alpha_1={alpha1}$, $\alpha_2={alpha2}$)'
#     )
plt.ylabel('Spectral Function (1/eV)')
plt.title(f'{title} - eta=0.8 eV')
plt.legend()
plt.tight_layout()
plt.xlim(-20, -4)
plt.ylim(0, 0.4)
plt.savefig('refpgw.png', dpi=300)
# plt.savefig('scgw_epepqp.png', dpi=300)