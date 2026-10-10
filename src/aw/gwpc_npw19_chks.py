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
nocc = 57
k_label = r"$k=(0,0,0)$" if p_idx == 0 else r"$k=(2\pi/L,0,0)$"
title = rf"Spectral functions for {k_label} at $r_s={r}$"
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
#fig1a refs
garnet_data = np.genfromtxt("fig1a.csv", delimiter=',', skip_header=2, filling_values=np.nan)
hf_gw_x_1a = garnet_data[:, 0]
hf_gw_y_1a = garnet_data[:, 1]
lda_gwc_x_1a = garnet_data[:, 2]
lda_gwc_y_1a = garnet_data[:, 3]
ccsd_x_1a = garnet_data[:, 4]
ccsd_y_1a = garnet_data[:, 5]
lda_gw_x_1a = garnet_data[:, 6]
lda_gw_y_1a = garnet_data[:, 7]
ref_data_extra = np.genfromtxt('digitized_curves.csv', delimiter=',', skip_header=0, filling_values=np.nan)
lda_gw_x_extra = ref_data_extra[:, 0]
lda_gw_y_extra = ref_data_extra[:, 1]
# exact_data= 'lanczos_gpt.csv'
exact_data = np.genfromtxt('lanczos_gpt.csv', delimiter=',', skip_header=0, filling_values=np.nan)
# import python data
pyscf_gw_cd = 'gw_spectral_ne14_nbsf19_rs4_cd.h5'
inspect_hdf5(pyscf_gw_cd)
'''
DATASET: omega (5001,) float64
DATASET: sigma_im_cd (5001,) float64
DATASET: sigma_re_cd (5001,) float64
DATASET: spectral_cd (5001,) float64
'''
with h5py.File(pyscf_gw_cd, "r") as f:
    python_omegas = f['omega'][:]
    python_spectral_cd = f['spectral_cd'][:].flatten()
pyscf_gw_cd_bw='gw_spectral_ne14_cd_bw.h5'
inspect_hdf5(pyscf_gw_cd_bw)
with h5py.File(pyscf_gw_cd_bw, "r") as f:
    python_omegas_bw = f['omega'][:]
    python_spectral_cd_bw = f['spectral_cd'][:].flatten()
pyscf_gw_cd_to_nw50 = 'gw_spectral_ne14_nbsf19_rs4_etasign.h5'
inspect_hdf5(pyscf_gw_cd_to_nw50)
with h5py.File(pyscf_gw_cd_to_nw50, "r") as f:
    python_omegas_to = f['omega'][:]
    python_spectral_to = f['spectral_cd'][:].flatten()
pyscf_gw_cd_to_nw100 = 'gw_spectral_ne14_nbsf19_rs4_to_nw100.h5'
inspect_hdf5(pyscf_gw_cd_to_nw100)
with h5py.File(pyscf_gw_cd_to_nw100, "r") as f:
    python_omegas_to_nw100 = f['omega'][:]
    python_spectral_to_nw100 = f['spectral_cd'][:].flatten()
pyscf_gw_cd_to_nw500 = 'gw_spectral_ne14_nbsf19_rs4_to_nw500.h5'
inspect_hdf5(pyscf_gw_cd_to_nw500)
with h5py.File(pyscf_gw_cd_to_nw500, "r") as f:
    python_omegas_to_nw500 = f['omega'][:]
    python_spectral_to_nw500 = f['spectral_cd'][:].flatten()
python_gw_cache = "ne14_rs4_gw_cache_python.h5"
inspect_hdf5(python_gw_cache)
with h5py.File(python_gw_cache, "r") as f:
    omegas = f['grid/ce_omegas'][:]
    qpes_g0w_hf = f['hf/qp_energies/g0w'][:]
    qpes_gw0_hf = f['hf/qp_energies/gw0'][:]
    qpes_gw_hf = f['hf/qp_energies/gw'][:]
    gw_spec_fn_hf = f['hf/gw_spec_fn'][:]
    g0w0pC_qpepep_hf = f['hf/cumulant/g0w0_qp_epep/spectral_fn'][:]
    g0wpC_qpepep_hf = f['hf/cumulant/g0w_qp_epep/spectral_fn'][:]
    gw0pC_qpepep_hf = f['hf/cumulant/gw0_qp_epep/spectral_fn'][:]
    gwpC_qpepep_hf = f['hf/cumulant/gw_qp_epep/spectral_fn'][:]
    g0w0pC_mfepep_hf = f['hf/cumulant/g0w0_mf_epep/spectral_fn'][:]
    g0wpC_mfepep_hf = f['hf/cumulant/g0w_mf_epep/spectral_fn'][:]
    gw0pC_mfepep_hf = f['hf/cumulant/gw0_mf_epep/spectral_fn'][:]
    gwpC_mfepep_hf = f['hf/cumulant/gw_mf_epep/spectral_fn'][:]
    qpes_g0w_lda = f['lda/qp_energies/g0w'][:]
    qpes_gw0_lda = f['lda/qp_energies/gw0'][:]
    qpes_gw_lda = f['lda/qp_energies/gw'][:]
    gw_spec_fn_lda = f['lda/gw_spec_fn'][:]
    g0w0pC_qpepep_lda = f['lda/cumulant/g0w0_qp_epep/spectral_fn'][:]
    g0wpC_qpepep_lda = f['lda/cumulant/g0w_qp_epep/spectral_fn'][:]
    gw0pC_qpepep_lda = f['lda/cumulant/gw0_qp_epep/spectral_fn'][:]
    gwpC_qpepep_lda = f['lda/cumulant/gw_qp_epep/spectral_fn'][:]   
    g0w0pC_mfepep_lda = f['lda/cumulant/g0w0_mf_epep/spectral_fn'][:]
    g0wpC_mfepep_lda = f['lda/cumulant/g0w_mf_epep/spectral_fn'][:]
    gw0pC_mfepep_lda = f['lda/cumulant/gw0_mf_epep/spectral_fn'][:]
    gwpC_mfepep_lda = f['lda/cumulant/gw_mf_epep/spectral_fn'][:]
cpp_gw_cache = "ne14_rs4_gw_cache_cpp.h5"
inspect_hdf5(cpp_gw_cache)
'''
DATASET: hf_g0w0_spec_ne14 (1, 7997) float64
DATASET: hf_g0w0_spec_xB_ne14 (1, 7997) float64
DATASET: hf_g0w0pC_mfepep_ne14 (1, 7997) float64
DATASET: hf_g0w0pC_qpepep_ne14 (1, 7997) float64
DATASET: hf_g0wpC_mfepep_ne14 (1, 7997) float64
DATASET: hf_g0wpC_qpepep_ne14 (1, 7997) float64
DATASET: hf_gw0pC_mfepep_ne14 (1, 7997) float64
DATASET: hf_gw0pC_qpepep_ne14 (1, 7997) float64
DATASET: hf_gwpC_mfepep_ne14 (1, 7997) float64
DATASET: hf_gwpC_qpepep_ne14 (1, 7997) float64
DATASET: hf_omegas_ne14 (1, 7997) float64
DATASET: hf_qpes_g0w0_ne14 (1, 19) float64
DATASET: hf_qpes_g0w_ne14 (1, 19) float64
DATASET: hf_qpes_gw0_ne14 (1, 19) float64
DATASET: hf_qpes_gw_ne14 (1, 19) float64
DATASET: lda_g0w0_spec_ne14 (1, 7997) float64
DATASET: lda_g0w0_spec_xB_ne14 (1, 7997) float64
DATASET: lda_g0w0pC_mfepep_ne14 (1, 7997) float64
DATASET: lda_g0w0pC_qpepep_ne14 (1, 7997) float64
DATASET: lda_g0wpC_mfepep_ne14 (1, 7997) float64
DATASET: lda_g0wpC_qpepep_ne14 (1, 7997) float64
DATASET: lda_gw0pC_mfepep_ne14 (1, 7997) float64
DATASET: lda_gw0pC_qpepep_ne14 (1, 7997) float64
DATASET: lda_gwpC_mfepep_ne14 (1, 7997) float64
DATASET: lda_gwpC_qpepep_ne14 (1, 7997) float64
DATASET: lda_omegas_ne14 (1, 7997) float64
DATASET: lda_qpes_g0w0_ne14 (1, 19) float64
DATASET: lda_qpes_g0w_ne14 (1, 19) float64
DATASET: lda_qpes_gw0_ne14 (1, 19) float64
DATASET: lda_qpes_gw_ne14 (1, 19) float64
'''
with h5py.File(cpp_gw_cache, "r") as f:
    cpp_omegas_hf = f['hf_omegas_ne14'][:].flatten()
    cpp_g0w0_spec_hf = f['hf_g0w0_spec_ne14'][:].flatten()
    cpp_g0w0_spec_xB_hf = f['hf_g0w0_spec_xB_ne14'][:].flatten()
    cpp_g0w0pC_qpepep_hf = f['hf_g0w0pC_qpepep_ne14'][:].flatten()
    cpp_g0wpC_qpepep_hf = f['hf_g0wpC_qpepep_ne14'][:].flatten()
    cpp_gw0pC_qpepep_hf = f['hf_gw0pC_qpepep_ne14'][:].flatten()
    cpp_gwpC_qpepep_hf = f['hf_gwpC_qpepep_ne14'][:].flatten()
    cpp_g0w0pC_mfepep_hf = f['hf_g0w0pC_mfepep_ne14'][:].flatten()
    cpp_g0wpC_mfepep_hf = f['hf_g0wpC_mfepep_ne14'][:].flatten()
    cpp_gw0pC_mfepep_hf = f['hf_gw0pC_mfepep_ne14'][:].flatten()
    cpp_gwpC_mfepep_hf = f['hf_gwpC_mfepep_ne14'][:].flatten()
    cpp_omegas_lda = f['lda_omegas_ne14'][:].flatten()
    cpp_g0w0_spec_lda = f['lda_g0w0_spec_ne14'][:].flatten()
    cpp_g0w0_spec_xB_lda = f['lda_g0w0_spec_xB_ne14'][:].flatten()
    cpp_g0w0pC_qpepep_lda = f['lda_g0w0pC_qpepep_ne14'][:].flatten()
    cpp_g0wpC_qpepep_lda = f['lda_g0wpC_qpepep_ne14'][:].flatten()
    cpp_gw0pC_qpepep_lda = f['lda_gw0pC_qpepep_ne14'][:].flatten()
    cpp_gwpC_qpepep_lda = f['lda_gwpC_qpepep_ne14'][:].flatten()
    cpp_g0w0pC_mfepep_lda = f['lda_g0w0pC_mfepep_ne14'][:].flatten()
    cpp_g0wpC_mfepep_lda = f['lda_g0wpC_mfepep_ne14'][:].flatten()
    cpp_gw0pC_mfepep_lda = f['lda_gw0pC_mfepep_ne14'][:].flatten()
    cpp_gwpC_mfepep_lda = f['lda_gwpC_mfepep_ne14'][:].flatten()
    cpp_qpes_g0w0_lda = f['lda_qpes_g0w0_ne14'][:].flatten()
    cpp_qpes_g0w_lda = f['lda_qpes_g0w_ne14'][:].flatten()
    cpp_qpes_gw0_lda = f['lda_qpes_gw0_ne14'][:].flatten()
    cpp_qpes_gw_lda = f['lda_qpes_gw_ne14'][:].flatten()
pyscf_gw = 'gw_spectral_ne14_nbsf19_rs4.h5'
inspect_hdf5(pyscf_gw)
'''
Top-level keys: ['hf_spes', 'omega', 'qp_energy', 'sigma_im', 'sigma_re', 'spectral']
DATASET: hf_spes (19,) float64
DATASET: omega (1001,) float64
DATASET: qp_energy (1, 19) float64
DATASET: sigma_im (19, 1001) float64
DATASET: sigma_re (19, 1001) float64
DATASET: spectral (19, 1001) float64
'''
with h5py.File(pyscf_gw, "r") as f:
    pyscf_omegas = f['omega'][:]
    pyscf_spectral = f['spectral'][:][0].flatten()

cpp_gwac = '14e19o_gwac.h5'
inspect_hdf5(cpp_gwac)
'''
DATASET: hf_g0w0_spec_ne14 (1, 7997) float64
DATASET: hf_g0w0_spec_xB_ne14 (1, 7997) float64
DATASET: hf_g0w0pC_mfepep_ne14 (1, 7997) float64
DATASET: hf_g0w0pC_qpepep_ne14 (1, 7997) float64
DATASET: hf_g0wpC_mfepep_ne14 (1, 7997) float64
DATASET: hf_g0wpC_qpepep_ne14 (1, 7997) float64
DATASET: hf_gw0pC_mfepep_ne14 (1, 7997) float64
DATASET: hf_gw0pC_qpepep_ne14 (1, 7997) float64
DATASET: hf_gwpC_mfepep_ne14 (1, 7997) float64
DATASET: hf_gwpC_qpepep_ne14 (1, 7997) float64
DATASET: hf_omegas_ne14 (1, 7997) float64
DATASET: hf_qpes_g0w0_ne14 (1, 19) float64
DATASET: hf_qpes_g0w_ne14 (1, 19) float64
DATASET: hf_qpes_gw0_ne14 (1, 19) float64
DATASET: hf_qpes_gw_ne14 (1, 19) float64
DATASET: lda_g0w0_spec_ne14 (1, 0) float64
DATASET: lda_g0w0_spec_xB_ne14 (1, 0) float64
DATASET: lda_g0w0pC_mfepep_ne14 (1, 0) float64
DATASET: lda_g0w0pC_qpepep_ne14 (1, 0) float64
DATASET: lda_g0wpC_mfepep_ne14 (1, 0) float64
DATASET: lda_g0wpC_qpepep_ne14 (1, 0) float64
DATASET: lda_gw0pC_mfepep_ne14 (1, 0) float64
DATASET: lda_gw0pC_qpepep_ne14 (1, 0) float64
DATASET: lda_gwpC_mfepep_ne14 (1, 0) float64
DATASET: lda_gwpC_qpepep_ne14 (1, 0) float64
DATASET: lda_omegas_ne14 (1, 0) float64
DATASET: lda_qpes_g0w0_ne14 (1, 0) float64
DATASET: lda_qpes_g0w_ne14 (1, 0) float64
DATASET: lda_qpes_gw0_ne14 (1, 0) float64
DATASET: lda_qpes_gw_ne14 (1, 0) float64
'''
with h5py.File(cpp_gwac, "r") as f:
    cpp_omegas_hf_ac = f['hf_omegas_ne14'][:].flatten()
    cpp_g0w0_spec_hf_ac = f['hf_g0w0_spec_ne14'][:].flatten()
    cpp_g0w0_spec_xB_hf_ac = f['hf_g0w0_spec_xB_ne14'][:].flatten()
# with h5py.File(cpp_gw_cache, "r") as f:
cpp_gwcd = '14e19o_gwcd.h5'
inspect_hdf5(cpp_gwcd)
with h5py.File(cpp_gwcd, "r") as f:
    cpp_omegas_hf_cd = f['hf_omegas_ne14'][:].flatten()
    cpp_g0w0_spec_hf_cd = f['hf_g0w0_spec_ne14'][:].flatten()
fast_gwcd = '14e19o_nwcd100.h5'
inspect_hdf5(fast_gwcd)
'''
DATASET: hf_g0w0_spec_ne14 (1, 2001) float64
DATASET: hf_g0w0pC_mfepep_ne14 (1, 797) float64
DATASET: hf_g0w0pC_qpepep_ne14 (1, 797) float64
DATASET: hf_g0wpC_mfepep_ne14 (1, 0) float64
DATASET: hf_g0wpC_qpepep_ne14 (1, 0) float64
DATASET: hf_gw0pC_mfepep_ne14 (1, 0) float64
DATASET: hf_gw0pC_qpepep_ne14 (1, 0) float64
DATASET: hf_gwpC_mfepep_ne14 (1, 0) float64
DATASET: hf_gwpC_qpepep_ne14 (1, 0) float64
DATASET: hf_omegas_mC_ne14 (1, 2001) float64
DATASET: hf_omegas_pC_ne14 (1, 797) float64
DATASET: hf_qpes_g0w0_ne14 (1, 19) float64
DATASET: hf_qpes_g0w_ne14 (1, 0) float64
DATASET: hf_qpes_gw0_ne14 (1, 0) float64
DATASET: hf_qpes_gw_ne14 (1, 0) float64
DATASET: lda_g0w0_spec_ne14 (1, 2001) float64
DATASET: lda_g0w0pC_mfepep_ne14 (1, 797) float64
DATASET: lda_g0w0pC_qpepep_ne14 (1, 797) float64
DATASET: lda_g0wpC_mfepep_ne14 (1, 0) float64
DATASET: lda_g0wpC_qpepep_ne14 (1, 0) float64
DATASET: lda_gw0pC_mfepep_ne14 (1, 0) float64
DATASET: lda_gw0pC_qpepep_ne14 (1, 0) float64
DATASET: lda_gwpC_mfepep_ne14 (1, 0) float64
DATASET: lda_gwpC_qpepep_ne14 (1, 0) float64
DATASET: lda_omegas_mC_ne14 (1, 2001) float64
DATASET: lda_omegas_pC_ne14 (1, 797) float64
DATASET: lda_qpes_g0w0_ne14 (1, 19) float64
DATASET: lda_qpes_g0w_ne14 (1, 0) float64
DATASET: lda_qpes_gw0_ne14 (1, 0) float64
DATASET: lda_qpes_gw_ne14 (1, 0) float64
'''
with h5py.File(fast_gwcd, "r") as f:
    fast_gwcd_omegas_mC = f['hf_omegas_mC_ne14'][:].flatten()
    fast_gwcd_omegas_pC = f['hf_omegas_pC_ne14'][:].flatten()
    fast_gwcd_g0w0_spec_hf = f['hf_g0w0_spec_ne14'][:].flatten()
    fast_gwcd_g0w0pC_mfepep_hf = f['hf_g0w0pC_mfepep_ne14'][:].flatten()
    fast_gwcd_g0w0pC_qpepep_hf = f['hf_g0w0pC_qpepep_ne14'][:].flatten()
    fast_gwcd_g0w0_spec_lda = f['lda_g0w0_spec_ne14'][:].flatten()
    fast_gwcd_g0w0pC_mfepep_lda = f['lda_g0w0pC_mfepep_ne14'][:].flatten()
    fast_gwcd_g0w0pC_qpepep_lda = f['lda_g0w0pC_qpepep_ne14'][:].flatten()
gwcd_coarse = '14e19o_nwcd100_coarse.h5'
with h5py.File(gwcd_coarse, "r") as f:
    gwcd_coarse_omegas_mC = f['hf_omegas_mC_ne14'][:].flatten()
    gwcd_coarse_g0w0_spec_hf = f['hf_g0w0_spec_ne14'][:].flatten()
    gwcd_coarse_g0w0_spec_lda = f['lda_g0w0_spec_ne14'][:].flatten()
cpp_gwcd_nw400_hf = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/14e19o_nwcd400_HF_cpp.h5'
inspect_hdf5(cpp_gwcd_nw400_hf)
with h5py.File(cpp_gwcd_nw400_hf, "r") as f:
    cpp_gwcd_nw400_hf_omegas_mC = np.squeeze(f['hf_omegas_mC_ne14'][:])
    cpp_gwcd_nw400_hf_g0w0_spec = np.squeeze(f['hf_g0w0_spec_ne14'][:])
    cpp_gwcd_nw400_hf_g0w0pC_qpepep = np.squeeze(f['hf_g0w0pC_qpepep_ne14'][:])
    cpp_gwcd_nw400_hf_g0w0pC_mfepep = np.squeeze(f['hf_g0w0pC_mfepep_ne14'][:])
    cpp_gwcd_nw400_hf_omegas_pC = np.squeeze(f['hf_omegas_pC_ne14'][:])
cpp_gwcd_nw400_lda = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/14e19o_nwcd400_LDA_cpp.h5'
inspect_hdf5(cpp_gwcd_nw400_lda)
with h5py.File(cpp_gwcd_nw400_lda, "r") as f:
    cpp_gwcd_nw400_lda_omegas_mC = np.squeeze(f['lda_omegas_mC_ne14'][:])
    cpp_gwcd_nw400_lda_g0w0_spec = np.squeeze(f['lda_g0w0_spec_ne14'][:])
    cpp_gwcd_nw400_lda_g0w0pC_qpepep = np.squeeze(f['lda_g0w0pC_qpepep_ne14'][:])
    cpp_gwcd_nw400_lda_g0w0pC_mfepep = np.squeeze(f['lda_g0w0pC_mfepep_ne14'][:])
    cpp_gwcd_nw400_lda_omegas_pC = np.squeeze(f['lda_omegas_pC_ne14'][:]) 
py_gwcd_nw400_hf = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/14e19o_nwcd400_HF_py.h5'
inspect_hdf5(py_gwcd_nw400_hf)
'''
DATASET: ce_omegas (7997,) float64
DATASET: gw_spec_cd (280,) float64
DATASET: omegas (280,) float64
DATASET: qp_g0w0_cd () float64
DATASET: spec_fn_g0w0pc (7997,) float64
'''
with h5py.File(py_gwcd_nw400_hf, "r") as f:
    py_gwcd_nw400_hf_omegas_mC = f['omegas'][:].flatten()
    py_gwcd_nw400_hf_g0w0_spec = f['gw_spec_cd'][:].flatten()
    py_gwcd_nw400_hf_omegas_pC = f['ce_omegas'][:].flatten()
    py_gwcd_nw400_hf_g0w0pC_qpepep = f['spec_fn_g0w0pc'][:].flatten()
py_gwcd_nw400_lda = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/14e19o_nwcd400_LDA_py.h5'
inspect_hdf5(py_gwcd_nw400_lda)
with h5py.File(py_gwcd_nw400_lda, "r") as f:
    py_gwcd_nw400_lda_omegas_mC = f['omegas'][:].flatten()
    py_gwcd_nw400_lda_g0w0_spec = f['gw_spec_cd'][:].flatten()
    py_gwcd_nw400_lda_omegas_pC = f['ce_omegas'][:].flatten()
    py_gwcd_nw400_lda_g0w0pC_qpepep = f['spec_fn_g0w0pc'][:].flatten()
plt.figure(figsize=(10, 6))
# plot fig1a refs
mask = ~np.isnan(hf_gw_y_1a)
norm_hf_gw_1a = np.trapz(hf_gw_y_1a[mask], hf_gw_x_1a[mask])
plt.plot(hf_gw_x_1a, hf_gw_y_1a, label=f'Garnet\'s HF+GW', color='blue', linestyle='none', marker='o')
plt.plot(lda_gwc_x_1a, lda_gwc_y_1a, label='Garnet\'s LDA+GW+C', color='orange', linestyle='none', marker='s')
# plt.plot(ccsd_x_1a, ccsd_y_1a, label='CCSD', color='green', linestyle='-', marker='^')
# plot extra refs
norm_lda_gw_extra = np.trapz(lda_gw_y_extra, lda_gw_x_extra)
plt.plot(lda_gw_x_extra, lda_gw_y_extra, label=f'Garnet\'s LDA+GW', color='orange', linestyle='none', marker='^')
plt.plot(cpp_gwcd_nw400_hf_omegas_mC*HARTREE_TO_EV, cpp_gwcd_nw400_hf_g0w0_spec/HARTREE_TO_EV, label='My CPP HF+G0W0', color='red', linestyle='-', marker=None)
plt.plot(cpp_gwcd_nw400_lda_omegas_mC*HARTREE_TO_EV, cpp_gwcd_nw400_lda_g0w0_spec/HARTREE_TO_EV, label='My CPP LDA+G0W0', color='green', linestyle='-', marker=None)
plt.plot(cpp_gwcd_nw400_lda_omegas_pC*HARTREE_TO_EV, cpp_gwcd_nw400_lda_g0w0pC_qpepep/HARTREE_TO_EV, label='My CPP LDA+GW+C', color='purple', linestyle='-', marker=None)
plt.plot(py_gwcd_nw400_hf_omegas_mC*HARTREE_TO_EV, py_gwcd_nw400_hf_g0w0_spec/HARTREE_TO_EV, label='My PY HF+G0W0', color='blue', linestyle=':', marker=None)
plt.plot(py_gwcd_nw400_lda_omegas_mC*HARTREE_TO_EV, py_gwcd_nw400_lda_g0w0_spec/HARTREE_TO_EV, label='My PY LDA+G0W0', color='cyan', linestyle=':', marker=None)
plt.plot(py_gwcd_nw400_lda_omegas_pC*HARTREE_TO_EV, py_gwcd_nw400_lda_g0w0pC_qpepep/HARTREE_TO_EV, label='My PY LDA+GW+C', color='yellow', linestyle=':', marker=None)
plt.ylabel('Spectral Function (1/eV)')
plt.title(f'{title} - eta=0.2 eV')
plt.legend()
plt.tight_layout()
plt.xlim(-18, -4)
plt.ylim(-0.1, 1.5)
plt.savefig('gwc_verification.png', dpi=300)