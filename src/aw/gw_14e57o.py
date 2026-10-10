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
nocc = 7
k_label = r"$k=(0,0,0)$" if p_idx == 0 else r"$k=(2\pi/L,0,0)$"
title = rf"14e/57o {k_label} at $r_s={r}$"
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
python_gw_cache = "gw_spectral_ne14_nbsf57_rs4.h5"
inspect_hdf5(python_gw_cache)

with h5py.File(python_gw_cache, "r") as f:
    python_omegas = f['omega'][:]
    python_spectral = f['spectral'][:][0].flatten()
cpp_gw_cache = "ne14_rs4_gw_cache_cpp.h5"
inspect_hdf5(cpp_gw_cache)
with h5py.File(cpp_gw_cache, "r") as f:
    cpp_omegas_hf = f['hf_omegas_ne14'][:].flatten()
    cpp_g0w0_spec_hf = f['hf_g0w0_spec_ne14'][:].flatten()
    cpp_g0w0_spec_xB_hf = f['hf_g0w0_spec_xB_ne14'][:].flatten()
cpp_gwac='14e57o_gwac.h5'
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
'''
with h5py.File(cpp_gwac, "r") as f:
    cpp_g0w0_spec_ne14_ac = f['hf_g0w0_spec_ne14'][:].flatten()
    cpp_g0w0_spec_xB_ne14_ac = f['hf_g0w0_spec_xB_ne14'][:].flatten()
    cpp_omegas_ne14_ac = f['hf_omegas_ne14'][:].flatten()
py_gwcd_nw400_hf = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/14e57o_nwcd400_HF_py.h5'
inspect_hdf5(py_gwcd_nw400_hf)
'''/ce_omegas               Dataset {7997}
/gw_spec_cd              Dataset {280}
/omegas                  Dataset {280}
/qp_g0w0_cd              Dataset {SCALAR}
/spec_fn_g0w0pc          Dataset {7997}
'''
with h5py.File(py_gwcd_nw400_hf, "r") as f:
    py_gwcd_nw400_hf_omegas_mC = f['omegas'][:].flatten()
    py_gwcd_nw400_hf_g0w0_spec = f['gw_spec_cd'][:].flatten()
    py_gwcd_nw400_hf_omegas_pC = f['ce_omegas'][:].flatten()
    py_gwcd_nw400_hf_g0w0pC_qpepep = f['spec_fn_g0w0pc'][:].flatten()
    py_gwcd_nw400_hf_qpe_g0w0 = f['qp_g0w0_cd'][()]
cpp_gwcd_nw400_hf = '/Users/patrykkozlowski/Library/CloudStorage/OneDrive-HarvardUniversity/Attachments/pk_notes/src/aw/14e57o_nwcd400_HF_cpp.h5'
inspect_hdf5(cpp_gwcd_nw400_hf)
'''
DATASET: hf_g0w0_spec_ne14 (1, 280) float64
DATASET: hf_g0w0pC_mfepep_ne14 (1, 7997) float64
DATASET: hf_g0w0pC_qpepep_ne14 (1, 7997) float64
DATASET: hf_omegas_mC_ne14 (1, 280) float64
DATASET: hf_omegas_pC_ne14 (1, 7997) float64
DATASET: hf_qpes_g0w0_ne14 (1, 1) float64
'''
with h5py.File(cpp_gwcd_nw400_hf, "r") as f:
    cpp_gwcd_nw400_hf_omegas_mC = f['hf_omegas_mC_ne14'][:].flatten()
    cpp_gwcd_nw400_hf_g0w0_spec = f['hf_g0w0_spec_ne14'][:].flatten()
    cpp_gwcd_nw400_hf_omegas_pC = f['hf_omegas_pC_ne14'][:].flatten()
    cpp_gwcd_nw400_hf_g0w0pC_qpepep = f['hf_g0w0pC_qpepep_ne14'][:].flatten()
    cpp_gwcd_nw400_hf_qpe_g0w0 = f['hf_qpes_g0w0_ne14'][()].squeeze()
# Print the difference between the qpes
qpe_diff = py_gwcd_nw400_hf_qpe_g0w0 - cpp_gwcd_nw400_hf_qpe_g0w0
print(f"Difference in QP energy: {qpe_diff:.6e} Hartree, {qpe_diff*HARTREE_TO_EV:.6e} eV")
# Get the difference between the python and C++ HF+G0W0 spectra
difference = py_gwcd_nw400_hf_g0w0_spec - cpp_gwcd_nw400_hf_g0w0_spec
# get the maximum absolute difference and its frequency At which it occurs  as well as the mean
max_diff = np.max(np.abs(difference))
max_diff_freq = py_gwcd_nw400_hf_omegas_mC[np.argmax(np.abs(difference))]*HARTREE_TO_EV
mean_diff = np.mean(difference)
print(f"Max absolute difference: {max_diff:.6e} at frequency {max_diff_freq:.2f} eV, mean difference: {mean_diff:.6e}")
plt.figure(figsize=(10, 6))
plt.plot(py_gwcd_nw400_hf_omegas_mC*HARTREE_TO_EV, py_gwcd_nw400_hf_g0w0_spec/HARTREE_TO_EV, label='My PY HF+G0W0', color='blue', linestyle='-', marker=None)
plt.plot(cpp_gwcd_nw400_hf_omegas_mC*HARTREE_TO_EV, cpp_gwcd_nw400_hf_g0w0_spec/HARTREE_TO_EV, label='My CPP HF+G0W0', color='red', linestyle='--', marker=None)
plt.xlabel('Energy (eV)')
plt.ylabel('Spectral Function (1/eV)')
plt.title(f'{title} - eta=0.2 eV')
plt.legend()
plt.tight_layout()
plt.xlim(-18, -4)
plt.ylim(0, 1.5)
plt.savefig('gw_14e57o.png', dpi=300)