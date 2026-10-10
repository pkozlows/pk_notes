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
nocc = 19
k_label = r"$k=(0,0,0)$" if p_idx == 0 else r"$k=(2\pi/L,0,0)$"
title = rf"HF+GW at 38e/123o with AC; {k_label} and $r_s={r}$"
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
cpp_gwac_npade20_nw40 = "38e123o_nw40_npade20_gwac.h5"
with h5py.File(cpp_gwac_npade20_nw40, "r") as f:
    cpp_g0w0_spec_ne38_nw40_ac = f['hf_g0w0_spec_ne38'][:].flatten()
    cpp_omegas_ne38_nw40_ac = f['hf_omegas_ne38'][:].flatten()
cpp_gwac_npade30_nw60 = "38e123o_nw60_npade30_gwac.h5"
with h5py.File(cpp_gwac_npade30_nw60, "r") as f:
    cpp_g0w0_spec_ne38_nw60_ac = f['hf_g0w0_spec_ne38'][:].flatten()
    cpp_omegas_ne38_nw60_ac = f['hf_omegas_ne38'][:].flatten()
cpp_gwac_npade40_nw80 = "38e123o_nw80_npade40_gwac.h5"
with h5py.File(cpp_gwac_npade40_nw80, "r") as f:
    cpp_g0w0_spec_ne38_nw80_ac = f['hf_g0w0_spec_ne38'][:].flatten()
    cpp_omegas_ne38_nw80_ac = f['hf_omegas_ne38'][:].flatten()
cpp_gwac_npade50_nw100 = "38e123o_nw100_npade50_gwac.h5"
with h5py.File(cpp_gwac_npade50_nw100, "r") as f:
    cpp_g0w0_spec_ne38_nw100_ac = f['hf_g0w0_spec_ne38'][:].flatten()
    cpp_omegas_ne38_nw100_ac = f['hf_omegas_ne38'][:].flatten()
cpp_gwac_npade100_nw200 = "38e123o_nw200_npade100_gwac.h5"
with h5py.File(cpp_gwac_npade100_nw200, "r") as f:
    cpp_g0w0_spec_ne38_nw200_ac = f['hf_g0w0_spec_ne38'][:].flatten()
    cpp_omegas_ne38_nw200_ac = f['hf_omegas_ne38'][:].flatten()
plt.figure(figsize=(10, 6))
plt.plot(cpp_omegas_ne38_nw40_ac * HARTREE_TO_EV, cpp_g0w0_spec_ne38_nw40_ac/HARTREE_TO_EV, label='NW=40, NPade=20', color='blue', linestyle='-')
plt.plot(cpp_omegas_ne38_nw60_ac * HARTREE_TO_EV, cpp_g0w0_spec_ne38_nw60_ac/HARTREE_TO_EV, label='NW=60, NPade=30', color='orange', linestyle='--')
plt.plot(cpp_omegas_ne38_nw80_ac * HARTREE_TO_EV, cpp_g0w0_spec_ne38_nw80_ac/HARTREE_TO_EV, label='NW=80, NPade=40', color='green', linestyle=':')
plt.plot(cpp_omegas_ne38_nw100_ac * HARTREE_TO_EV, cpp_g0w0_spec_ne38_nw100_ac/HARTREE_TO_EV, label='NW=100, NPade=50', color='red', linestyle='-.')
plt.plot(cpp_omegas_ne38_nw200_ac * HARTREE_TO_EV, cpp_g0w0_spec_ne38_nw200_ac/HARTREE_TO_EV, label='NW=200, NPade=100', color='purple', linestyle='-')
plt.xlabel('Energy (eV)')
plt.ylabel('Spectral Function (1/eV)')
plt.title(f'{title} - eta=0.4 eV')
plt.legend()
plt.tight_layout()
plt.xlim(-20, -4)
plt.ylim(0, 0.6)
plt.savefig('gw_38e123o.png', dpi=300)