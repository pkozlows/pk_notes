import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from ueg_plt import load_hdf5_amp_ne_flow, load_hdf5_sps_ne_flow, merge_aw_with_sps, pack_res_to_arrays
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
HARTREE_TO_EV = 27.2114
r = 4
p_idx = 0
nocc = 57
k_label = r"$k=(0,0,0)$" if p_idx == 0 else r"$k=(2\pi/L,0,0)$"
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
def extract_and_squeeze(h5file, dataset_names):
    """
    Extracts datasets from an h5py file and squeezes singleton dimensions.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file handle.
    dataset_names : list of str
        Names of datasets to extract.

    Returns
    -------
    dict
        Dictionary {name: numpy_array_or_scalar}
        Arrays are squeezed to remove singleton dimensions.
        Scalars are returned as Python floats.
    """
    data = {}

    for name in dataset_names:
        arr = np.array(h5file[name])
        arr = np.squeeze(arr)

        # Convert 0-d array to scalar
        if arr.shape == ():
            arr = arr.item()

        data[name] = arr

    return data
fl10_name = "cu_fl10_rs4_npw485.h5"
fl1_name = "cu_fl1_rs4_npw485.h5"
inspect_hdf5(fl10_name)
inspect_hdf5(fl1_name)
datasets = [
    "hf_moes",
    "omegas_p0",
    "spec_fn_cumulant_p0",
    "spec_fn_g0_p0",
    "srg_mp2_flow",
    "srg_mp2_moes",
    "srg_mp2_niters"
]
with h5py.File(fl10_name, "r") as f:
    fl10_data = extract_and_squeeze(f, datasets)
with h5py.File(fl1_name, "r") as f:
    fl1_data = extract_and_squeeze(f, datasets)
print("FL10 data:")
for key, value in fl10_data.items():
    print(f"{key}: shape={np.shape(value)}, dtype={type(value)}, value={value if np.isscalar(value) else 'array'}")
print("\nFL1 data:")
for key, value in fl1_data.items():
    print(f"{key}: shape={np.shape(value)}, dtype={type(value)}, value={value if np.isscalar(value) else 'array'}")

# Read the CSV file using numpy
data = np.genfromtxt('fig2.csv', delimiter=',', skip_header=2, filling_values=np.nan)
# Extract data for each method (columns are: ccsd_X, ccsd_Y, hf_gw_X, hf_gw_Y, lda_gw_X, lda_gw_Y, lda_gw_c_X, lda_gw_c_Y)
ccsd_x = data[:, 0]
ccsd_y = data[:, 1]
lda_gw_c_x = data[:, 6]
lda_gw_c_y = data[:, 7]
# Increase the default line width for the entire plot
plt.rcParams['axes.linewidth'] = 2 

fig, ax = plt.subplots(figsize=(10, 6))

# Define font sizes
LABEL_SIZE = 28  # Slightly bigger for poster impact
LEGEND_SIZE = 22
LINE_WIDTH = 4   # Thick lines for visibility
EQUATION_SIZE = 26 # Size for the LaTeX formula

system = '114e/485o'

# Plotting
plt.plot(ccsd_x, ccsd_y, label='CCSD', color='blue', linewidth=LINE_WIDTH)
plt.plot(lda_gw_c_x, lda_gw_c_y, label='GW+C', color='orange', linewidth=LINE_WIDTH)

# --- ADDING THE LATEX EQUATION ---
# Coordinates (0.95, 0.05) places it in the bottom-right corner
# 'transform=ax.transAxes' ensures 0,0 is bottom-left and 1,1 is top-right
plt.text(0.95, 0.05, r'$G(t) = G_0(t)e^{C(t)}$', 
         transform=ax.transAxes, 
         fontsize=EQUATION_SIZE, 
         color='black',
         verticalalignment='bottom',
         horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) # Optional: adds a slight background if it overlaps lines

# Labels
plt.xlabel('Frequency', fontsize=LABEL_SIZE, fontweight='bold', labelpad=15)
plt.ylabel('Intensity', fontsize=LABEL_SIZE, fontweight='bold', labelpad=15)

# Legend
plt.legend(fontsize=LEGEND_SIZE, frameon=False, loc='upper left')

# Remove all tick labels (numbers) as requested
plt.xticks([])
plt.yticks([])

# Remove tick marks and grid lines
plt.tick_params(axis='both', which='both', length=0)
plt.grid(False)

# Setting limits
rightmost_valid_index = np.where(~np.isnan(ccsd_x))[0][-1]
plt.xlim(-20, ccsd_x[rightmost_valid_index])
plt.ylim(0, None)

plt.tight_layout()

# Save with high DPI
plt.savefig('fig2_infinite_order.png', dpi=600, bbox_inches='tight')
# plt.figure(figsize=(10, 6))
# system='114e/485o'
# plt.plot(ccsd_x, ccsd_y, label='CCSD', color='blue')
# plt.plot(lda_gw_c_x, lda_gw_c_y, label='GW+C', color='orange')
# # plt.plot(fl10_data["omegas_p0"] * HARTREE_TO_EV, fl10_data["spec_fn_cumulant_p0"] / HARTREE_TO_EV, label='SRG-MP2+C (flow=10)', color='green')
# # plt.plot(fl1_data["omegas_p0"] * HARTREE_TO_EV, fl1_data["spec_fn_cumulant_p0"] / HARTREE_TO_EV, label='SRG-MP2+C (flow=1)', color='red')
# # # show location of the srg-mp2 poles for flow=10 and flow=1
# # plt.axvline(fl10_data["srg_mp2_moes"][0] * HARTREE_TO_EV, color='green', linestyle='-', label='SRG-MP2 pole (flow=10)')
# # plt.axvline(fl1_data["srg_mp2_moes"][0] * HARTREE_TO_EV, color='red', linestyle='-', label='SRG-MP2 pole (flow=1)')
# # # find loc of the argmax of spec_fn_g0_p0
# # hf_pole = fl10_data["omegas_p0"][np.argmax(fl10_data["spec_fn_g0_p0"])]
# # plt.axvline(hf_pole * HARTREE_TO_EV, color='purple', linestyle='--', label='HF pole')
# plt.xlabel('Frequency')
# plt.ylabel('Intensity')
# # plt.title(f'Cumulant spectral function to infinite order at rs={r}, {k_label}, {system}, $\eta=0.8$ eV')
# plt.legend()
# # find the last not nan point in ccsd_x to set xlim
# rightmost_valid_index = np.where(~np.isnan(ccsd_x))[0][-1]
# plt.xlim(-20, ccsd_x[rightmost_valid_index] + 1)  # add a small margin to the right
# plt.ylim(0, None)
# plt.grid()
# plt.tight_layout()
# plt.savefig('fig2_infinite_order.png', dpi=300)

nes=(38,54)
flows=[0,100]
parts=["total"]
aw_out  = load_hdf5_amp_ne_flow("ce_conv_npw257_rs4.h5", nw=401, nes=nes, flows=flows, parts=parts)
sps_out = load_hdf5_sps_ne_flow("ce_sps_conv_npw257_rs4.h5", npw=257, nes=nes, flows=flows)

out = merge_aw_with_sps(aw_out, sps_out)
omegas, nes, flows, niters, a, poles = pack_res_to_arrays(
    out,
    nes=nes,
    flows=flows,
    parts=parts,
    return_sps=True,
)
plt.figure(figsize=(10, 6))
# system='54e/257o'
plt.plot(ccsd_x, ccsd_y, label='CCSD', color='blue')
# plt.plot(omegas*HARTREE_TO_EV, a[0,0,0,:]/HARTREE_TO_EV, label=rf'SRG-MP2+C ne=54, (flow=0)$\equiv$ HF-MP2+C', color='blue')
# plt.plot(omegas*HARTREE_TO_EV, a[0,0,1,:]/HARTREE_TO_EV, label=rf'SRG-MP2+C ne=54, (flow=10)', color='orange')
# plt.plot(omegas*HARTREE_TO_EV, a[0,1,0,:]/HARTREE_TO_EV, label=rf'SRG-MP2+C ne=38, (flow=0)$\equiv$ HF-MP2+C', color='cyan')
# plt.plot(omegas*HARTREE_TO_EV, a[0,1,1,:]/HARTREE_TO_EV, label=rf'SRG-MP2+C ne=38, (flow=10)', color='magenta')
with h5py.File("ce_hf^c2_npw485_rs4.h5", "r") as f:
    omegas_hf = np.array(np.squeeze(f["omegas"][:]))
    a_2h1p = np.array(np.squeeze(f["a_2h1p_ne114_flow0"][:]))
    a_2p1h = np.array(np.squeeze(f["a_2p1h_ne114_flow0"][:]))
    a_qp = np.array(np.squeeze(f["a_qp_ne114_flow0"][:]))
    a_total = np.array(np.squeeze(f["a_total_ne114_flow0"][:]))
# plt.plot(omegas_hf*HARTREE_TO_EV, a_2h1p/HARTREE_TO_EV, label='HF-MP2+C 2h1p', color='green')
# plt.plot(omegas_hf*HARTREE_TO_EV, a_2p1h/HARTREE_TO_EV, label='HF-MP2+C 2p1h', color='red')
# plt.plot(omegas_hf*HARTREE_TO_EV, a_qp/HARTREE_TO_EV, label='HF-MP2+C QP', color='cyan')
plt.plot(omegas_hf*HARTREE_TO_EV, a_total/HARTREE_TO_EV, label='HF+C(2) to first order', color='magenta')
with h5py.File("cu_dt0.05_rs4_npw485_thresh0.0001_wrange150.h5", "r") as f:
    omegas_p0 = np.array(np.squeeze(f["omegas_p0"][:]))
    spec_fn_g0_p0 = np.array(np.squeeze(f["spec_fn_g0_p0"][:]))
    spec_fn_cumulant_analytic_p0 = np.array(np.squeeze(f["spec_fn_cumulant_analytic_p0"][:]))
plt.plot(omegas_p0*HARTREE_TO_EV, spec_fn_g0_p0/HARTREE_TO_EV, label='HF spectral function', color='cyan')
plt.plot(omegas_p0*HARTREE_TO_EV, spec_fn_cumulant_analytic_p0/HARTREE_TO_EV, label='HF+C(2) to infinite order', color='green')
# plt.plot(omegas*HARTREE_TO_EV, a[0,0,2,:]/HARTREE_TO_EV, label='SRG-MP2+C (flow=50)', color='green')
# plt.plot(omegas*HARTREE_TO_EV, a[0,0,3,:]/HARTREE_TO_EV, label='SRG-MP2+C (flow=100)', color='red')
# plt.axvline(poles[0,0]*HARTREE_TO_EV, color='blue', linestyle='-', label=rf'SRG-MP2 pole (flow=0)$\equiv$ HF pole')
# plt.axvline(poles[0,1]*HARTREE_TO_EV, color='orange', linestyle='-', label='SRG-MP2 pole (flow=10)')
# plt.axvline(poles[0,2]*HARTREE_TO_EV, color='green', linestyle='-', label='SRG-MP2 pole (flow=50)')
# plt.axvline(poles[0,3]*HARTREE_TO_EV, color='red', linestyle='-', label='SRG-MP2 pole (flow=100)')
# plt.plot(omegas*HARTREE_TO_EV, a[0,0,1,:]/HARTREE_TO_EV, label='SRG-MP2+C (flow=100)', color='orange')
# plt.axvline(poles[0,0]*HARTREE_TO_EV, color='blue', linestyle='-', label=rf'SRG-MP2 pole (flow=0)$\equiv$ HF pole')
# plt.axvline(poles[0,1]*HARTREE_TO_EV, color='orange', linestyle='-', label='SRG-MP2 pole (flow=100)')
plt.xlabel('Energy (eV)')
plt.ylabel('Spectral Function (1/eV)')
plt.title(rf'Cumulant spectral function to first order at rs={r}, {k_label}, {system}, $\eta=0.8$ eV')
plt.legend()
plt.xlim(-20, 0)
plt.ylim(0, None)
plt.grid()
plt.tight_layout()
plt.savefig('fig2_first_order.png', dpi=300)
