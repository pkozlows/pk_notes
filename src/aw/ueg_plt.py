import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
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
def _read_scalar_or_array(ds):
    """Armadillo often stores scalars as length-1 vectors in HDF5."""
    arr = ds[()]
    arr = np.asarray(arr)
    if arr.shape == ():          # true scalar
        return float(arr)
    if arr.size == 1:            # length-1 vector/matrix
        return float(arr.ravel()[0])
    return arr
def load_hdf5_for_pole_v2(filename: str, p: int):
    """
    Load only what is needed to plot pole p from the HDF5 produced by your C++ code:
      - hf_moes
      - srg_mp2_moes
      - srg_mp2_niters (scalar-ish)
      - srg_mp2_flow (scalar-ish)
      - spec_fn_g0_p{p}
      - spec_fn_cumulant_p{p}
      - omegas_p{p}
    """
    with h5py.File(filename, "r") as f:
        # hf_moes = np.asarray(f["hf_moes"][0, :])
        srg_mp2_moes = np.asarray(f["srg_mp2_moes"][0, :])

        flow = _read_scalar_or_array(f["srg_mp2_flow"])
        niters = _read_scalar_or_array(f["srg_mp2_niters"])

        cum_key = f"spec_fn_cumulant_analytic_p{p}"
        om_key = f"omegas_p{p}"

        if cum_key not in f:
            missing = [k for k in (cum_key) if k not in f]
            raise KeyError(f"Missing datasets for p={p}: {missing}")

        spec_fn_cumulant = np.asarray(f[cum_key][0, :])
        omegas = np.asarray(f[om_key][0, :])

    return {
        # "hf_moes": hf_moes,
        "srg_mp2_moes": srg_mp2_moes,
        "srg_mp2_flow": flow,
        "srg_mp2_niters": niters,
        "spec_fn_cumulant": spec_fn_cumulant,
        "omegas": omegas,
    }

def load_hf(filename: str, p: int):
    """
    Load only what is needed to plot pole p from the HDF5 produced by your C++ code:
      - spec_fn_g0_p{p}
      - spec_fn_cumulant_p{p}
      - omegas_p{p}
    """
    with h5py.File(filename, "r") as f:

        g0_key = f"spec_fn_g0_p{p}"
        cum_key = f"spec_fn_cumulant_analytic_p{p}"
        om_key = f"omegas_p{p}"

        if g0_key not in f or cum_key not in f or om_key not in f:
            missing = [k for k in (g0_key, cum_key, om_key) if k not in f]
            raise KeyError(f"Missing datasets for p={p}: {missing}")

        spec_fn_g0 = np.asarray(f[g0_key][0, :])
        spec_fn_cumulant = np.asarray(f[cum_key][0, :])
        omegas = np.asarray(f[om_key][0, :])

    return {
        "spec_fn_g0": spec_fn_g0,
        "spec_fn_cumulant": spec_fn_cumulant,
        "omegas": omegas,
    }

def load_old(filename: str, p: int):
    """
    Load only what is needed to plot pole p from the HDF5 produced by your C++ code:
      - hf_moes
    """
    with h5py.File(filename, "r") as f:
        hf_moes = np.asarray(f["hf_moes"][0, :])

    return {
        "hf_moes": hf_moes
    }

HARTREE_TO_EV = 27.2114
r = 4
p_idx = 0
nocc = 57
k_label = r"$k=(0,0,0)$" if p_idx == 0 else r"$k=(2\pi/L,0,0)$"
hfgw_factor = 1.5

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
 #fig2
# Read the CSV file using numpy
data = np.genfromtxt('fig2.csv', delimiter=',', skip_header=2, filling_values=np.nan)

cpp_path_fl0 = "/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/cu_fl0_rs4_npw485.h5"
hdf5_data_fl0 = load_hf(cpp_path_fl0, p_idx)
cpp_path_dt001 = "/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/cu_dt0.01_rs4_npw485.h5"
hdf5_data_fl001 = load_hf(cpp_path_dt001, p_idx)
cu_path_dt0001 = "/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/cu_dt0.001_rs4_npw485.h5"
hdf5_data_fl0001 = load_hf(cu_path_dt0001, p_idx)
cpp_path_fl01 = "/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/cu_fl0.1_rs4_npw485.h5"
hdf5_data_fl01 = load_old(cpp_path_fl01, p_idx)
# cpp_data_fl10l = "/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/cu_fl10_rs4_npw485.h5"
# inspect_hdf5(cpp_data_fl10l)
# hdf5_data_fl10l = load_hdf5_for_pole_v2(cpp_data_fl10l, p_idx)
cpp_data_fl20l = "/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/cu_fl20_rs4_npw485.h5"
# inspect_hdf5(cpp_data_fl20l)
hdf5_data_fl20l = load_hdf5_for_pole_v2(cpp_data_fl20l, p_idx)
# cpp_data_fl60l = "/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/cu_fl60_rs4_npw485.h5"
# inspect_hdf5(cpp_data_fl60l)
# hdf5_data_fl60l = load_hdf5_for_pole_v2(cpp_data_fl60l, p_idx)

# Extract data for each method (columns are: ccsd_X, ccsd_Y, hf_gw_X, hf_gw_Y, lda_gw_X, lda_gw_Y, lda_gw_c_X, lda_gw_c_Y)
ccsd_x = data[:, 0]
ccsd_y = data[:, 1]
hf_gw_x = data[:, 2]
hf_gw_y = data[:, 3]*hfgw_factor
lda_gw_x = data[:, 4]
lda_gw_y = data[:, 5]
lda_gw_c_x = data[:, 6]
lda_gw_c_y = data[:, 7]

hf_pole = hdf5_data_fl01["hf_moes"][p_idx] * HARTREE_TO_EV
# print hf band gap
homo = hdf5_data_fl01["hf_moes"][nocc-1] * HARTREE_TO_EV
lumo = hdf5_data_fl01["hf_moes"][nocc] * HARTREE_TO_EV
print(f"HF band gap: {lumo - homo:.4f} eV")
# flow 0, rs=4, npw=485
mp2c_x = hdf5_data_fl0["omegas"] * HARTREE_TO_EV
mp2c_y = hdf5_data_fl0["spec_fn_cumulant"] / HARTREE_TO_EV
hf_y = hdf5_data_fl0["spec_fn_g0"] / HARTREE_TO_EV
# dt 0.01
mp2c_x_dt01 = hdf5_data_fl001["omegas"] * HARTREE_TO_EV
mp2c_y_dt01 = hdf5_data_fl001["spec_fn_cumulant"] / HARTREE_TO_EV
# dt 0.001
mp2c_x_dt0001 = hdf5_data_fl0001["omegas"] * HARTREE_TO_EV
mp2c_y_dt0001 = hdf5_data_fl0001["spec_fn_cumulant"] / HARTREE_TO_EV
# long time, no reg, flow 10
# mp2c_x_fl10 = hdf5_data_fl10l["omegas"] * HARTREE_TO_EV
# mp2c_y_fl10 = hdf5_data_fl10l["spec_fn_cumulant"] / HARTREE_TO_EV
# srgmp2_pole_fl10 = hdf5_data_fl10l["srg_mp2_moes"][p_idx] * HARTREE_TO_EV
# print("SRG-MP2 niters (s=10):", hdf5_data_fl10l["srg_mp2_niters"])
# long time, no reg, flow 20
mp2c_x_fl20 = hdf5_data_fl20l["omegas"] * HARTREE_TO_EV
mp2c_y_fl20 = hdf5_data_fl20l["spec_fn_cumulant"] / HARTREE_TO_EV
srgmp2_pole_fl20 = hdf5_data_fl20l["srg_mp2_moes"][p_idx] * HARTREE_TO_EV
print("SRG-MP2 niters (s=20):", hdf5_data_fl20l["srg_mp2_niters"])
# # long time, no reg, flow 60
# mp2c_x_fl60 = hdf5_data_fl60l["omegas"] * HARTREE_TO_EV
# mp2c_y_fl60 = hdf5_data_fl60l["spec_fn_cumulant"] / HARTREE_TO_EV
# srgmp2_pole_fl60 = hdf5_data_fl60l["srg_mp2_moes"][p_idx] * HARTREE_TO_EV
# print("SRG-MP2 niters (s=60):", hdf5_data_fl60l["srg_mp2_niters"])
# # flow 0.1, rs=4, npw=485
# mp2c_x_fl01 = hdf5_data_fl01["omegas"] * HARTREE_TO_EV
# mp2c_y_fl01 = hdf5_data_fl01["spec_fn_cumulant"] / HARTREE_TO_EV
# srgmp2_pole_fl01 = hdf5_data_fl01["srg_mp2_moes"][p_idx] * HARTREE_TO_EV
# print("SRG-MP2 niters (s=0.1):", hdf5_data_fl01["srg_mp2_niters"])
# # flow 1, rs=4, npw=485
# mp2c_x_fl1 = hdf5_data_fl1["omegas"] * HARTREE_TO_EV
# mp2c_y_fl1 = hdf5_data_fl1["spec_fn_cumulant"] / HARTREE_TO_EV
# srgmp2_pole_fl1 = hdf5_data_fl1["srg_mp2_moes"][p_idx] * HARTREE_TO_EV
# print("SRG-MP2 niters (s=1):", hdf5_data_fl1["srg_mp2_niters"])
# # flow 10, rs=4, npw=485
# mp2c_x_fl10 = hdf5_data_fl10["omegas"] * HARTREE_TO_EV
# mp2c_y_fl10 = hdf5_data_fl10["spec_fn_cumulant"] / HARTREE_TO_EV
# srgmp2_pole_fl10 = hdf5_data_fl10["srg_mp2_moes"][p_idx] * HARTREE_TO_EV
# print("SRG-MP2 niters (s=10):", hdf5_data_fl10["srg_mp2_niters"])

# Remove NaN values for each dataset
ccsd_mask = ~(np.isnan(ccsd_x) | np.isnan(ccsd_y))
hf_gw_mask = ~(np.isnan(hf_gw_x) | np.isnan(hf_gw_y))
lda_gw_mask = ~(np.isnan(lda_gw_x) | np.isnan(lda_gw_y))
lda_gw_c_mask = ~(np.isnan(lda_gw_c_x) | np.isnan(lda_gw_c_y))

# Create the plot
plt.figure(figsize=(10, 6))
p_idx = 0
system_size = '114e/485o'
eta = 0.8

# plt.plot(ccsd_x[ccsd_mask], ccsd_y[ccsd_mask], label='CCSD')
# plt.plot(hf_gw_x[hf_gw_mask], hf_gw_y[hf_gw_mask], label='HF+GW')
# # plt.plot(lda_gw_x[lda_gw_mask], lda_gw_y[lda_gw_mask], label='LDA+GW')
# plt.plot(lda_gw_c_x[lda_gw_c_mask], lda_gw_c_y[lda_gw_c_mask], label='LDA+GW+C')
plt.plot(mp2c_x, hf_y, label='HF', linestyle=':')
plt.axvline(hf_pole, color='black', linestyle='-', label='HF pole')

# ========== CONVERGENCE STUDY DATA ==========
# Extract data from convergence study files
convergence_files = [
    # "cu_dt0.01_rs4_npw485_thresh0.0001_wrange0.918733.h5",
    # "cu_dt0.01_rs4_npw485_thresh0.0001_wrange9.18733.h5",
    # "cu_dt0.01_rs4_npw485_thresh0.001_wrange0.918733.h5",
    # "cu_dt0.01_rs4_npw485_thresh0.001_wrange18.3747.h5",
    # "cu_dt0.01_rs4_npw485_thresh0.001_wrange9.18733.h5",
    # "cu_dt0.1_rs4_npw485_thresh0.0001_wrange0.918733.h5",
    # "cu_dt0.1_rs4_npw485_thresh0.0001_wrange18.3747.h5",
    # "cu_dt0.1_rs4_npw485_thresh0.0001_wrange9.18733.h5",
    # "cu_dt0.1_rs4_npw485_thresh0.001_wrange0.918733.h5",
    # "cu_dt0.1_rs4_npw485_thresh0.001_wrange18.3747.h5",
    # "cu_dt0.1_rs4_npw485_thresh0.001_wrange9.18733.h5",1
    "cu_dt0.1_rs4_npw485_thresh0.001_wrange100.h5",
    "cu_dt0.1_rs4_npw485_thresh0.0001_wrange100.h5",
    "cu_dt0.1_rs4_npw485_thresh0.001_wrange150.h5",
    "cu_dt0.1_rs4_npw485_thresh0.0001_wrange150.h5",
    "cu_dt0.1_rs4_npw485_thresh0.001_wrange200.h5",
    "cu_dt0.1_rs4_npw485_thresh0.0001_wrange200.h5",
    "cu_dt0.1_rs4_npw485_thresh0.0001_wrange200.h5",
    "cu_dt0.05_rs4_npw485_thresh0.0001_wrange100.h5",
    "cu_dt0.05_rs4_npw485_thresh0.0001_wrange150.h5",
    "cu_dt0.05_rs4_npw485_thresh0.0001_wrange200.h5"
]

def extract_convergence_params(filename):
    """Extract dt, thresh, wrange from filename."""
    import re
    pattern = r'cu_dt([\d.]+)_rs4_npw485_thresh([\d.]+)_wrange([\d.]+)\.h5'
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    return None, None, None

# Load convergence study data
convergence_data = {}
for filename in convergence_files:
    filepath = os.path.join(script_dir, filename)
    if os.path.exists(filepath):
        dt, thresh, wrange = extract_convergence_params(filename)
        if dt is not None:
            try:
                data = load_hf(filepath, p_idx)
                convergence_data[(dt, thresh, wrange)] = {
                    'omegas': data['omegas'] * HARTREE_TO_EV,
                    'spec_fn': data['spec_fn_cumulant'] / HARTREE_TO_EV
                }
                print(f"Loaded convergence: dt={dt}, thresh={thresh}, wrange={wrange}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

# Plot a few representative convergence curves
# Example: Compare dt=0.01 vs dt=0.1 for fixed thresh=0.001, wrange=9.18733
# if (0.01, 0.001, 0.918733) in convergence_data:
#     data_dt01 = convergence_data[(0.01, 0.001, 0.918733)]
#     plt.plot(data_dt01['omegas'], data_dt01['spec_fn'], 
#              label='MP2+C (dt=0.01, thresh=0.001, wrange=0.918733)', linestyle='--', alpha=0.7)
if (0.1, 0.001, 100) in convergence_data:
    data_dt1 = convergence_data[(0.1, 0.001, 100)]
    plt.plot(data_dt1['omegas'], data_dt1['spec_fn'], 
             label='MP2+C (dt=0.1, thresh=0.001, wrange=100)', linestyle='--', alpha=0.7)
if (0.1, 0.0001, 100) in convergence_data:
    data_dt1_strict = convergence_data[(0.1, 0.0001, 100)]
    plt.plot(data_dt1_strict['omegas'], data_dt1_strict['spec_fn'], 
             label='MP2+C (dt=0.1, thresh=0.0001, wrange=100)', linestyle='--', alpha=0.7)
if (0.1, 0.001, 150) in convergence_data:
    data_dt1_wide = convergence_data[(0.1, 0.001, 150)]
    plt.plot(data_dt1_wide['omegas'], data_dt1_wide['spec_fn'], 
             label='MP2+C (dt=0.1, thresh=0.001, wrange=150)', linestyle='--', alpha=0.7)
if (0.1, 0.0001, 150) in convergence_data:
    data_dt1_strict_wide = convergence_data[(0.1, 0.0001, 150)]
    plt.plot(data_dt1_strict_wide['omegas'], data_dt1_strict_wide['spec_fn'], 
             label='MP2+C (dt=0.1, thresh=0.0001, wrange=150)', linestyle='--', alpha=0.7)
if (0.1, 0.001, 200) in convergence_data:
    data_dt1_wider = convergence_data[(0.1, 0.001, 200)]
    plt.plot(data_dt1_wider['omegas'], data_dt1_wider['spec_fn'], 
             label='MP2+C (dt=0.1, thresh=0.001, wrange=200)', linestyle='--', alpha=0.7)
if (0.1, 0.0001, 200) in convergence_data:
    data_dt1_strict_wider = convergence_data[(0.1, 0.0001, 200)]
    plt.plot(data_dt1_strict_wider['omegas'], data_dt1_strict_wider['spec_fn'], 
             label='MP2+C (dt=0.1, thresh=0.0001, wrange=200)', linestyle='--', alpha=0.7)
if (0.05, 0.0001, 100) in convergence_data:
    data_dt05_strict = convergence_data[(0.05, 0.0001, 100)]
    plt.plot(data_dt05_strict['omegas'], data_dt05_strict['spec_fn'], 
             label='MP2+C (dt=0.05, thresh=0.0001, wrange=100)', linestyle='--', alpha=0.7)
if (0.05, 0.0001, 150) in convergence_data:
    data_dt05_strict_wide = convergence_data[(0.05, 0.0001, 150)]
    plt.plot(data_dt05_strict_wide['omegas'], data_dt05_strict_wide['spec_fn'], 
             label='MP2+C (dt=0.05, thresh=0.0001, wrange=150)', linestyle='--', alpha=0.7)
if (0.05, 0.0001, 200) in convergence_data:
    data_dt05_strict_wider = convergence_data[(0.05, 0.0001, 200)]
    plt.plot(data_dt05_strict_wider['omegas'], data_dt05_strict_wider['spec_fn'], 
             label='MP2+C (dt=0.05, thresh=0.0001, wrange=200)', linestyle='--', alpha=0.7)
# ...existing code...
plt.xlabel("Frequency ω (eV)")
# make xmin = ccsd_x.min() , xmax = ccsd_x.max() for masked ccsd_x
plt.xlim(ccsd_x[ccsd_mask].min(), ccsd_x[ccsd_mask].max())
plt.ylabel("Spectral Function A(ω) (1/eV)")
plt.title(rf"Spectral function ({k_label}, rs={r}, system={system_size}, $\eta$={eta} eV)")
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig('fig2.png', dpi=300, bbox_inches='tight')

#fig1a/1b
system_size = '14e/19o'
eta = 0.2
# Read the CSV file using numpy
data = np.genfromtxt('fig1a.csv', delimiter=',', skip_header=2, filling_values=np.nan)

# Extract data for each method (columns are: ccsd_X, ccsd_Y, hf_gw_X, hf_gw_Y, lda_gw_X, lda_gw_Y, lda_gw_c_X, lda_gw_c_Y)
ccsd_x = data[:, 4]
ccsd_y = data[:, 5]
hf_gw_x = data[:, 0]
hf_gw_y = data[:, 1]*hfgw_factor
lda_gw_c_x = data[:, 2]
lda_gw_c_y = data[:, 3]

# Remove NaN values for each dataset
ccsd_mask = ~(np.isnan(ccsd_x) | np.isnan(ccsd_y))
hf_gw_mask = ~(np.isnan(hf_gw_x) | np.isnan(hf_gw_y))
lda_gw_mask = ~(np.isnan(lda_gw_x) | np.isnan(lda_gw_y))
lda_gw_c_mask = ~(np.isnan(lda_gw_c_x) | np.isnan(lda_gw_c_y))

# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(ccsd_x[ccsd_mask], ccsd_y[ccsd_mask], label='CCSD')
plt.plot(hf_gw_x[hf_gw_mask], hf_gw_y[hf_gw_mask], label='HF-GW')
plt.plot(lda_gw_c_x[lda_gw_c_mask], lda_gw_c_y[lda_gw_c_mask], label='LDA-GW+C')

plt.xlabel("Frequency ω (eV)")
plt.ylabel("Spectral Function A(ω) (1/eV)")
k_label = r"$k=(0,0,0)$" if p_idx == 0 else r"$k=(2\pi/L,0,0)$"
plt.title(rf"Spectral function ({k_label}, rs={r}, system={system_size}, $\eta$={eta} eV)")
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig('fig1a.png', dpi=300, bbox_inches='tight')


