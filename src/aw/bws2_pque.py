import numpy as np
import h5py
import matplotlib.pyplot as plt
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

filename = "ce_bws2_comparison_rs4_npw485.h5"
# Top-level keys: ['a_2h1ps', 'a_2p1hs', 'a_fft', 'a_qps', 'a_totals', 'omegas_fft', 'omegas_freqdir', 'sps']
datasets = [
    "sps",
    "a_qps",
    "a_2h1ps",
    "a_2p1hs",
    "a_totals",
    "a_fft",
    "omegas_fft",
]
with h5py.File(filename, "r") as f:
    data = {name: np.array(f[name]) for name in datasets}
hf_sps = data["sps"][0, :]
# srgmp2_sps = data["sps"][1, :]
bfgs_path = "bfgs_srg_mp2_ne114_npw485_rs4.h5"
inspect_hdf5(bfgs_path)
srg_dataset_names = [
    "srg_energies"
]
with h5py.File(bfgs_path, "r") as f:
    bfgs_data = {name: np.array(f[name]) for name in srg_dataset_names}
srg_sps = (bfgs_data["srg_energies"])
mp2_sps = srg_sps[0, :]
a_fft_bws2 = data["a_fft"][0, :]
omegas = np.squeeze(data["omegas_fft"])
big_bws2_path = "ce_bws2_ascan_rs4_npw485.h5"
inspect_hdf5(big_bws2_path)
with h5py.File(big_bws2_path, "r") as f:
    big_omegas_fft = np.squeeze(f["omegas_fft"])
    big_a_fft = np.array(f["cumulant_spec_fn"])
    big_a_sp_energies = np.array(f["bws2_sp_energies"])
    alpha_pairs = np.array(f["alpha_pairs"])
a1eq2_path = "ce_bws2_a1e2scan_rs4_npw485.h5"
inspect_hdf5(a1eq2_path)
with h5py.File(a1eq2_path, "r") as f:
    a1eq2_alpha_pairs = np.array(f["alpha_pairs"])
    a1eq2_a_fft = np.array(f["cumulant_spec_fn"])
    a1eq2_a_sp_energies = np.array(f["bws2_sp_energies"])
    a1eq2_omegas_fft = np.squeeze(f["omegas_fft"])
ref_data = np.genfromtxt('fig2.csv', delimiter=',', skip_header=2, filling_values=np.nan)
ccsd_x = ref_data[:, 0]
ccsd_y = ref_data[:, 1]
lda_gw_c_x = ref_data[:, 6]
lda_gw_c_y = ref_data[:, 7]
alpha2_colors = {
    -3: "teal",
    -2: "cyan",
    -1: "green",
    0: "orange",
    1: "magenta",
    2: "red",
    3: "blue",
    4: "purple",
    5: "brown",
    6: "pink",
}
x_label = rf'$\omega$ (eV)'
y_label = r'$A(\omega)$ (1/eV)'
aspect_ratio = (14, 5)
plt.figure(figsize=aspect_ratio)
plt.plot(ccsd_x, ccsd_y, color="black", linestyle="-", label="EOM-CCSD")
plt.plot(lda_gw_c_x, lda_gw_c_y, color="black", linestyle=":", label="GW+C")
hfpc_path = "/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/cu_fl0_rs4_npw485.h5"
inspect_hdf5(hfpc_path)
'''
] $ h5ls cu_fl0_rs4_npw485.h5 
cumulant_analytic_p0     Dataset {1, 1837}
omegas_p0                Dataset {1, 3673}
spec_fn_cumulant_analytic_p0 Dataset {1, 3673}
spec_fn_g0_p0            Dataset {1, 3673}
'''
with h5py.File(hfpc_path, "r") as f:
    omegas_p0 = np.array(np.squeeze(f["omegas_p0"][:]))
    spec_fn_g0_p0 = np.array(np.squeeze(f["spec_fn_g0_p0"][:]))
    spec_fn_cumulant_analytic_p0 = np.array(np.squeeze(f["spec_fn_cumulant_analytic_p0"][:]))
# plt.plot(omegas_p0*HARTREE_TO_EV, spec_fn_g0_p0/HARTREE_TO_EV, color="red", linestyle="-", label="HF+PT2")
# plt.plot(omegas_p0*HARTREE_TO_EV, spec_fn_cumulant_analytic_p0/HARTREE_TO_EV, color="blue", linestyle="-", label="HF+C(PT2)")
# plt.axvline(mp2_sps[p_idx]*HARTREE_TO_EV, color="blue", linestyle="--", label="MP2")
plt.xlabel(x_label)
plt.ylabel(y_label)
# plt.title(rf"Spectral functions for {k_label} at rs={r}")
plt.legend(loc="upper left")
plt.xlim(-20, -2.5)
plt.ylim(0, 0.25)
# plt.grid()
plt.tight_layout()
plt.savefig("bws2_pqe_1.png", dpi=300)
plt.figure(figsize=aspect_ratio)
plt.plot(ccsd_x, ccsd_y, color="black", linestyle="-", label="EOM-CCSD")
plt.plot(lda_gw_c_x, lda_gw_c_y, color="black", linestyle=":", label="GW+C")
hfpc_path = "/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/cu_fl0_rs4_npw485.h5"
inspect_hdf5(hfpc_path)
'''
] $ h5ls cu_fl0_rs4_npw485.h5 
cumulant_analytic_p0     Dataset {1, 1837}
omegas_p0                Dataset {1, 3673}
spec_fn_cumulant_analytic_p0 Dataset {1, 3673}
spec_fn_g0_p0            Dataset {1, 3673}
'''
with h5py.File(hfpc_path, "r") as f:
    omegas_p0 = np.array(np.squeeze(f["omegas_p0"][:]))
    spec_fn_g0_p0 = np.array(np.squeeze(f["spec_fn_g0_p0"][:]))
    spec_fn_cumulant_analytic_p0 = np.array(np.squeeze(f["spec_fn_cumulant_analytic_p0"][:]))
# plt.plot(omegas_p0*HARTREE_TO_EV, spec_fn_g0_p0/HARTREE_TO_EV, color="red", linestyle="-", label="HF+PT2")
plt.plot(omegas_p0*HARTREE_TO_EV, spec_fn_cumulant_analytic_p0/HARTREE_TO_EV, color="blue", linestyle="-", label="HF+C(2)")
# plt.axvline(mp2_sps[p_idx]*HARTREE_TO_EV, color="blue", linestyle="--", label="MP2")
plt.xlabel(x_label)
plt.ylabel(y_label)
# plt.title(rf"Spectral functions for {k_label} at rs={r}")
plt.legend(loc="upper left")
plt.xlim(-20, -2.5)
plt.ylim(0, 0.25)
# plt.grid()
plt.tight_layout()
plt.savefig("bws2_pqe_2.png", dpi=300)

# alpha1=2, alpha2=0; obtuse legend
plt.figure(figsize=aspect_ratio)
plt.plot(ccsd_x, ccsd_y, color="black", linestyle="-", label="EOM-CCSD")
plt.plot(lda_gw_c_x, lda_gw_c_y, color="black", linestyle=":", label="GW+C")
for alpha1, alpha2 in a1eq2_alpha_pairs:
    if alpha1 != 2 or alpha2 != 0:
        continue
    # method_label = rf'BWs2; $\alpha_2$={alpha2}'
    idx = np.where((a1eq2_alpha_pairs[:, 0] == alpha1) & (a1eq2_alpha_pairs[:, 1] == alpha2))[0]
    if len(idx) > 0:
        a_fft = a1eq2_a_fft[idx[0], :]
        norm_a = np.trapz(a_fft, a1eq2_omegas_fft)
        plt.plot(a1eq2_omegas_fft*HARTREE_TO_EV, a_fft/HARTREE_TO_EV, color=alpha2_colors[alpha2], linestyle="-", label='BWs2+C')#+f" (norm={norm_a:.2f})")
        # plt.axvline(a1eq2_a_sp_energies[idx[0], p_idx]*HARTREE_TO_EV, color=alpha2_colors[alpha2], linestyle="-")

# plt.axvline(mp2_sps[p_idx]*HARTREE_TO_EV, color="blue", linestyle="--", label="MP2")
plt.xlabel(x_label)
plt.ylabel(y_label)
# plt.title(rf"BWs2+C(BWs2) for fixed $\alpha_1=2$ with rs={r}, {k_label}, npw=485, $\eta=0.8$ eV")
plt.legend(loc="upper left")
plt.xlim(-20, -2.5)
plt.ylim(0, 0.30)
# plt.grid()
plt.tight_layout()
plt.savefig("bws2_pqe_3.png", dpi=300)

# alpha1=1-3, scan across alpha2=-1-1; verbose legend
plt.figure(figsize=(10, 6))
plt.plot(ccsd_x, ccsd_y, color="black", linestyle="-", label="EOM-CCSD")
plt.plot(lda_gw_c_x, lda_gw_c_y, color="black", linestyle=":", label="GW+C")
for alpha1, alpha2 in a1eq2_alpha_pairs:
    if alpha1 < 1 or alpha1 > 3 or alpha2 < -1 or alpha2 > 1:
        continue
    method_label = rf'BWs2+C; $\alpha_1={alpha1}, \alpha_2={alpha2}$'
    idx = np.where((a1eq2_alpha_pairs[:, 0] == alpha1) & (a1eq2_alpha_pairs[:, 1] == alpha2))[0]
    if len(idx) > 0:
        a_fft = a1eq2_a_fft[idx[0], :]
        norm_a = np.trapz(a_fft, a1eq2_omegas_fft)
        plt.plot(a1eq2_omegas_fft*HARTREE_TO_EV, a_fft/HARTREE_TO_EV, color=alpha2_colors[alpha2], linestyle="-", label=method_label)#+f" (norm={norm_a:.2f})")
        # plt.axvline(a1eq2_a_sp_energies[idx[0], p_idx]*HARTREE_TO_EV, color=alpha2_colors[alpha2], linestyle="-")

# plt.axvline(mp2_sps[p_idx]*HARTREE_TO_EV, color="blue", linestyle="--", label="MP2")
plt.xlabel(x_label)
plt.ylabel(y_label)
# plt.title(rf"BWs2+C(BWs2) for fixed $\alpha_1=2$ with rs={r}, {k_label}, npw=485, $\eta=0.8$ eV")
plt.legend(loc="upper left")
plt.xlim(-20, -2.5)
plt.ylim(0, 0.30)
# plt.grid()
plt.tight_layout()
plt.savefig("bws2_pqe_4.png", dpi=300)
