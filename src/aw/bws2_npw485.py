import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
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
bws2_sps = data["sps"][1, :]
srgmp2_path = "cu_fl60_rs4_npw485.h5"
'''
cu_fl60_rs4_npw485.h5 
srg_mp2_band_gap         Dataset {1, 1}
srg_mp2_flow             Dataset {1, 1}
srg_mp2_moes             Dataset {1, 485}
srg_mp2_niters           Dataset {1, 1}
'''
mp2_path = "cu_fl0_rs4_npw485.h5"
srg_dataset_names = [
    "srg_mp2_band_gap",
    "srg_mp2_flow",
    "srg_mp2_moes",
    "srg_mp2_niters",
]
with h5py.File(srgmp2_path, "r") as f:
    srg_data = {name: np.array(f[name]) for name in srg_dataset_names}
srgmp2_sps = np.squeeze(srg_data["srg_mp2_moes"])
bfgs_path = "bfgs_srg_mp2_ne114_npw485_rs4.h5"
inspect_hdf5(bfgs_path)
srg_dataset_names = [
    "srg_energies"
]
with h5py.File(bfgs_path, "r") as f:
    bfgs_data = {name: np.array(f[name]) for name in srg_dataset_names}
srg_sps = (bfgs_data["srg_energies"])
mp2_sps = srg_sps[0, :]
srg1_sps = srg_sps[1, :]
srg10_sps = srg_sps[2, :]
# # print all sps for debugging
# # print("HF sps:", hf_sps)
# # print("SRG-MP2 sps:", srgmp2_sps)
# # print("BWs2 sps:", bws2_sps)
# a_qp_hf = data["a_qps"][0, :]
# a_qp_srgmp2 = data["a_qps"][1, :]
# a_qp_bws2 = data["a_qps"][2, :]
# a_2h1p_hf = data["a_2h1ps"][0, :]
# a_2h1p_srgmp2 = data["a_2h1ps"][1, :]
# a_2h1p_bws2 = data["a_2h1ps"][2, :]
# a_2p1h_hf = data["a_2p1hs"][0, :]
# a_2p1h_srgmp2 = data["a_2p1hs"][1, :]
# a_2p1h_bws2 = data["a_2p1hs"][2, :]
# a_tot_hf = data["a_totals"][0, :]
# a_tot_srgmp2 = data["a_totals"][1, :]
# a_tot_bws2 = data["a_totals"][2, :]
# a_fft_hf = data["a_fft"][0, :]
# a_fft_srgmp2 = data["a_fft"][1, :]
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
plt.figure(figsize=(10, 6))
# put axvline at hf, srgmp2, and bws2 sps[p_idx]
# plt.axvline(hf_sps[p_idx]*HARTREE_TO_EV, color="red", linestyle="-", label="HF")
# plt.axvline(srgmp2_sps[p_idx]*HARTREE_TO_EV, color="blue", linestyle="--", label="SRG-MP2 w/ s=60")
# plt.axvline(srg1_sps[p_idx]*HARTREE_TO_EV, color="orange", linestyle="--", label="SRG-MP2 w/ s=1")
# plt.axvline(srg10_sps[p_idx]*HARTREE_TO_EV, color="magenta", linestyle="--", label="SRG-MP2 w/ s=10")
# plt.axvline(bws2_sps[p_idx]*HARTREE_TO_EV, color="green", linestyle="--", label=rf"BWs2; $\alpha_1$=1, $\alpha_2$=0")
# # plot atot for each method
# plt.plot(omegas*HARTREE_TO_EV, a_tot_hf/HARTREE_TO_EV, color="black", linestyle="--", label="G0@HF; C(2@HF) to first order")
# plt.plot(omegas*HARTREE_TO_EV, a_tot_srgmp2/HARTREE_TO_EV, color="red", linestyle="--", label="G0@SRG-MP2; C(2@SRG-MP2) to first order")
# plt.plot(omegas*HARTREE_TO_EV, a_tot_bws2/HARTREE_TO_EV, color="green", linestyle="--", label="G0@BWs2; C(2@BWs2) to first order")
# plot ref data
plt.plot(ccsd_x, ccsd_y, color="black", linestyle="-", label="CCSD (Ref.)")
plt.plot(lda_gw_c_x, lda_gw_c_y, color="black", linestyle=":", label="LDA+GW+C (Ref.)")
# plot a_fft for each method
# plt.plot(omegas*HARTREE_TO_EV, a_fft_hf/HARTREE_TO_EV, color="red", linestyle="-", label="G0@HF; C(2@HF) to infinite order")
# plt.plot(omegas*HARTREE_TO_EV, a_fft_srgmp2/HARTREE_TO_EV, color="blue", linestyle="-", label="G0@SRG-MP2; C(2@SRG-MP2) to infinite order")
# plt.plot(omegas*HARTREE_TO_EV, a_fft_bws2/HARTREE_TO_EV, color="green", linestyle="-", label="G0@BWs2; C(2@BWs2)")

# alpha1_colors = {
#     0: "cyan",
#     1: "green",
#     2: "orange",
#     3: "magenta",
#     4: "red",
#     5: "blue",
# }
# alpha2_linestyles = {
#     0: "-",
#     1: "--",
#     2: "-",
#     3: "-",
#     4: "-.",
#     5: ":",
# }
alpha2_colors = {
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
# for alpha1, alpha2 in alpha_pairs:
for alpha1, alpha2 in a1eq2_alpha_pairs:
    # Only plot pairs where alpha1 is 1-4 and alpha2 is 0-3
    # if alpha1 < 1 or alpha1 > 2 or alpha2 < 0:# or alpha2 > 5:
    # if alpha1<0 or alpha1>3 or alpha2<3:# or (alpha1==3 and alpha2==5):
    if alpha2 < -2 or alpha2 > 6:
        continue
    
    label = rf"BWs2+C; $\alpha_1$={alpha1}, $\alpha_2$={alpha2}"
    pole_label = rf'BWs2; $\alpha_1$={alpha1}, $\alpha_2$={alpha2}'
    # find the corresponding a_fft for this pair
    idx = np.where((a1eq2_alpha_pairs[:, 0] == alpha1) & (a1eq2_alpha_pairs[:, 1] == alpha2))[0]
    if len(idx) > 0:
        a_fft = a1eq2_a_fft[idx[0], :]
        plt.plot(a1eq2_omegas_fft*HARTREE_TO_EV, a_fft/HARTREE_TO_EV, color=alpha2_colors[alpha2], linestyle="-", label=label)
        plt.axvline(a1eq2_a_sp_energies[idx[0], p_idx]*HARTREE_TO_EV, color=alpha2_colors[alpha2], linestyle="-")
plt.axvline(mp2_sps[p_idx]*HARTREE_TO_EV, color="blue", linestyle="--", label="MP2")
plt.xlabel("Energy (eV)")
plt.ylabel("Spectral Function (1/eV)")
plt.title(rf"Cumulant spectral function at rs={r}, {k_label}, npw=485, $\eta=0.8$ eV")
plt.legend()
plt.xlim(-20, 0)
plt.ylim(0, None)
plt.grid()
plt.tight_layout()
plt.savefig("fig2_bws2_comparison.png", dpi=300)

