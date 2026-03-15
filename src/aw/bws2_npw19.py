import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
HARTREE_TO_EV = 27.2114
r = 4
p_idx = 0
nocc = 7
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

filename = "ce_bws2_comparison_rs4_npw19.h5"
inspect_hdf5(filename)
datasets = [
    "hf_sps_p0",
    "srgmp2_sps_p0",
    "bws2_sps_p0",
    "a_qp_hf_p0",
    "a_qp_srgmp2_p0",
    "a_qp_bws2_p0",
    "a_2h1p_hf_p0",
    "a_2h1p_srgmp2_p0",
    "a_2h1p_bws2_p0",
    "a_2p1h_hf_p0",
    "a_2p1h_srgmp2_p0",
    "a_2p1h_bws2_p0",
    "a_tot_hf_p0",
    "a_tot_srgmp2_p0",
    "a_tot_bws2_p0",
    "a_fft_hf_p0",
    "a_fft_srgmp2_p0",
    "a_fft_bws2_p0",
    "omegas_p0",
]
# hf sps are in the first row of the sps dataset
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
srgmp2_sps = data["sps"][1, :]
bws2_sps = data["sps"][2, :]
# print all sps for debugging
print("HF sps:", hf_sps)
print("SRG-MP2 sps:", srgmp2_sps)
print("BWs2 sps:", bws2_sps)
a_qp_hf = data["a_qps"][0, :]
a_qp_srgmp2 = data["a_qps"][1, :]
a_qp_bws2 = data["a_qps"][2, :]
a_2h1p_hf = data["a_2h1ps"][0, :]
a_2h1p_srgmp2 = data["a_2h1ps"][1, :]
a_2h1p_bws2 = data["a_2h1ps"][2, :]
a_2p1h_hf = data["a_2p1hs"][0, :]
a_2p1h_srgmp2 = data["a_2p1hs"][1, :]
a_2p1h_bws2 = data["a_2p1hs"][2, :]
a_tot_hf = data["a_totals"][0, :]
a_tot_srgmp2 = data["a_totals"][1, :]
a_tot_bws2 = data["a_totals"][2, :]
a_fft_hf = data["a_fft"][0, :]
a_fft_srgmp2 = data["a_fft"][1, :]
a_fft_bws2 = data["a_fft"][2, :]
omegas = np.squeeze(data["omegas_fft"])
ref_data = np.genfromtxt('fig1a.csv', delimiter=',', skip_header=2, filling_values=np.nan)
ccsd_x = ref_data[:, 4]
ccsd_y = ref_data[:, 5]
lda_gw_c_x = ref_data[:, 2]
lda_gw_c_y = ref_data[:, 3]
plt.figure(figsize=(10, 6))
# put axvline at hf, srgmp2, and bws2 sps[p_idx]
plt.axvline(hf_sps[p_idx]*HARTREE_TO_EV, color="red", linestyle="--", label="HF")
plt.axvline(srgmp2_sps[p_idx]*HARTREE_TO_EV, color="blue", linestyle="--", label="SRG-MP2 w/ s=1000")
plt.axvline(bws2_sps[p_idx]*HARTREE_TO_EV, color="green", linestyle="--", label="BWs2")
# # plot atot for each method
# plt.plot(omegas*HARTREE_TO_EV, a_tot_hf/HARTREE_TO_EV, color="black", linestyle="--", label="G0@HF; C(2@HF) to first order")
# plt.plot(omegas*HARTREE_TO_EV, a_tot_srgmp2/HARTREE_TO_EV, color="red", linestyle="--", label="G0@SRG-MP2; C(2@SRG-MP2) to first order")
# plt.plot(omegas*HARTREE_TO_EV, a_tot_bws2/HARTREE_TO_EV, color="green", linestyle="--", label="G0@BWs2; C(2@BWs2) to first order")
# plot ref data
plt.plot(ccsd_x, ccsd_y, color="black", linestyle="-", label="CCSD (Ref.)")
plt.plot(lda_gw_c_x, lda_gw_c_y, color="black", linestyle=":", label="LDA+GW+C (Ref.)")
# plot a_fft for each method
plt.plot(omegas*HARTREE_TO_EV, a_fft_hf/HARTREE_TO_EV, color="red", linestyle="-", label="G0@HF; C(2@HF) to infinite order")
plt.plot(omegas*HARTREE_TO_EV, a_fft_srgmp2/HARTREE_TO_EV, color="blue", linestyle="-", label="G0@SRG-MP2; C(2@SRG-MP2) to infinite order")
plt.plot(omegas*HARTREE_TO_EV, a_fft_bws2/HARTREE_TO_EV, color="green", linestyle="-", label="G0@BWs2; C(2@BWs2) to infinite order")
plt.xlabel("Energy (eV)")
plt.ylabel("Spectral Function (1/eV)")
plt.title(rf"Cumulant spectral function at rs={r}, {k_label}, npw=19, $\eta=0.2$ eV")
plt.legend()
plt.xlim(-20, 0)
plt.ylim(0, None)
plt.grid()
plt.tight_layout()
plt.savefig("fig_bws2_comparison.png", dpi=300)
