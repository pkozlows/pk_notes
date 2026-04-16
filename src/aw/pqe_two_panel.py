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
#fig1a refs
ref_data_1a = np.genfromtxt('fig1a.csv', delimiter=',', skip_header=2, filling_values=np.nan)
lda_gw_c_x_1a = ref_data_1a[:, 2]
lda_gw_c_y_1a = ref_data_1a[:, 3]
ccsd_x_1a = ref_data_1a[:, 4]
ccsd_y_1a = ref_data_1a[:, 5]
# exact_data= 'lanczos_gpt.csv'
exact_data = np.genfromtxt('lanczos_gpt.csv', delimiter=',', skip_header=0, filling_values=np.nan)
'''
-14.999845581694133,0.003592006389561194
-14.59319875260902,0.004585174538158598
-14.392786406172569,0.004774349135339668
-14.092503665720349,0.004916227056431917
-13.873505574983252,0.00628774591278824
-13.67958696615742,0.007564671416966918
...
'''
lanczos_x = exact_data[:, 0]
lanczos_y = exact_data[:, 1]
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
# --- plotting ---
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=aspect_ratio,
    gridspec_kw={'wspace': 0}
)

# Left panel: 14e in 19 orbs, η = 0.2 eV
ax1.plot(lanczos_x, lanczos_y, color="gray", linestyle="-", label="Lanczos")
ax1.plot(ccsd_x_1a,     ccsd_y_1a,     color="black", linestyle="-", label="EOM-CCSD")
ax1.plot(lda_gw_c_x_1a, lda_gw_c_y_1a, color="blue",  linestyle="-", label="GW+C")
ax1.set_ylabel(y_label)

# Right panel: 114e in 485 orbs, η = 0.8 eV
ax2.plot(ccsd_x,     ccsd_y,     color="black", linestyle="-", label="EOM-CCSD")
ax2.plot(lda_gw_c_x, lda_gw_c_y, color="blue",  linestyle="-", label="GW+C")
ax2.yaxis.tick_right()
# (optionally) right-side y-label:
# ax2.yaxis.set_label_position("right")
# ax2.set_ylabel(y_label)

# ---- PANEL LABELS INSIDE AXES (top-center) ----
ax1.text(0.5, 0.98, "$r_s = 4$, 14e/19o, $\eta = 0.2$ eV",
         transform=ax1.transAxes,
         ha="center", va="top")
ax2.text(0.5, 0.98, "$r_s = 4$, 114e/485o, $\eta = 0.8$ eV",
         transform=ax2.transAxes,
         ha="center", va="top")

# custom xticks
xticksa = [-5, -7, -9, -11, -13]
xticksb = [-3, -7, -11, -15, -19]
ax1.set_xticks(xticksa)
ax2.set_xticks(xticksb)

# Remove inner spines so panels visually join
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)

# x-limits (different for each panel as you set)
ax1.set_xlim(-14, -4)
ax2.set_xlim(-20, -2.5)

# Shared x-label
try:
    fig.supxlabel(x_label)
except AttributeError:
    ax1.set_xlabel(x_label)
    ax2.set_xlabel(x_label)

# # ---- GLOBAL TITLE (NO BOX, ABOVE PANELS) ----
# title_y = 0.96
# fig.suptitle(title, y=title_y)
left   = 0.05
right  = 0.95
bottom = 0.14
top    = .97

fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=0)

# --- LEGEND: top of legend aligned with top of panels ---
legend_height = 0.08   # in figure coordinates; tweak as needed

# Place legend so its top is at y = top
leg = fig.legend(
    *ax1.get_legend_handles_labels(),
    loc="upper center",
    bbox_to_anchor=(0.5, top),   # x = center, y = top of legend
    ncol=1,
    frameon=True
)

# Style the legend box
frame = leg.get_frame()
frame.set_edgecolor("black")
frame.set_linewidth(0.8)
frame.set_facecolor("white")

# Compute legend bottom and center in figure coords (for the gap in the divider)
legend_y_top    = top
legend_y_bottom = top - legend_height
legend_y_center = 0.5 * (legend_y_top + legend_y_bottom)

# --- VERTICAL DIVIDER WITH GAP WHERE LEGEND IS ---
from matplotlib.lines import Line2D

# bottom segment: from panel bottom up to bottom of legend box
line_bottom = Line2D(
    [0.5, 0.5],
    [bottom, top],
    transform=fig.transFigure,
    color="black",
    linewidth=0.8
)

# # top segment: from top of legend box up to top of panels
# line_top = Line2D(
#     [0.5, 0.5],
#     [legend_y_top, top],
#     transform=fig.transFigure,
#     color="black",
#     linewidth=0.8
# )

fig.add_artist(line_bottom)
# fig.add_artist(line_top)
plt.savefig("two_panel_0.png", dpi=300)


bws2_small_path = "/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/ce_bws2_a1eq0d2a2eq0d2_rs4_npw19.h5"
inspect_hdf5(bws2_small_path)
'''
DATASET: spec_fn (3999,) float64
Top-level keys: ['alpha_pairs', 'bws2_sp_energies', 'cumulant_spec_fn', 'omegas_fft', 'omegas_freqdir']
DATASET: alpha_pairs (9, 2) float64
DATASET: bws2_sp_energies (9, 19) float64
DATASET: cumulant_spec_fn (9, 1597) float64
DATASET: omegas_fft (1, 1597) float64
DATASET: omegas_freqdir (1, 401) float64
'''
with h5py.File(bws2_small_path, "r") as f:
    alpha_pairs_14 = np.array(f["alpha_pairs"])
    cumulant_spec_14 = np.array(f["cumulant_spec_fn"])
    omegas_fft_14 = np.squeeze(f["omegas_fft"])
# alpha1=2, alpha2=0; obtuse legend
# --- plotting ---
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=aspect_ratio,
    gridspec_kw={'wspace': 0}
)

# Left panel: 14e in 19 orbs, η = 0.2 eV
ax1.plot(lanczos_x, lanczos_y, color="gray", linestyle="-", label="Lanczos")
ax1.plot(ccsd_x_1a,     ccsd_y_1a,     color="black", linestyle="-", label="EOM-CCSD")
ax1.plot(lda_gw_c_x_1a, lda_gw_c_y_1a, color="blue",  linestyle="-", label="GW+C")
ax1.set_ylabel(y_label)

# --- BWs2+C for 14e/19o on ax1 ---
# assume variables for the 14e case:
# alpha_pairs_14   : shape (9, 2)
# cumulant_spec_14 : shape (9, 1597)  (cumulant_spec_fn)
# omegas_fft_14    : shape (1, 1597) or (1597,)
omegas_14 = np.squeeze(omegas_fft_14)

for (alpha1, alpha2) in alpha_pairs_14:
    # keep the same selection logic as 485e case; adjust if you want a different pair
    if alpha1 != 2 or alpha2 != 0:
        continue
    idx = np.where(
        (alpha_pairs_14[:, 0] == alpha1) &
        (alpha_pairs_14[:, 1] == alpha2)
    )[0]
    if len(idx) == 0:
        continue
    idx = idx[0]
    a_fft_14 = cumulant_spec_14[idx, :]
    norm_14 = np.trapz(a_fft_14, omegas_14)  # optional normalization if you want it

    ax1.plot(
        omegas_14 * HARTREE_TO_EV,
        a_fft_14 / HARTREE_TO_EV,
        color="green",
        linestyle="-",
        label=rf'BWs2+C'
    )

# Right panel: 114e in 485 orbs, η = 0.8 eV
ax2.plot(ccsd_x,     ccsd_y,     color="black", linestyle="-", label="EOM-CCSD")
ax2.plot(lda_gw_c_x, lda_gw_c_y, color="blue",  linestyle="-", label="GW+C")
ax2.yaxis.tick_right()

# --- BWs2+C for 114e/485o on ax2 ---
# existing working variables for 485e case:
# a1eq2_alpha_pairs : shape (13, 2)
# a1eq2_a_fft       : shape (13, 1597) or same as cumulant_spec_fn
# a1eq2_omegas_fft  : shape (1, 1597)
omegas_485 = np.squeeze(a1eq2_omegas_fft)

for (alpha1, alpha2) in a1eq2_alpha_pairs:
    if alpha1 != 2 or alpha2 != 0:
        continue
    idx = np.where(
        (a1eq2_alpha_pairs[:, 0] == alpha1) &
        (a1eq2_alpha_pairs[:, 1] == alpha2)
    )[0]
    if len(idx) == 0:
        continue
    idx = idx[0]
    a_fft_485 = a1eq2_a_fft[idx, :]
    norm_485 = np.trapz(a_fft_485, omegas_485)

    ax2.plot(
        omegas_485 * HARTREE_TO_EV,
        a_fft_485 / HARTREE_TO_EV,
        color="green",
        linestyle="-",
        label=rf'BWs2+C ($\alpha_1={alpha1}$, $\alpha_2={alpha2}$)'
    )

# ---- PANEL LABELS INSIDE AXES (top-center) ----
ax1.text(0.5, 0.98, "$r_s=4$, 14e/19o, $\eta = 0.2$ eV",
         transform=ax1.transAxes,
         ha="center", va="top")
ax2.text(0.5, 0.98, "$r_s=4$, 114e/485o, $\eta = 0.8$ eV",
         transform=ax2.transAxes,
         ha="center", va="top")

# custom xticks
xticksa = [-5, -7, -9, -11, -13]
xticksb = [-3, -7, -11, -15, -19]
ax1.set_xticks(xticksa)
ax2.set_xticks(xticksb)

# Remove inner spines so panels visually join
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)

# x-limits (different for each panel)
ax1.set_xlim(-14, -4)
ax2.set_xlim(-20, -2.5)

# Shared x-label
try:
    fig.supxlabel(x_label)
except AttributeError:
    ax1.set_xlabel(x_label)
    ax2.set_xlabel(x_label)

# # ---- GLOBAL TITLE (NO BOX, ABOVE PANELS) ----
# title_y = 0.96
# fig.suptitle(title, y=title_y)

# left   = 0.05
# right  = 0.95
# bottom = 0.14
# top    = 1

fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=0)

# --- LEGEND: top of legend aligned with top of panels ---
legend_height = 0.08  # tweak if needed

leg = fig.legend(
    *ax1.get_legend_handles_labels(),
    loc="upper center",
    bbox_to_anchor=(0.5, top),
    ncol=1,
    frameon=True
)

frame = leg.get_frame()
frame.set_edgecolor("black")
frame.set_linewidth(0.8)
frame.set_facecolor("white")

legend_y_top    = top
legend_y_bottom = top - legend_height

# --- VERTICAL DIVIDER WITH GAP WHERE LEGEND IS ---
from matplotlib.lines import Line2D

line_bottom = Line2D(
    [0.5, 0.5],
    [bottom, top],
    transform=fig.transFigure,
    color="black",
    linewidth=0.8
)
# line_top = Line2D(
#     [0.5, 0.5],
#     [legend_y_top, top],
#     transform=fig.transFigure,
#     color="black",
#     linewidth=0.8
# )

fig.add_artist(line_bottom)
# fig.add_artist(line_top)

plt.savefig("two_panel_2.png", dpi=300)


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
hfpc_small_path = "/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/mp2pc_analytic_r4.0_n14.h5"
inspect_hdf5(hfpc_small_path)
'''
Top-level keys: ['omegas', 'spec_fn']
DATASET: omegas (3999,) float64
DATASET: spec_fn (3999,) float64
Backend macosx is interactive backend. Turning interactive mode on.
'''
with h5py.File(hfpc_small_path, "r") as f:
    omegas_mp2 = np.array(np.squeeze(f["omegas"][:]))
    spec_fn_mp2 = np.array(np.squeeze(f["spec_fn"][:]))
# --- plotting ---
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=aspect_ratio,
    gridspec_kw={'wspace': 0}
)

# Left panel: 14e in 19 orbs, η = 0.2 eV
ax1.plot(lanczos_x, lanczos_y, color="gray", linestyle="-", label="Lanczos")
ax1.plot(ccsd_x_1a,     ccsd_y_1a,     color="black", linestyle="-", label="EOM-CCSD")
ax1.plot(lda_gw_c_x_1a, lda_gw_c_y_1a, color="blue",  linestyle="-", label="GW+C")
ax1.set_ylabel(y_label)
# add pt2+c
ax1.plot(omegas_mp2*HARTREE_TO_EV, spec_fn_mp2/HARTREE_TO_EV, color="red", linestyle="-", label="RSPT2+C")

# Right panel: 114e in 485 orbs, η = 0.8 eV
ax2.plot(ccsd_x,     ccsd_y,     color="black", linestyle="-", label="EOM-CCSD")
ax2.plot(lda_gw_c_x, lda_gw_c_y, color="blue",  linestyle="-", label="GW+C")
ax2.yaxis.tick_right()
# add pt2+c
ax2.plot(omegas_p0*HARTREE_TO_EV, spec_fn_cumulant_analytic_p0/HARTREE_TO_EV, color="red", linestyle="-", label="RSPT2+C")

# ---- PANEL LABELS INSIDE AXES (top-center) ----
ax1.text(0.5, 0.98, "$r_s=4$, 14e/19o, $\eta = 0.2$ eV",
         transform=ax1.transAxes,
         ha="center", va="top")
ax2.text(0.5, 0.98, "$r_s=4$, 114e/485o, $\eta = 0.8$ eV",
         transform=ax2.transAxes,
         ha="center", va="top")

# custom xticks
xticksa = [-5, -7, -9, -11, -13]
xticksb = [-3, -7, -11, -15, -19]
ax1.set_xticks(xticksa)
ax2.set_xticks(xticksb)

# Remove inner spines so panels visually join
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)

# x-limits (different for each panel)
ax1.set_xlim(-14, -4)
ax2.set_xlim(-20, -2.5)

# Shared x-label
try:
    fig.supxlabel(x_label)
except AttributeError:
    ax1.set_xlabel(x_label)
    ax2.set_xlabel(x_label)

# ---- GLOBAL TITLE (NO BOX, ABOVE PANELS) ----
# title_y = 0.96
# fig.suptitle(title, y=title_y)

# left   = 0.05
# right  = 0.95
# bottom = 0.14
# top    = 0.85

fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=0)

# --- LEGEND: top of legend aligned with top of panels ---
legend_height = 0.08  # tweak if needed

leg = fig.legend(
    *ax1.get_legend_handles_labels(),
    loc="upper center",
    bbox_to_anchor=(0.5, top),
    ncol=1,
    frameon=True
)

frame = leg.get_frame()
frame.set_edgecolor("black")
frame.set_linewidth(0.8)
frame.set_facecolor("white")

legend_y_top    = top
legend_y_bottom = top - legend_height

# --- VERTICAL DIVIDER WITH GAP WHERE LEGEND IS ---
from matplotlib.lines import Line2D

line_bottom = Line2D(
    [0.5, 0.5],
    [bottom, top],
    transform=fig.transFigure,
    color="black",
    linewidth=0.8
)
# line_top = Line2D(
#     [0.5, 0.5],
#     [legend_y_top, top],
#     transform=fig.transFigure,
#     color="black",
#     linewidth=0.8
# )

fig.add_artist(line_bottom)
# fig.add_artist(line_top)

plt.savefig("two_panel_1.png", dpi=300)