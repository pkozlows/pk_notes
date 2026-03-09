import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import re
from typing import Dict, Any, Optional, Iterable
import numpy as np
import h5py

_A_RE = re.compile(r"^a_(?P<part>qp|2h1p|2p1h|total)_ne(?P<ne>\d+)_flow(?P<flow>\d+)$")
_NITERS_RE = re.compile(r"^niters_ne(?P<ne>\d+)_flow(?P<flow>\d+)$")
_SPS_RE = re.compile(r"^ce_sps_ne(?P<ne>\d+)_flow(?P<flow>\d+)$")

def merge_aw_with_sps(aw_out: Dict[str, Any], sps_out: Dict[int, Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    for ne, flow_dict in aw_out.get("data", {}).items():
        for flow, entry in flow_dict.items():
            sps_entry = sps_out.get(ne, {}).get(flow, None)
            entry["sps"] = None if sps_entry is None else sps_entry.get("sps", None)

            # optional sanity check: niters consistent if both exist
            if sps_entry is not None and entry.get("niters") is not None and sps_entry.get("niters") is not None:
                if int(entry["niters"]) != int(sps_entry["niters"]):
                    print(f"[warn] niters mismatch ne={ne} flow={flow}: aw={entry['niters']} sps={sps_entry['niters']}")
    return aw_out

def load_hdf5_sps_ne_flow(
    filename: str,
    npw: int,
    nes: Optional[Iterable[int]] = None,
    flows: Optional[Iterable[int]] = None,
    squeeze: bool = True
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """
    Returns:
      sps_data[ne][flow] = {"sps": (npw,), "niters": int or None}
    """
    nes_set = set(map(int, nes)) if nes is not None else None
    flows_set = set(map(int, flows)) if flows is not None else None

    out: Dict[int, Dict[int, Dict[str, Any]]] = {}

    with h5py.File(filename, "r") as f:
        for name, obj in f.items():
            if not isinstance(obj, h5py.Dataset):
                continue

            mS = _SPS_RE.match(name)
            if mS:
                ne = int(mS.group("ne"))
                flow = int(mS.group("flow"))
                if nes_set is not None and ne not in nes_set:
                    continue
                if flows_set is not None and flow not in flows_set:
                    continue

                sps_full = _read_1xN(obj, npw, squeeze=squeeze)

                if p_idx is not None:
                    if p_idx < 0 or p_idx >= len(sps_full):
                        raise IndexError(f"p_idx={p_idx} out of range for npw={len(sps_full)}")
                    sps_val = float(sps_full[p_idx])
                else:
                    sps_val = sps_full  # keep entire array

                out.setdefault(ne, {}).setdefault(flow, {"sps": None, "niters": None})
                out[ne][flow]["sps"] = sps_val
                continue

            mN = _NITERS_RE.match(name)
            if mN:
                ne = int(mN.group("ne"))
                flow = int(mN.group("flow"))
                if nes_set is not None and ne not in nes_set:
                    continue
                if flows_set is not None and flow not in flows_set:
                    continue

                nit = _read_scalar_1x1(obj)  # reuse your helper
                out.setdefault(ne, {}).setdefault(flow, {"sps": None, "niters": None})
                out[ne][flow]["niters"] = nit
                continue

    return out

def _read_1xN(ds: h5py.Dataset, N: int, squeeze: bool = True) -> np.ndarray:
    arr = np.asarray(ds)
    if arr.ndim == 2 and arr.shape == (1, N):
        return arr[0, :] if squeeze else arr
    if arr.ndim == 1 and arr.shape == (N,):
        return arr
    raise ValueError(f"Unexpected shape {arr.shape}; expected (1,{N}) or ({N},)")

def _read_scalar_1x1(ds: h5py.Dataset) -> int:
    arr = np.asarray(ds)
    if arr.shape == (1, 1):
        return int(arr[0, 0])
    if arr.shape == ():
        return int(arr)
    if arr.shape == (1,):
        return int(arr[0])
    raise ValueError(f"Unexpected scalar shape {arr.shape}; expected (1,1), (), or (1,)")

def load_hdf5_amp_ne_flow(
    filename: str,
    nw: int = 401,
    nes: Optional[Iterable[int]] = None,
    flows: Optional[Iterable[int]] = None,
    parts: Optional[Iterable[str]] = None,
    squeeze: bool = True,
) -> Dict[str, Any]:
    """
    Loads:
      - omegas (1,nw)
      - a_{part}_ne{ne}_flow{flow} for part in {qp,2h1p,2p1h,total}
      - niters_ne{ne}_flow{flow}

    Returns:
      {
        "omegas": (nw,),
        "data": {
           ne: {
             flow: {
               "a": {"qp":..., "2h1p":..., "2p1h":..., "total":...},
               "niters": int (if present)
             }
           }
        }
      }
    """
    nes_set = set(int(x) for x in nes) if nes is not None else None
    flows_set = set(int(x) for x in flows) if flows is not None else None
    parts_set = set(parts) if parts is not None else None

    out: Dict[str, Any] = {"omegas": None, "data": {}}

    with h5py.File(filename, "r") as f:
        # omegas
        if "omegas" not in f:
            raise KeyError("Missing required dataset: 'omegas'")
        out["omegas"] = _read_1xN(f["omegas"], nw, squeeze=True)

        # scan root datasets
        for name, obj in f.items():
            if not isinstance(obj, h5py.Dataset):
                continue

            mA = _A_RE.match(name)
            if mA:
                part = mA.group("part")
                ne = int(mA.group("ne"))
                flow = int(mA.group("flow"))

                if nes_set is not None and ne not in nes_set:
                    continue
                if flows_set is not None and flow not in flows_set:
                    continue
                if parts_set is not None and part not in parts_set:
                    continue

                arr = _read_1xN(obj, nw, squeeze=squeeze)

                ne_dict = out["data"].setdefault(ne, {})
                flow_dict = ne_dict.setdefault(flow, {"a": {}, "niters": None})
                flow_dict["a"][part] = arr
                continue

            mN = _NITERS_RE.match(name)
            if mN:
                ne = int(mN.group("ne"))
                flow = int(mN.group("flow"))

                if nes_set is not None and ne not in nes_set:
                    continue
                if flows_set is not None and flow not in flows_set:
                    continue

                nit = _read_scalar_1x1(obj)

                ne_dict = out["data"].setdefault(ne, {})
                flow_dict = ne_dict.setdefault(flow, {"a": {}, "niters": None})
                flow_dict["niters"] = nit
                continue

    return out

def niters_to_lw(n: int, lw_min=1.0, lw_max=4.0, n_ref=50):
    """
    Map iteration count -> line width in [lw_min, lw_max].
    n_ref is a typical/max iteration count to normalize against.
    """
    if n is None:
        return lw_min
    n = int(n)
    if n <= 0:
        return lw_min
    t = min(n / n_ref, 1.0)
    return lw_min + t * (lw_max - lw_min)


def pack_res_to_arrays(
    res,
    parts=("qp", "2h1p", "2p1h", "total"),
    nes=None,
    flows=None,
    sort_if_none=True,
    return_sps=True,
    npw=None,  # optional override; if None we infer from data
):
    """
    Pack nested dict res["data"][ne][flow] into dense arrays.

    Returns
    -------
    omegas : (nw,)
    nes_arr : (n_ne,)
    flows_arr : (n_flow,)
    niters : (n_ne, n_flow) float (NaN if missing)
    a : (n_part, n_ne, n_flow, nw) float (NaN if missing)
    sps : (n_ne, n_flow, npw) float (NaN if missing)  [only if return_sps]
    """
    omegas = np.asarray(res["omegas"])
    nw = omegas.shape[-1]

    data = res.get("data", {})

    # --- choose ne list ---
    if nes is None:
        nes_list = list(data.keys())
        if sort_if_none:
            nes_list = sorted(nes_list)
    else:
        nes_list = [int(x) for x in nes]

    # --- choose flow list ---
    if flows is None:
        flow_set = set()
        for ne in nes_list:
            if ne in data:
                flow_set.update(data[ne].keys())
        flows_list = list(flow_set)
        if sort_if_none:
            flows_list = sorted(flows_list)
    else:
        flows_list = [int(x) for x in flows]

    parts_list = list(parts)

    nes_arr = np.array(nes_list, dtype=int)
    flows_arr = np.array(flows_list, dtype=int)

    # niters: (n_ne, n_flow)
    niters = np.full((len(nes_arr), len(flows_arr)), np.nan)
    for i, ne in enumerate(nes_arr):
        ne_dict = data.get(int(ne), {})
        for j, flow in enumerate(flows_arr):
            flow_dict = ne_dict.get(int(flow))
            if flow_dict is not None and flow_dict.get("niters") is not None:
                niters[i, j] = float(flow_dict["niters"])

    # a: (n_part, n_ne, n_flow, nw)
    a = np.full((len(parts_list), len(nes_arr), len(flows_arr), nw), np.nan, dtype=float)
    for p, part in enumerate(parts_list):
        for i, ne in enumerate(nes_arr):
            ne_dict = data.get(int(ne), {})
            for j, flow in enumerate(flows_arr):
                flow_dict = ne_dict.get(int(flow))
                if flow_dict is None:
                    continue
                arr = flow_dict.get("a", {}).get(part)
                if arr is None:
                    continue
                a[p, i, j, :] = np.asarray(arr).reshape(-1)  # ensure (nw,)

    if not return_sps:
        return omegas, nes_arr, flows_arr, niters, a

    # Decide SPS mode: scalar poles or full arrays
    # Look for the first non-missing SPS entry
    example = None
    for ne in nes_arr:
        for flow in flows_arr:
            flow_dict = data.get(int(ne), {}).get(int(flow), None)
            if flow_dict is None:
                continue
            s = flow_dict.get("sps", None)
            if s is not None:
                example = s
                break
        if example is not None:
            break

    if example is None:
        # no SPS anywhere
        poles = np.full((len(nes_arr), len(flows_arr)), np.nan, dtype=float)
        sps = np.full((len(nes_arr), len(flows_arr), 0), np.nan, dtype=float)
        return omegas, nes_arr, flows_arr, niters, a, poles, sps

    ex = np.asarray(example)
    is_scalar_sps = (ex.size == 1)

    if is_scalar_sps:
        # pack as poles: (n_ne, n_flow)
        poles = np.full((len(nes_arr), len(flows_arr)), np.nan, dtype=float)
        for i, ne in enumerate(nes_arr):
            ne_dict = data.get(int(ne), {})
            for j, flow in enumerate(flows_arr):
                flow_dict = ne_dict.get(int(flow))
                if flow_dict is None:
                    continue
                s = flow_dict.get("sps", None)
                if s is None:
                    continue
                poles[i, j] = float(np.asarray(s).reshape(-1)[0])
        return omegas, nes_arr, flows_arr, niters, a, poles

    # otherwise: full SPS arrays (npw per entry)
    if npw is None:
        npw = int(ex.size)

    sps = np.full((len(nes_arr), len(flows_arr), npw), np.nan, dtype=float)
    for i, ne in enumerate(nes_arr):
        ne_dict = data.get(int(ne), {})
        for j, flow in enumerate(flows_arr):
            flow_dict = ne_dict.get(int(flow))
            if flow_dict is None:
                continue
            s = flow_dict.get("sps", None)
            if s is None:
                continue
            s = np.asarray(s).reshape(-1)
            if s.size != npw:
                raise ValueError(f"sps size mismatch at ne={ne} flow={flow}: got {s.size}, expected {npw}")
            sps[i, j, :] = s

    return omegas, nes_arr, flows_arr, niters, a, sps
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
# def _read_scalar_or_array(ds):
#     """Armadillo often stores scalars as length-1 vectors in HDF5."""
#     arr = ds[()]
#     arr = np.asarray(arr)
#     if arr.shape == ():          # true scalar
#         return float(arr)
#     if arr.size == 1:            # length-1 vector/matrix
#         return float(arr.ravel()[0])
#     return arr
# def load_hdf5_for_pole_v2(filename: str, p: int):
#     """
#     Load only what is needed to plot pole p from the HDF5 produced by your C++ code:
#       - hf_moes
#       - srg_mp2_moes
#       - srg_mp2_niters (scalar-ish)
#       - srg_mp2_flow (scalar-ish)
#       - spec_fn_g0_p{p}
#       - spec_fn_cumulant_p{p}
#       - omegas_p{p}
#     """
#     with h5py.File(filename, "r") as f:
#         # hf_moes = np.asarray(f["hf_moes"][0, :])
#         srg_mp2_moes = np.asarray(f["srg_mp2_moes"][0, :])

#         flow = _read_scalar_or_array(f["srg_mp2_flow"])
#         niters = _read_scalar_or_array(f["srg_mp2_niters"])

#         cum_key = f"spec_fn_cumulant_analytic_p{p}"
#         om_key = f"omegas_p{p}"

#         if cum_key not in f:
#             missing = [k for k in (cum_key) if k not in f]
#             raise KeyError(f"Missing datasets for p={p}: {missing}")

#         spec_fn_cumulant = np.asarray(f[cum_key][0, :])
#         omegas = np.asarray(f[om_key][0, :])

#     return {
#         # "hf_moes": hf_moes,
#         "srg_mp2_moes": srg_mp2_moes,
#         "srg_mp2_flow": flow,
#         "srg_mp2_niters": niters,
#         "spec_fn_cumulant": spec_fn_cumulant,
#         "omegas": omegas,
#     }



# def load_hf(filename: str, p: int):
#     """
#     Load only what is needed to plot pole p from the HDF5 produced by your C++ code:
#       - spec_fn_g0_p{p}
#       - spec_fn_cumulant_p{p}
#       - omegas_p{p}
#     """
#     with h5py.File(filename, "r") as f:

#         g0_key = f"spec_fn_g0_p{p}"
#         cum_key = f"spec_fn_cumulant_analytic_p{p}"
#         om_key = f"omegas_p{p}"

#         if g0_key not in f or cum_key not in f or om_key not in f:
#             missing = [k for k in (g0_key, cum_key, om_key) if k not in f]
#             raise KeyError(f"Missing datasets for p={p}: {missing}")

#         spec_fn_g0 = np.asarray(f[g0_key][0, :])
#         spec_fn_cumulant = np.asarray(f[cum_key][0, :])
#         omegas = np.asarray(f[om_key][0, :])

#     return {
#         "spec_fn_g0": spec_fn_g0,
#         "spec_fn_cumulant": spec_fn_cumulant,
#         "omegas": omegas,
#     }

# def load_old(filename: str, p: int):
#     """
#     Load only what is needed to plot pole p from the HDF5 produced by your C++ code:
#       - hf_moes
#     """
#     with h5py.File(filename, "r") as f:
#         hf_moes = np.asarray(f["hf_moes"][0, :])

#     return {
#         "hf_moes": hf_moes
#     }

HARTREE_TO_EV = 27.2114
r = 4
p_idx = 0
nocc = 57
k_label = r"$k=(0,0,0)$" if p_idx == 0 else r"$k=(2\pi/L,0,0)$"
# hfgw_factor = 1.5

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
data_file = "ce_conv_npw257_rs4.h5"
#  #fig2
# Read the CSV file using numpy
data = np.genfromtxt('fig2.csv', delimiter=',', skip_header=2, filling_values=np.nan)
# Extract data for each method (columns are: ccsd_X, ccsd_Y, hf_gw_X, hf_gw_Y, lda_gw_X, lda_gw_Y, lda_gw_c_X, lda_gw_c_Y)
ccsd_x = data[:, 0]
ccsd_y = data[:, 1]
# hf_gw_x = data[:, 2]
# hf_gw_y = data[:, 3]*hfgw_factor
lda_gw_x = data[:, 4]
lda_gw_y = data[:, 5]
lda_gw_c_x = data[:, 6]
lda_gw_c_y = data[:, 7]


# # Remove NaN values for each dataset
ccsd_mask = ~(np.isnan(ccsd_x) | np.isnan(ccsd_y))
# hf_gw_mask = ~(np.isnan(hf_gw_x) | np.isnan(hf_gw_y))
lda_gw_mask = ~(np.isnan(lda_gw_x) | np.isnan(lda_gw_y))
lda_gw_c_mask = ~(np.isnan(lda_gw_c_x) | np.isnan(lda_gw_c_y))
nes=(66,114)
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
from matplotlib.lines import Line2D

# Map ne to color using a colormap
cmap_ne = plt.get_cmap("viridis")
ne_min, ne_max = float(np.min(nes)), float(np.max(nes))
den_ne = (ne_max - ne_min) if ne_max > ne_min else 1.0

def color_for_ne(ne):
    """Map ne to a color from the viridis colormap."""
    t = (ne - ne_min) / den_ne  # normalize to [0, 1]
    return cmap_ne(t)

# Map flow to line style
# Define a mapping from flow values to line styles
flow_linestyles = {
    0: ":",      # solid
    50: ":",   # dotted
    100: "--",   # dashed
}

plt.figure(figsize=(10, 6))
# plot ccsd data
plt.plot(ccsd_x[ccsd_mask], ccsd_y[ccsd_mask],
         color='red',
         linestyle='-',
         linewidth=2,
         label='CCSD (114e/485o, eta=0.8 eV)')
# Plot only the total spectral function (a[0, :, :, :] since parts=["total"])
for i, ne in enumerate(nes):
    col = color_for_ne(ne)
    for j, flow in enumerate(flows):
        y = a[0, i, j, :]  # a[part_idx=0, ne_idx=i, flow_idx=j, :]
        if np.all(np.isnan(y)):
            continue
        ls = flow_linestyles.get(flow, "-")  # default to solid if flow not in map
        plt.plot(
            omegas * HARTREE_TO_EV,
            y / HARTREE_TO_EV,
            color=col,
            linestyle=ls,
            linewidth=1.5,
        )
for i, ne in enumerate(nes):
    col = color_for_ne(ne)
    for j, flow in enumerate(flows):
        x = poles[i, j] * HARTREE_TO_EV
        # if flow == 0, make the label say "s=0/HF pole"
        if np.isfinite(x) and (-20 <= x <= 0):
            label = f"HF pole" if flow == 0 else f"s={flow}"
            plt.axvline(
                x,
                color=col,
                linestyle=flow_linestyles.get(int(flow), "-"),
                linewidth=1.2,
                alpha=0.6,
                zorder=0,
                label=label if (i == 0 and j == 0) else None  # only label once
            )

# Plot reference data (Lanczos)
lanczos_path = "/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/lanczos_gpt.csv"
lanczos = np.loadtxt(lanczos_path, delimiter=",")
integral_lanczos = np.trapz(lanczos[:,1], x=lanczos[:,0])
# plt.plot(lanczos[:,0], lanczos[:,1],
#          color='black',
#          linestyle='-',
#          linewidth=2,
#          label='Lanczos (14e/19o, eta=0.2 eV)')




p_idx = 0
system_size = '257'
eta = 0.2


plt.xlabel("Frequency ω (eV)")
plt.xlim(-20, 0)
plt.ylim(0, 0.4)
plt.ylabel("Spectral Function A(ω) (1/eV)")
plt.title(rf"Spectral function varying ne and flow parameter({k_label}, rs={r}, npw={system_size}, $\eta$={eta} eV)")

# Create a single comprehensive legend
# 1. Create custom handles for ne (color)
ne_handles = [
    Line2D([0], [0], color=color_for_ne(ne), linestyle="-", linewidth=2, label=f"ne={ne}")
    for ne in nes
]

# 2. Create custom handles for flow (line style)
flow_handles = [
    Line2D([0], [0], color="black", linestyle=flow_linestyles.get(f, "-"), linewidth=2, label=f"s={f}")
    for f in flows
]

# 3. Get handles/labels from actual plotted lines (like Lanczos)
ref_handles, ref_labels = plt.gca().get_legend_handles_labels()

# Combine all handles and labels
all_handles = ne_handles + flow_handles + ref_handles
all_labels = [h.get_label() for h in ne_handles] + [h.get_label() for h in flow_handles] + ref_labels

# Create a single legend with all elements
plt.legend(all_handles, all_labels, 
          title="ne (color) | flow (style) | ref",
          loc="upper left",
          fontsize=8,
          title_fontsize=9,
          framealpha=0.9)

plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig('fig2.png', dpi=300, bbox_inches='tight')


#fig1a/1b
filename = "/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/ce_conv_npw19_rs4.h5"
res = load_hdf5_amp_ne_flow(filename)


omegas = res["omegas"]
niters_ne14_flow0 = res["data"][14][0]["niters"]
niters_ne14_flow10 = res["data"][14][10]["niters"]
niters_ne14_flow20 = res["data"][14][20]["niters"]
print("Niters (s=0):", niters_ne14_flow0)
print("Niters (s=10):", niters_ne14_flow10)
print("Niters (s=20):", niters_ne14_flow20)
a_qp_ne14_flow0 = res["data"][14][0]["a"]["qp"]
a_qp_ne14_flow10 = res["data"][14][10]["a"]["qp"]
a_qp_ne14_flow20 = res["data"][14][20]["a"]["qp"]
a_2h1p_ne14_flow0 = res["data"][14][0]["a"]["2h1p"]
a_2h1p_ne14_flow10 = res["data"][14][10]["a"]["2h1p"]
a_2h1p_ne14_flow20 = res["data"][14][20]["a"]["2h1p"]
a_2p1h_ne14_flow0 = res["data"][14][0]["a"]["2p1h"]
a_2p1h_ne14_flow10 = res["data"][14][10]["a"]["2p1h"]
a_2p1h_ne14_flow20 = res["data"][14][20]["a"]["2p1h"]
a_total_ne14_flow0 = res["data"][14][0]["a"]["total"]
a_total_ne14_flow10 = res["data"][14][10]["a"]["total"]
a_total_ne14_flow20 = res["data"][14][20]["a"]["total"]

system_size = '14e/19o'
eta = 0.2
# Read the CSV file using numpy
data = np.genfromtxt('/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/fig1a.csv', delimiter=',', skip_header=2, filling_values=np.nan)

# Extract data for each method (columns are: ccsd_X, ccsd_Y, hf_gw_X, hf_gw_Y, lda_gw_X, lda_gw_Y, lda_gw_c_X, lda_gw_c_Y)
ccsd_x = data[:, 4]
ccsd_y = data[:, 5]
# hf_gw_x = data[:, 0]
# hf_gw_y = data[:, 1]*hfgw_factor
# lda_gw_c_x = data[:, 2]
# lda_gw_c_y = data[:, 3]

# Remove NaN values for each dataset
ccsd_mask = ~(np.isnan(ccsd_x) | np.isnan(ccsd_y))
# hf_gw_mask = ~(np.isnan(hf_gw_x) | np.isnan(hf_gw_y))
# lda_gw_mask = ~(np.isnan(lda_gw_x) | np.isnan(lda_gw_y))
# lda_gw_c_mask = ~(np.isnan(lda_gw_c_x) | np.isnan(lda_gw_c_y))

# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(ccsd_x[ccsd_mask], ccsd_y[ccsd_mask], label='CCSD (114e/485o, eta=0.8 eV)')
# plt.plot(hf_gw_x[hf_gw_mask], hf_gw_y[hf_gw_mask], label='HF-GW')
# plt.plot(lda_gw_c_x[lda_gw_c_mask], lda_gw_c_y[lda_gw_c_mask], label='LDA-GW+C')
# plot all a_qp vs omegas for flow 0, 10, 20; be able to see niters depending on the linewidth
lw0 = niters_to_lw(niters_ne14_flow0)
lw10 = niters_to_lw(niters_ne14_flow10)
lw20 = niters_to_lw(niters_ne14_flow20)
# plt.plot(omegas * HARTREE_TO_EV, a_qp_ne14_flow0 / HARTREE_TO_EV, label='a_qp (s=0)', linestyle='--', linewidth=lw0)
# plt.plot(omegas * HARTREE_TO_EV, a_qp_ne14_flow10 / HARTREE_TO_EV, label='a_qp (s=10)', linestyle='--', linewidth=lw10)
# plt.plot(omegas * HARTREE_TO_EV, a_qp_ne14_flow20 / HARTREE_TO_EV, label='a_qp (s=20)', linestyle='--', linewidth=lw20)
# plt.plot(omegas * HARTREE_TO_EV, a_2h1p_ne14_flow0 / HARTREE_TO_EV, label='a_2h1p (s=0)', linestyle='-.', linewidth=lw0)
# plt.plot(omegas * HARTREE_TO_EV, a_2h1p_ne14_flow10 / HARTREE_TO_EV, label='a_2h1p (s=10)', linestyle='-.', linewidth=lw10)
# plt.plot(omegas * HARTREE_TO_EV, a_2h1p_ne14_flow20 / HARTREE_TO_EV, label='a_2h1p (s=20)', linestyle='-.', linewidth=lw20)
# plt.plot(omegas * HARTREE_TO_EV, a_2p1h_ne14_flow0 / HARTREE_TO_EV, label='a_2p1h (s=0)', linestyle=':', linewidth=lw0)
# plt.plot(omegas * HARTREE_TO_EV, a_2p1h_ne14_flow10 / HARTREE_TO_EV, label='a_2p1h (s=10)', linestyle=':', linewidth=lw10)
# plt.plot(omegas * HARTREE_TO_EV, a_2p1h_ne14_flow20 / HARTREE_TO_EV, label='a_2p1h (s=20)', linestyle=':', linewidth=lw20)
plt.plot(omegas * HARTREE_TO_EV, a_total_ne14_flow0 / HARTREE_TO_EV, label='a_total (s=0)', linestyle='--', linewidth=lw0)
plt.plot(omegas * HARTREE_TO_EV, a_total_ne14_flow10 / HARTREE_TO_EV, label='a_total (s=10)', linestyle='--', linewidth=lw10)
plt.plot(omegas * HARTREE_TO_EV, a_total_ne14_flow20 / HARTREE_TO_EV, label='a_total (s=20)', linestyle='--', linewidth=lw20)

plt.xlabel("Frequency ω (eV)")
plt.ylabel("Spectral Function A(ω) (1/eV)")
k_label = r"$k=(0,0,0)$" if p_idx == 0 else r"$k=(2\pi/L,0,0)$"
plt.title(rf"Spectral function ({k_label}, rs={r}, system={system_size}, $\eta$={eta} eV)")
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig('/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/fig1a.png', dpi=300, bbox_inches='tight')


