import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import re
from pathlib import Path

HARTREE_TO_EV = 27.2114

def extract_params_from_filename(filename):
    """
    Extract dt, thresh, and wrange from filename like:
    cu_dt0.01_rs4_npw485_thresh0.001_wrange9.18733.h5
    """
    pattern = r'cu_dt([\d.]+)_rs4_npw485_thresh([\d.]+)_wrange([\d.]+)\.h5'
    match = re.search(pattern, filename)
    if match:
        dt = float(match.group(1))
        thresh = float(match.group(2))
        wrange = float(match.group(3))
        return dt, thresh, wrange
    return None, None, None

def load_spectral_data(filepath, p_idx=0):
    """Load spectral function data from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        omegas_key = f'omegas_p{p_idx}'
        spec_fn_key = f'spec_fn_cumulant_analytic_p{p_idx}'
        
        if omegas_key not in f or spec_fn_key not in f:
            raise KeyError(f"Missing datasets in {filepath}")
        
        omegas = np.asarray(f[omegas_key][0, :])
        spec_fn = np.asarray(f[spec_fn_key][0, :])
    
    return omegas * HARTREE_TO_EV, spec_fn / HARTREE_TO_EV

def main():
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # List of all files to process
    filenames = [
        "cu_dt0.01_rs4_npw485_thresh0.0001_wrange0.918733.h5",
        "cu_dt0.01_rs4_npw485_thresh0.0001_wrange9.18733.h5",
        "cu_dt0.01_rs4_npw485_thresh0.001_wrange0.918733.h5",
        "cu_dt0.01_rs4_npw485_thresh0.001_wrange18.3747.h5",
        "cu_dt0.01_rs4_npw485_thresh0.001_wrange9.18733.h5",
        "cu_dt0.1_rs4_npw485_thresh0.0001_wrange0.918733.h5",
        "cu_dt0.1_rs4_npw485_thresh0.0001_wrange18.3747.h5",
        "cu_dt0.1_rs4_npw485_thresh0.0001_wrange9.18733.h5",
        "cu_dt0.1_rs4_npw485_thresh0.001_wrange0.918733.h5",
        "cu_dt0.1_rs4_npw485_thresh0.001_wrange18.3747.h5",
        "cu_dt0.1_rs4_npw485_thresh0.001_wrange9.18733.h5",
    ]
    
    p_idx = 0
    
    # Extract and organize data by parameters
    data_dict = {}
    for filename in filenames:
        filepath = os.path.join(script_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found, skipping...")
            continue
        
        dt, thresh, wrange = extract_params_from_filename(filename)
        if dt is None:
            print(f"Warning: Could not parse parameters from {filename}")
            continue
        
        try:
            omegas, spec_fn = load_spectral_data(filepath, p_idx)
            data_dict[(dt, thresh, wrange)] = {
                'omegas': omegas,
                'spec_fn': spec_fn,
                'filename': filename
            }
            print(f"Loaded: dt={dt}, thresh={thresh}, wrange={wrange} ({len(omegas)} points)")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    # Create figure with multiple subplots for different parameter sweeps
    
    # Plot 1: Effect of dt (fixing thresh and wrange)
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('Convergence Study: Effect of Time Step (dt)', fontsize=14)
    
    param_sets = [
        (0.001, 0.918733),
        (0.001, 9.18733),
        (0.001, 18.3747),
        (0.0001, 0.918733),
        (0.0001, 9.18733),
        (0.0001, 18.3747),
    ]
    
    for idx, (thresh, wrange) in enumerate(param_sets):
        ax = axes1.flatten()[idx]
        for dt in [0.01, 0.1]:
            key = (dt, thresh, wrange)
            if key in data_dict:
                data = data_dict[key]
                ax.plot(data['omegas'], data['spec_fn'], 
                       label=f'dt={dt}', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Energy (eV)', fontsize=10)
        ax.set_ylabel('A(ω) (1/eV)', fontsize=10)
        ax.set_title(f'thresh={thresh}, wrange={wrange}', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_dt.png', dpi=300, bbox_inches='tight')
    
    # Plot 2: Effect of threshold (fixing dt and wrange)
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Convergence Study: Effect of Threshold', fontsize=14)
    
    param_sets = [
        (0.01, 0.918733),
        (0.01, 9.18733),
        (0.01, 18.3747),
        (0.1, 0.918733),
        (0.1, 9.18733),
        (0.1, 18.3747),
    ]
    
    for idx, (dt, wrange) in enumerate(param_sets):
        ax = axes2.flatten()[idx]
        for thresh in [0.001, 0.0001]:
            key = (dt, thresh, wrange)
            if key in data_dict:
                data = data_dict[key]
                ax.plot(data['omegas'], data['spec_fn'], 
                       label=f'thresh={thresh}', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Energy (eV)', fontsize=10)
        ax.set_ylabel('A(ω) (1/eV)', fontsize=10)
        ax.set_title(f'dt={dt}, wrange={wrange}', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_thresh.png', dpi=300, bbox_inches='tight')
    
    # Plot 3: Effect of wrange (fixing dt and thresh)
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Convergence Study: Effect of Frequency Range', fontsize=14)
    
    param_sets = [
        (0.01, 0.001),
        (0.01, 0.0001),
        (0.1, 0.001),
        (0.1, 0.0001),
    ]
    
    for idx, (dt, thresh) in enumerate(param_sets):
        ax = axes3.flatten()[idx]
        for wrange in [0.918733, 9.18733, 18.3747]:
            key = (dt, thresh, wrange)
            if key in data_dict:
                data = data_dict[key]
                ax.plot(data['omegas'], data['spec_fn'], 
                       label=f'wrange={wrange:.2f}', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Energy (eV)', fontsize=10)
        ax.set_ylabel('A(ω) (1/eV)', fontsize=10)
        ax.set_title(f'dt={dt}, thresh={thresh}', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_wrange.png', dpi=300, bbox_inches='tight')
    
    # Plot 4: All curves on one plot (may be crowded but useful for overview)
    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(data_dict)))
    for idx, ((dt, thresh, wrange), data) in enumerate(sorted(data_dict.items())):
        label = f'dt={dt}, t={thresh}, w={wrange:.2f}'
        ax4.plot(data['omegas'], data['spec_fn'], 
                label=label, linewidth=1.5, alpha=0.7, color=colors[idx])
    
    ax4.set_xlabel('Energy (eV)', fontsize=12)
    ax4.set_ylabel('Spectral Function A(ω) (1/eV)', fontsize=12)
    ax4.set_title('All Convergence Runs', fontsize=14)
    ax4.legend(fontsize=8, loc='best', ncol=2)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence_all.png', dpi=300, bbox_inches='tight')
    
    print("\nPlots saved:")
    print("  - convergence_dt.png")
    print("  - convergence_thresh.png")
    print("  - convergence_wrange.png")
    print("  - convergence_all.png")
    
    plt.show()

if __name__ == "__main__":
    main()
