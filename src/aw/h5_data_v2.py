import h5py
import numpy as np
import os
HARTREE_TO_EV = 27.2114
# change working directory to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# make the first row of the txt indicates that it is eip data
with open("eip.txt", "w") as f:
    # f.write("This is EIP data\n")
    # with h5py.File("h_40.h5", "r") as h5f:
    #     np.savetxt(f, h5f["eip"][:]*HARTREE_TO_EV)
    f.write("These are observed poles\n")
    # scan through Aw_hf to find the index where it is max and match the omegas_hf
    with h5py.File("plot_ready_spectra_rs40.h5", "r") as h5f:
        Aw_hf = h5f["Aw_hf"][:].real
        Aw_diag_r = np.diagonal(Aw_hf, axis1=1, axis2=2)
        

        omegas_hf = h5f["omegas_hf"][:]
        print(omegas_hf.shape, Aw_diag_r.shape)
        max_indices = np.argmax(Aw_diag_r, axis=0)
        observed_poles = omegas_hf[max_indices]  
        np.savetxt(f, observed_poles/HARTREE_TO_EV)