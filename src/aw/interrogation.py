import h5py
import numpy as np
with h5py.File("/Users/patrykkozlowski/harvard/qcpbc/libpbc/gw_tests/pk_notes/src/aw/h_40.h5","r") as f:
    print(list(f.keys()))
    data = f["eip"][:]
    print(data)