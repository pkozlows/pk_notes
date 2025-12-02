import h5py

with h5py.File('plot_ready_spectra_rs4.0.h5', 'r') as f:
    for key in f.keys():
        print(key)
        if key == 'omegas_hf':
            omegas_hf = f[key][:]
        if key == 'Aw_hf':
            Aw_hf = f[key][:]    
print(omegas_hf.shape, Aw_hf.shape)
# save corr omegas, aw to
with h5py.File('hf.4.0.h5', 'r') as f:
    for key in f.keys():
        print(key)
        if key == 'eip':
            eip = f[key][:]
        if key == 'eip_vec':
            eip_vec = f[key][:]    
print(eip.shape, eip_vec.shape)
print(eip)
