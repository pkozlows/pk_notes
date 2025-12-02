from pyscf import gto, df, scf

mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz')

# Integrals in memory
int3c = df.incore.cholesky_eri(mol, auxbasis='ccpvdz-fit')
print("3-center integrals shape (in memory):", int3c.shape)