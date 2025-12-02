import numpy as np
from pyscf import gto, scf, df, lib, gw, tddft, ao2mo, df, cc
from pyscf.pbc import gto as pbcgto, scf as pbcscf, df as pbcdf, cc as pbccc, ao2mo as pbcao2mo
import pyscf
import os
import h5py
script_dir = os.path.dirname(os.path.abspath(__file__))

print(pyscf.__version__)

filename = os.path.join(os.path.dirname(__file__), "ueg_r0.5_n14.h5")

with h5py.File(filename, "r") as f:
    U = f["U"][()]                 # complex or real matrix
    finite_eriao8 = f["finite_eriao8"][()]
    finite_cholao = f["finite_cholao"][()]
    hcore = f["hcore"][()]
    ovlp = f["ovlp"][()]

print("U:", U.shape, U.dtype)
print("finite_eriao8:", finite_eriao8.shape, finite_eriao8.dtype)
print("finite_cholao:", finite_cholao.shape, finite_cholao.dtype)
print("hcore:", hcore.shape, hcore.dtype)
print("ovlp:", ovlp.shape, ovlp.dtype)

n_elec = 14
nbsf = U.shape[0]
mol = gto.M()
mol.nelectron = n_elec
n = nbsf

# mol scf
mol = gto.M()
mol.nelectron = n_elec
n = nbsf
mf = scf.RHF(mol)
transformed_hcore = np.einsum("ip,ij,jq->pq", U.conj(), hcore, U, optimize=True)
mf._eri = ao2mo.restore(4, finite_eriao8, n)
# mf.with_df._cderi = finite_cholao
mf.get_hcore = lambda *args: transformed_hcore
transformed_ovlp = np.einsum("ip,ij,jq->pq", U.conj(), ovlp, U, optimize=True)
mf.get_ovlp = lambda *args: transformed_ovlp
mf.verbose = 4
mf.kernel()
# C = mf.mo_coeff
# print(np.imag(C))
# 1. Get AO overlap and Fock (take real parts to kill ~1e-16 noise)
S = mf.get_ovlp()
assert np.allclose(S, S.conj().T, atol=1e-12)
S = mf.get_ovlp().real           # (nao, nao)
F = mf.get_fock()
assert np.allclose(F, F.real)
F = mf.get_fock().real           # (nao, nao)

# 2. Build S^{-1/2} (orthogonalization matrix)
se, sv = np.linalg.eigh(S)
# guard against tiny negative eigenvalues from numerical noise
se[se < 0] = 0.0
s_inv_sqrt = np.diag(1.0 / np.sqrt(se + 1e-16))
X = sv @ s_inv_sqrt @ sv.T       # X is real, (nao, nao)

# 3. Transform F into orthogonal basis and diagonalize
F_ortho = X.T @ F @ X            # real symmetric
eps, Uortho = np.linalg.eigh(F_ortho)

# 4. Back-transform to AO basis: real canonical MOs
C_real = X @ Uortho              # (nao, nmo), real
assert np.allclose(eps, mf.mo_energy, atol=1e-8)

print("Max imag part of C_real:", np.max(np.abs(np.imag(C_real))))
e, c = np.linalg.eigh(F)
assert np.allclose(e, mf.mo_energy, atol=1e-8)

# 5. Overwrite mf.mo_coeff and mf.mo_energy with the real canonical ones
mf.mo_coeff = np.array(C_real, dtype=np.double)
mf.mo_energy = np.array(eps, dtype=np.double)

# (Optional sanity checks)
print("Orthonormal in AO metric:",
      np.allclose(C_real.T @ S @ C_real, np.eye(C_real.shape[1]), atol=1e-10))

F_mo = C_real.T @ F @ C_real
print("Fock approximately diagonal in MO basis:",
      np.allclose(F_mo, np.diag(np.diag(F_mo)), atol=1e-8))




mycc = cc.CCSD(mf)
mycc.verbose = 4

# 1) Build DF/eris once and inspect
eris = mycc.ao2mo()
eris.fock = eris.fock.real


mycc.kernel(eris=eris)

