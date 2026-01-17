import pyscf

mol = pyscf.M(
atom = '''
O  0.000000    0.000000    0.000000
H  0.000000    0.960000    0.000000
H  0.740000   -0.240000    0.000000
''',
basis = 'aug-cc-pvdz',
)

mf = mol.RHF().run()

# Option 1: Utilize the dfmp2.DFMP2 implementation via the mf.DFMP2 function
mf.DFMP2().run()