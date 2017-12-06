#
#    generate a table of luminosity distances (without c/H0 factor) for 
#    zCMB of each supernova in the JLA sample using colossus package
#
#   
from colossus.cosmology import cosmology
import numpy as np

zCMB, zhel, mB, x1, c, mst, emst, dset, biascor = np.loadtxt('jla_lcparams.txt', usecols=(1, 2, 4, 6, 8, 10, 11, 17, 20),  unpack=True)
Ns = np.size(zCMB)

params = {'flat': False, 'H0': 70.0, 'Om0': 0.3, 'OL0': 0.55, 
          'Ob0': 0.049, 'sigma8': 0.8, 'ns': 0.968, 
          'interpolation': False, 'storage': '', 'print_warnings': False}
cosmo = cosmology.setCosmology('runcosmo', params )

# grid size for the grid of Om0 and OmL0
Ng = 60
Ommin = 0.; Ommax = 1.2; dOm = (Ommax - Ommin)/Ng
Om = np.arange(Ommin, Ommax, dOm)
Oml = np.arange(Ommin, Ommax, dOm)

intsp = []; intsp2 = []
inttab = np.zeros((np.size(zCMB),Ng,Ng));
dtab = np.zeros((Ng,Ng))
cHi = cosmo.H0 / 2.99792e5
          
for iz, zs in enumerate(zCMB):
    for iOl, Old in enumerate(Oml):
        for iOm, Omd in enumerate(Om):
            cosmo.Om0 = Omd; cosmo.OL0 = Old
            cosmo.checkForChangedCosmology()
            dL = cosmo.luminosityDistance(zs) * cHi / cosmo.h
            dtab[iOm,iOl] = dL 
    inttab[iz,:,:] = dtab
    print("processing redshift N = %d, z = %.3f"%(iz, zs))

np.save('dL_int_tab_740x60x60.npy',inttab)
