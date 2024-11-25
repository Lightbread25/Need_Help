import numpy as np
import matplotlib.pyplot as plt
import time
import healpy as hp
import pdb
from halo_mass_function import dN_dz,N


data = np.load('DATA/HalfDome/HalfDome_100.npz')
# ran_data = np.load('M500/Random_catalogue.npz')
RA = data['RA']
DEC = data['DEC']
redshift = data['redshift']
M500c = data['M500c']
# signal = data['signal']
M200m = data['M200m']
# random_position = ran_data['Random']


order = redshift.argsort()
redshift = redshift[order]
RA = RA[order]
DEC = DEC[order]
# signal = signal[order]
M500c = M500c[order]
M200m = M200m[order]

# pdb.set_trace()

# Abstract Mask
NSIDE = 256
Npix = hp.nside2npix(NSIDE)


'''
Generating Mask
'''
vec = hp.ang2vec(np.pi, 0)
ipix_disc = hp.query_disc(NSIDE, vec=vec, radius=np.radians(40))
mask = np.zeros(Npix)
mask[ipix_disc] = 1

sky_perc = len(mask[mask>0])/len(mask)
print(len(mask[mask>0])/len(mask))

NSIDE = hp.npix2nside(len(mask))
pixels = hp.ang2pix(NSIDE,RA,DEC,lonlat=True)

# Filtering for mask
mask_values = mask[pixels]
masked_indices = mask_values > 0
redshift = redshift[masked_indices]
RA = RA[masked_indices]
DEC = DEC[masked_indices]
M500c = M500c[masked_indices]
M200m = M200m[masked_indices]


# Taking a mass cut
redshift = redshift[M200m >= 1e14]




test = np.linspace(0.2,3)
# f = (1/N(0.2,0.4,1e14,sky_coverage=10313,Method="Bocquet")) * dN_dz(test,1e14,sky_coverage=10313,Method="Bocquet")
# plt.plot(test,dN_dz(test,1e14,sky_perc=sky_perc,Method='Bocquet'),'r',label="theory")
plt.plot(test,dN_dz(test,1e14,sky_perc=sky_perc,Method='Bocquet')/44,'b',label = 'Theory/40')
plt.hist(redshift,bins=50,label = "DATA")
# plt.yscale('log')
# print(len(redshift_masked),N(min(redshift_masked),max(redshift_masked),1e14,sky_perc=sky_perc,Method="Bocquet"))
plt.legend()
plt.show()