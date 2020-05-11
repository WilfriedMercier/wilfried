# Mercier Wilfried - IRAP

import matplotlib.pyplot as     plt
import numpy             as     np
import astropy.io.fits   as     fits
from   matplotlib.colors import Normalize, LogNorm, DivergingNorm
from   wilfried.galaxy   import bulgeDiskOnSky, model2D, PSFconvolution2D, luminositySersics
from   math              import ceil

# Check that a 2D model (disk + bulge) works properly

offset          = 30
path            = '/home/wilfried//Thesis/galfit/Sersic_models/models/'
#galfitModelFile = path + 'sersic_n1_4_re0.76_3.92_magtot23.1_22.16_axisratio1_magoffset30.fits'
#magD            = 23.1
#magB            = 22.16
#Rd              = 0.76
#Rb              = 3.92
galfitModelFile = path + 'sersic_n1_4_re100_1.56_magtot21.07_22.61_axisratio1_magoffset30.fits'
magD            = 21.07
magB            = 22.61
Rd              = 100
Rb              = 1.56

b_a             = 1
inc             = np.arccos(b_a)*180/np.pi  # degrees
PA              = 0                       # degrees
R22             = 2.2*Rd/1.67835
R15             = 1.5/0.03


with fits.open(galfitModelFile) as hdul:
   galfitData       = hdul[0].data
   nx, ny           = np.shape(galfitData)
   xgalfit          = np.arange(0, nx, 1) - 99
   ygalfit          = np.arange(0, ny, 1) - 99
   Xgalfit, Ygalfit = np.meshgrid(xgalfit, ygalfit)
   RADgalfit        = np.sqrt(Xgalfit**2+Ygalfit**2)

arcsecToGrid = 0.03

FWHM        = 3*arcsecToGrid
fSampling   = 41
nn          = 2*ceil(R22)
if nn%2 == 0:
   nn      += 1

X, Y, model = bulgeDiskOnSky(nn, nn,
                             Rd=Rd, Rb=Rb, magD=magD, magB=magB, offsetD=offset, offsetB=offset, inclination=inc, PA=PA, combine=True,
                             fineSampling=fSampling, noPSF=True, samplingZone={'where':'all'})

RAD         = np.sqrt(X**2+Y**2)
model       = model/(fSampling**2)
intensity   = model.copy()
#intensity   = model.reshape(int(model.shape[0] / fSampling), fSampling, int(model.shape[1] / fSampling), fSampling)
#intensity   = intensity.mean(1).mean(2)

fluxGalfit   = np.nansum(galfitData[RADgalfit<=R22])
fluxModel    = np.nansum(intensity[RAD<=R22])
fluxGalfit15 = np.nansum(galfitData[RADgalfit<=R22])
fluxModel15  = np.nansum(intensity[RAD<=R15])

flux1Da     = luminositySersics(R22, [1, 4], [Rd, Rb], listMag=[magD, magB], listOffset=[offset, offset], analytical=True)['value']
flux1Dn     = luminositySersics(R22, [1, 4], [Rd, Rb], listMag=[magD, magB], listOffset=[offset, offset], analytical=False)['value']
flux1Da15   = luminositySersics(R15, [1, 4], [Rd, Rb], listMag=[magD, magB], listOffset=[offset, offset], analytical=True)['value']
flux1Dn15   = luminositySersics(R15, [1, 4], [Rd, Rb], listMag=[magD, magB], listOffset=[offset, offset], analytical=False)['value']

print('1D: analytical (R22) -> ', flux1Da, 'numerical ->', flux1Dn)
print('2D: galfit ->', fluxGalfit, 'model ->', fluxModel, 'rel err ->', (fluxModel-flux1Da)/flux1Da)

print('1D: analytical (R15) -> ', flux1Da15, 'numerical ->', flux1Dn15)
print('2D: galfit ->', fluxGalfit15, 'model ->', fluxModel15, 'rel err ->', (fluxModel15-flux1Da15)/flux1Da15)

cenX    = np.where(RAD<=R22)[1]
cenY    = np.where(RAD<=R22)[0]

f       = plt.figure(figsize=(12, 12))
ax1     = plt.subplot(121, title='Galfit 2D model')
ax2     = plt.subplot(122, title='2D model, centre, rebinned')
'''
ax5     = plt.subplot(233, title='2D model rebinned')
ax3     = plt.subplot(234, title='2D model/Galfit')
ax4     = plt.subplot(235, title='2D model - Galfit')
ax6     = plt.subplot(236, title='(2D model-Galfit)/Galfit')
'''
cmap    = 'plasma'
frac    = 0.06

norm    = LogNorm()
tmp1    = ax1.imshow(galfitData, origin='lower', cmap=cmap, norm=norm)
plt.colorbar(tmp1, ax=ax1, fraction=frac, orientation='horizontal')
circ = plt.Circle((99, 99), R22, fill=False, color='k')
ax1.add_artist(circ)

tmp2    = ax2.imshow(model, origin='lower', cmap=cmap, norm=norm)
plt.colorbar(tmp2, ax=ax2, fraction=frac, orientation='horizontal')
circ = plt.Circle((intensity.shape[0]//2, intensity.shape[1]//2), R22*fSampling, fill=False, color='k')
ax2.add_artist(circ)

'''
tmp5    = ax5.imshow(intensity, origin='lower', cmap=cmap, norm=norm)
plt.colorbar(tmp5, ax=ax5, fraction=frac, orientation='horizontal')

tmp3    = ax3.imshow(intensity/galfitData, origin='lower', cmap='bwr', norm=DivergingNorm(vcenter=1.0, vmax=np.nanmax(intensity/galfitData)))
plt.colorbar(tmp3, ax=ax3, fraction=frac, orientation='horizontal')

tmp4    = ax4.imshow(intensity-galfitData, origin='lower', cmap='bwr', norm=DivergingNorm(vcenter=0.0))
plt.colorbar(tmp4, ax=ax4, fraction=frac, orientation='horizontal')

tmp6    = ax6.imshow((intensity-galfitData)/galfitData, origin='lower', cmap='bwr', norm=DivergingNorm(vcenter=0.0))
plt.colorbar(tmp6, ax=ax6, fraction=frac, orientation='horizontal')
'''
plt.show()


# Check that galfit simply performs a sum of two profiles (for disk+bulge)
"""
path            = '/home/wilfried/bin/own_libs/'
galfitModelFile = path + 'sersic_n1_4_re10.5_6_magtot15_17_axisratio0.4_magoffset30.fits'
diskFile        = path + 'sersic_n1_re10.5_magtot15_axisratio0.4_pa0_magoffset30.fits'
bulgeFile       = path + 'sersic_n4_re6_magtot17_axisratio1_pa0_magoffset30.fits'

with fits.open(galfitModelFile) as hdul:
   galfitData   = hdul[0].data

with fits.open(diskFile) as hdul:
   diskData     = hdul[0].data

with fits.open(bulgeFile) as hdul:
   bulgeData    = hdul[0].data

f       = plt.figure(figsize=(28, 14))
ax1     = plt.subplot(231, title='2D model (all)')
ax2     = plt.subplot(232, title='Disk')
ax3     = plt.subplot(233, title='Bulge')
ax4     = plt.subplot(234, title='Sum')
ax5     = plt.subplot(235, title='2D model/Sum')
ax6     = plt.subplot(236, title='2D model - Sum')

cmap    = 'plasma'
frac    = 0.06

tmp1    = ax1.imshow(galfitData, origin='lower', cmap=cmap)
plt.colorbar(tmp1, ax=ax1, fraction=frac, orientation='horizontal')

tmp2    = ax2.imshow(diskData, origin='lower', cmap=cmap)
plt.colorbar(tmp2, ax=ax2, fraction=frac, orientation='horizontal')

tmp3    = ax3.imshow(bulgeData, origin='lower', cmap=cmap)
plt.colorbar(tmp3, ax=ax3, fraction=frac, orientation='horizontal')

tmp4    = ax4.imshow(diskData+bulgeData, origin='lower', cmap=cmap)
plt.colorbar(tmp4, ax=ax4, fraction=frac, orientation='horizontal')

tmp5   = ax5.imshow(galfitData/(diskData+bulgeData), origin='lower', cmap='bwr', norm=DivergingNorm(vcenter=1.0))
plt.colorbar(tmp5, ax=ax5, fraction=frac, orientation='horizontal')

tmp6   = ax6.imshow(galfitData-(diskData+bulgeData), origin='lower', cmap='bwr', norm=DivergingNorm(vcenter=0))
plt.colorbar(tmp6, ax=ax6, fraction=frac, orientation='horizontal')

plt.show()
"""

# Check that PSF convolution is fine
"""
# with a point
title          = 'Point-like function'
X, Y           = np.meshgrid(np.arange(-100, 100, 1), np.arange(-100, 100, 1))
RAD            = np.sqrt(X**2 + Y**2)
data           = X.copy()*0
data[100, 100] = 1e6

# or with a flat surface
title          = 'Flat surface'
data           = np.zeros((200,200))+10

psfconvolved   = PSFconvolution2D(data)

if title == 'Point-like function':
   minFWHM        = np.abs(psfconvolved - psfconvolved.max()/2.0)
   print(2*RAD[minFWHM == minFWHM.min()]*0.03)

f       = plt.figure(figsize=(14, 7))
ax1     = plt.subplot(121, title=title)
ax2     = plt.subplot(122, title='PSF convolution')

cmap    = 'plasma'
frac    = 0.06

tmp1    = ax1.imshow(data, origin='lower', cmap=cmap)
plt.colorbar(tmp1, ax=ax1, fraction=frac, orientation='horizontal')

tmp2    = ax2.imshow(psfconvolved, origin='lower', cmap=cmap)
plt.colorbar(tmp2, ax=ax2, fraction=frac, orientation='horizontal')

plt.show()
"""
