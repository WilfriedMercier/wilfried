import matplotlib.pyplot as     plt
import numpy             as     np
import astropy.io.fits   as     fits
from   matplotlib.colors import Normalize, LogNorm, DivergingNorm
from   wilfried.galaxy   import bulgeDiskOnSky, model2D, PSFconvolution2D

# Check that a 2D model for a disk (or bulge) only work properly by comparing it with galfit output
"""
path = '/home/wilfried/bin/own_libs/'

##################
#      Disk      #
##################

# Without inclincation, nor PA

#galfitModelFile = path + 'sersic_n1_re10.5_magtot15_axisratio1_pa0_magoffset30.fits'
#n               = 1
#offset          = 30
#mag             = 15
#Re              = 10.5
#b_a             = 1
#inc             = np.arccos(b_a)*180/np.pi # degrees
#PA              = 0                        # degrees


# With inclination
#galfitModelFile = path + 'sersic_n1_re10.5_magtot15_axisratio0.5_pa0_magoffset30.fits'
galfitModelFile = path + 'sersic_n1_re10.5_magtot15_axisratio0.4_pa0_magoffset30.fits'
n               = 1
offset          = 30
mag             = 15
Re              = 10.5
b_a             = 0.4
inc             = np.arccos(b_a)*180/np.pi # degrees
PA              = 0                        # degrees
#PA              = 10

###############
#    Bulge    #
###############

galfitModelFile = path + 'sersic_n4_re6_magtot17_axisratio1_pa0_magoffset30.fits'
n               = 4
offset          = 30
mag             = 17
Re              = 6
b_a             = 1
inc             = np.arccos(b_a)*180/np.pi # degrees
PA              = 0                        # degrees

with fits.open(galfitModelFile) as hdul:
   galfitData    = hdul[0].data

shp              = np.shape(galfitData)
x                = np.linspace(-(shp[0]//2), shp[0]//2, shp[0])
y                = np.linspace(-(shp[0]//2), shp[0]//2, shp[0])
Xgalfit, Ygalfit = np.meshgrid(x, y)

pixW   = 0.03 # arcsec/px
pixH   = 0.03 # arcsec/px
nx, ny = shp
pxScl  = 1

# 2D model, using inclination and PA
X, Y, model = model2D(nx, ny, [n], [Re], listMag=[mag], listOffset=[offset], listInclination=[inc], listPA=[PA],
                           pixScale=pxScl, combine=True)
model   = model[1:, 1:]

f       = plt.figure(figsize=(14, 14))
ax1     = plt.subplot(131, title='Galfit 2D model')
ax3     = plt.subplot(132, title='2D model with projection')
ax4     = plt.subplot(133, title='2D model/Galfit')

cmap    = 'plasma'

norm    = Normalize()
tmp1    = ax1.imshow(galfitData, origin='lower', cmap=cmap, norm=norm)
plt.colorbar(tmp1, ax=ax1, fraction=0.03, orientation='horizontal')

tmp3    = ax3.imshow(model, origin='lower', cmap=cmap, norm=norm)
plt.colorbar(tmp3, ax=ax3, fraction=0.03, orientation='horizontal')

tmp4    = ax4.imshow(model/galfitData, origin='lower', cmap='Reds', norm=Normalize(vmin=1))
plt.colorbar(tmp4, ax=ax4, fraction=0.03, orientation='horizontal')

plt.show()
"""

# Check that a 2D model (disk + bulge) works properly

path            = '/home/wilfried/bin/own_libs/'
galfitModelFile = path + 'sersic_n1_4_re10.5_6_magtot15_17_axisratio0.4_magoffset30.fits'
offset          = 30
magD            = 15
magB            = 17
Rd              = 10.5
Rb              = 6
b_a             = 0.4
inc             = np.arccos(b_a)*180/np.pi  # degrees
PA              = 0                       # degrees

with fits.open(galfitModelFile) as hdul:
   galfitData    = hdul[0].data

shp              = np.shape(galfitData)
x                = np.linspace(-(shp[0]//2), shp[0]//2, shp[0])
y                = np.linspace(-(shp[0]//2), shp[0]//2, shp[0])
Xgalfit, Ygalfit = np.meshgrid(x, y)

pixW   = 0.03 # arcsec/px
pixH   = 0.03 # arcsec/px
nx, ny = shp
pxScl  = 1

#X, Y, model = bulgeDiskOnSky(nx, ny, Rd=Rd, Rb=Rb, magD=magD, magB=magB, offsetD=offset, offsetB=offset, inclination=inc, PA=PA, noPSF=True, combine=True)
FWHM        = 3*0.03
fSampling   = 9
X, Y, model = bulgeDiskOnSky(nx, ny, x0=99, y0=99,
                             Rd=Rd, Rb=Rb, magD=magD, magB=magB, offsetD=offset, offsetB=offset, inclination=inc, PA=PA, 
                             combine=True, fineSampling=fSampling, noPSF=True,
                             PSF={'name':'Gaussian2D', 'FWHMX':FWHM, 'FWHMY':FWHM, 'sigmaX':None, 'sigmaY':None, 'unit':'arcsec'})

f       = plt.figure(figsize=(12, 12))
ax1     = plt.subplot(231, title='Galfit 2D model')
ax2     = plt.subplot(232, title='2D model with projection')
ax5     = plt.subplot(233, title='2D model rebinned')
ax3     = plt.subplot(234, title='2D model/Galfit')
ax4     = plt.subplot(235, title='2D model - Galfit')
ax6     = plt.subplot(236, title='(2D model-Galfit)/Galfit')

cmap    = 'plasma'
frac    = 0.06

norm    = Normalize()
tmp1    = ax1.imshow(galfitData, origin='lower', cmap=cmap, norm=norm)
plt.colorbar(tmp1, ax=ax1, fraction=frac, orientation='horizontal')

tmp2    = ax2.imshow(model, origin='lower', cmap=cmap, norm=norm)
plt.colorbar(tmp2, ax=ax2, fraction=frac, orientation='horizontal')

# We rebin out data because we oversampled it
model   = model.reshape(int(model.shape[0] / fSampling), fSampling, int(model.shape[1] / fSampling), fSampling)  # Note that data is the high resolution image and that its size has to be a multiple of the oversampling factor.
model   = model.mean(1).mean(2)  # note that it could also be sum instead of mean, depending on what we want.

tmp5    = ax5.imshow(model, origin='lower', cmap=cmap, norm=norm)
plt.colorbar(tmp5, ax=ax5, fraction=frac, orientation='horizontal')

tmp3    = ax3.imshow(model/galfitData, origin='lower', cmap='bwr', norm=DivergingNorm(vcenter=1.0, vmax=np.nanmax(model/galfitData)))
plt.colorbar(tmp3, ax=ax3, fraction=frac, orientation='horizontal')

tmp4    = ax4.imshow(model-galfitData, origin='lower', cmap='bwr', norm=Normalize())
plt.colorbar(tmp4, ax=ax4, fraction=frac, orientation='horizontal')

tmp6    = ax6.imshow((model-galfitData)/galfitData, origin='lower', cmap='bwr', norm=Normalize())
plt.colorbar(tmp6, ax=ax6, fraction=frac, orientation='horizontal')

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
