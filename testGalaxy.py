import matplotlib.pyplot as     plt
import numpy             as     np
import astropy.io.fits   as     fits
from   matplotlib.colors import Normalize
from   wilfried.galaxy   import mergeModelsIntoOne, bulgeDiskOnSky, model2D

# Check that the combine function behaves correctly
"""
f = plt.figure(figsize=(21, 7))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

x    = np.arange(0, 101, 1)
y    = x
X, Y = np.meshgrid(x, y)
Z    = X.copy()*0+1
Z2   = -Z
X2   = X.copy()+30

X3, Y3, Z3 = mergeModelsIntoOne([X, X2], [Y, Y], [Z, -Z], 1, 1)

contours = [-1, -0.5, 0.5, 1]
cmap     = 'bwr'
ax1.contourf(X, Y, Z, levels=contours, cmap=cmap)
ax2.contourf(X2, Y, Z, levels=contours, cmap=cmap)
data = ax3.contourf(X3, Y3, Z3, levels=contours, cmap=cmap)
plt.colorbar(data)
plt.show()
"""

# Check that a 2D model gives a result similar to Galfit 2D model
galfitModelFile = '/home/wilfried/Thesis/galfit/modelling/outputs/216_CGr32_out.fits'
with fits.open(galfitModelFile) as hdul:
   galfitData = hdul[2].data

shp = np.shape(galfitData)
Xgalfit, Ygalfit = np.meshgrid(np.linspace(-(shp[0]//2), shp[0]//2, shp[0]), np.linspace(-(shp[1]//2), shp[1]//2, shp[1]))

pixW   = 0.03 # arcsec/px
pixH   = 0.03 # arcsec/px

offset = 30
magD   = 18.78
magB   = 19.74
Rd     = 34.99
Rb     = 34.99
b_a    = 0.2
inc    = np.arccos(b_a)*180/np.pi # degrees
PA     = -34.41                   # degrees

nx, ny      = galfitData.shape
print(nx, ny)
# 2D model, using inclination and PA
X, Y, model = bulgeDiskOnSky(nx, ny, Rd, Rb, magD=magD, magB=magB, noPSF=True,
                             offsetD=offset, offsetB=offset, inclination=0, PA=0, combine=True)

# 2D "simple" model without sky prokection (to test whether the method above works)
Xsimple, Ysimple, modelSimple = model2D(nx, ny, listn=[1, 4], listRe=[Rd, Rb], listMag=[magD, magB], listOffset=[offset]*2)

f   = plt.figure(figsize=(14, 14))
ax1 = plt.subplot(221, title='Galfit 2D model')
ax2 = plt.subplot(222, title='2D model')
ax3 = plt.subplot(223, title='no projection')
ax4 = plt.subplot(224, title='residuals')

cmap = 'plasma'
norm = Normalize(vmin=galfitData.min(), vmax=galfitData.max())

#tmp1 = ax1.contourf(Xgalfit, Ygalfit, galfitData, cmap=cmap, norm=norm)
tmp1 = ax1.imshow(galfitData, origin='lower', cmap=cmap)
xbounds = ax1.get_xlim()
ybounds = ax1.get_ylim()

tmp2 = ax2.imshow(model, origin='lower', cmap=cmap, norm=norm)
#tmp2 = ax2.contourf(X[0], Y[0], model[0], cmap=cmap) #, norm=norm)
ax2.set_xlim(xbounds)
ax2.set_ylim(ybounds)

tmp3 = ax3.imshow(modelSimple, origin='lower', cmap=cmap, norm=norm)
ax3.set_xlim(xbounds)
ax3.set_ylim(ybounds)

tmp4 = ax4.imshow(modelSimple-model, origin='lower', cmap=cmap)
ax4.set_xlim(xbounds)
ax4.set_ylim(ybounds)
plt.colorbar(tmp4)
plt.show()


