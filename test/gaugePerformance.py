# Wilfried Mercier - IRAP

from   wilfried.galaxy                  import bulgeDiskOnSky
import numpy                            as     np
import astropy.io.fits                  as     fits
from   time                             import time
from   wilfried.utilities.plotUtilities import asManyPlots2

# Galfit model

listBin         = np.arange(1, 45, 2)
"""
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
   galfitData   = hdul[0].data

sumGalfit       = np.nansum(galfitData)
shp             = np.shape(galfitData)
pixW            = 0.03 # arcsec/px
pixH            = pixW # arcsec/px
nx, ny          = shp

# Time it takes when sampling the whole image or the 21x21 central pixels only
timeAll         = []
timeCen         = []

# Maximum relative difference between galfit model and projected one (my own) when sampling the whole image or just the central part
maxDiffRelAll   = []
maxDiffRelCen   = []

# Relative difference in the total flux
sumDiffRelAll   = []
sumDiffRelCen   = []

sumDiffRel3arcAll = []
sumDiffRel3arcCen = []

x0              = 99
y0              = 99
for bin in listBin:
   start          = time()
   X, Y, modelAll = bulgeDiskOnSky(nx, ny, x0=x0, y0=y0, Rd=Rd, Rb=Rb, magD=magD, magB=magB, offsetD=offset, offsetB=offset, inclination=inc, PA=PA,
                                   combine=True, noPSF=True, fineSampling=bin, samplingZone={'where':'all'})
   timeAll.append(time()-start)
   maxDiffRelAll.append(np.nanmax((modelAll-galfitData)/galfitData))
   sumDiffRelAll.append((np.nansum(modelAll)-sumGalfit)/sumGalfit)
   mask           = np.sqrt(X**2+Y**2) <= 1.5/0.3
   sumGalfitM = np.nansum(galfitData[mask])
   sumDiffRel3arcAll.append((np.nansum(modelAll[mask])-sumGalfitM)/sumGalfitM)

   start          = time()
   X, Y, modelCen = bulgeDiskOnSky(nx, ny, x0=99, y0=99, Rd=Rd, Rb=Rb, magD=magD, magB=magB, offsetD=offset, offsetB=offset, inclination=inc, PA=PA,
                                   combine=True, noPSF=True, fineSampling=bin, samplingZone={'where':'centre', 'dx':20, 'dy':20})
   timeCen.append(time()-start)
   maxDiffRelCen.append(np.nanmax((modelCen-galfitData)/galfitData))
   sumDiffRelCen.append((np.nansum(modelCen)-sumGalfit)/sumGalfit)
   mask           = np.sqrt(X**2+Y**2) <= 1.5/0.3
   sumDiffRel3arcCen.append((np.nansum(modelCen[mask])-sumGalfitM)/sumGalfitM)

print(timeAll)
print(timeCen)
print(maxDiffRelAll)
print(maxDiffRelCen)
print(sumDiffRelAll)
print(sumDiffRelCen)
np.savetxt('save.data', np.array([timeAll, timeCen, maxDiffRelAll, maxDiffRelCen, sumDiffRelAll, sumDiffRelCen, sumDiffRel3arcAll, sumDiffRel3arcCen]).T, header='timeAll timeCen maxDiffRelAll maxDiffRelCen sumDiffRelAll sumDiffRelCen sumDiffRel3arcAll sumDiffRel3arcCen')
"""


import matplotlib.pyplot as plt
file = 'save.data'
timeAll, timeCen, maxDiffRelAll, maxDiffRelCen, sumDiffRelAll, sumDiffRelCen, sumDiffRel3arcAll, sumDiffRel3arcCen = np.genfromtxt(file, skip_header=1, unpack=True)

f       = plt.figure(figsize=(12, 12))
ax, ret = asManyPlots2(111, [listBin, listBin], [timeAll, timeCen],
                       dataProperties={'color':['k', 'red'], 'marker':['o', 'o'], 'unfillMarker':[False, True], 'order':[2, 2]},
                       generalProperties={'gridZorder':0},
                       axesProperties={'xlabel':'Rebinning factor', 'ylabel':'Time (s)', 'yscale':'log'},
                       legendProperties={'labels':['Full rebinning', r'Central rebinning ($21\times 21 \, \rm{px}$)']}
                      )

f       = plt.figure(figsize=(12, 12))
ax, ret = asManyPlots2(111, [listBin, listBin], [maxDiffRelAll, maxDiffRelCen],
                       dataProperties={'color':['k', 'red'], 'marker':['o', 'o'], 'unfillMarker':[False, True], 'order':[2, 2]},
                       generalProperties={'gridZorder':0},
                       axesProperties={'xlabel':'Rebinning factor', 'ylabel':r'$\max \left [ \rm{( model - Galfit ) /Galfit} \right ]$', 'yscale':'log'},
                       legendProperties={'labels':['Full rebinning', r'Central rebinning ($21\times 21 \, \rm{px}$)']}
                      )

f       = plt.figure(figsize=(12, 12))
ax, ret = asManyPlots2(111, [listBin, listBin], [sumDiffRelAll, sumDiffRelCen],
                       dataProperties={'color':['k', 'red'], 'marker':['o', 'o'], 'unfillMarker':[False, True], 'order':[2, 2]},
                       generalProperties={'gridZorder':0},
                       axesProperties={'xlabel':'Rebinning factor', 'ylabel':r'$\rm{(Flux_{model} - Flux_{Galfit})/Flux_{Galfit}}$', 'yscale':'log'},
                       legendProperties={'labels':['Full rebinning', r'Central rebinning ($21\times 21 \, \rm{px}$)']}
                      )

f       = plt.figure(figsize=(12, 12))
ax, ret = asManyPlots2(111, [listBin, listBin], [sumDiffRel3arcAll, sumDiffRel3arcCen],
                       dataProperties={'color':['k', 'red'], 'marker':['o', 'o'], 'unfillMarker':[False, True], 'order':[2, 2]},
                       generalProperties={'gridZorder':0},
                       axesProperties={'xlabel':'Rebinning factor', 'ylabel':r'$\left [ {\rm{(Flux_{model} - Flux_{Galfit})/Flux_{Galfit}}} \, \right ] (3^{\prime \prime})$', 'yscale':'log'},
                       legendProperties={'labels':['Full rebinning', r'Central rebinning ($21\times 21 \, \rm{px}$)']}
                      )
plt.show()

