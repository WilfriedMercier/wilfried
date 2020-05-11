# Mercier Wilfried - IRAP

import numpy              as     np
from   astropy.io.votable import parse
from   astropy.table      import Table
from   wilfried.utilities.plotUtilities import asManyPlots2
import matplotlib.pyplot  as plt
from   wilfried.galaxy    import ratioLuminosities2D, ratioLuminosities1D, solve_re, luminositySersics

path  = '/home/wilfried/Thesis/catalogues/VOtables/perso/field_gals/newCatalogues/withMorpho/Valentina_paper/'
name  = 'fieldsGals_CONFID_2_3_withLaigle+16_withFAST_withPLATEFIT_jan20_withFOF_jan19_withGALFIT_jan20_removedNoMorpho.vot'
file  = file = path + name

table = parse(file).get_first_table().array

arcsecToGrid  = 0.03
offset        = 30

#table         = table[np.logical_and(table['ID']==54, table['COSMOS_Group_Number']==79.0)]
Rd            = table['R_d_GF']
Rb            = table['R_b_GF']
mask          = np.logical_and(Rd<=100, Rb<=100)

table         = table[mask]
Rd            = Rd[mask]
Rb            = Rb[mask]
magD          = table['Mag_d_GF']
magB          = table['Mag_b_GF']
#inc           = np.arccos(table['b_a_d_GF'])
inc           = [0]*len(Rd)
PA            = [0]*len(Rd)
#PA            = table['pa_d_GF']

# True effective radius
trueRe, conv, flag = solve_re(table)

# First radius = 2.2* disk scale length
r1            = 2.2*Rd/1.67835

# Second radius = 1.5" -> must be converted in pixels
r2            = 1.5/arcsecToGrid

ratioLum1D    = []
ratioLum2D    = []
ratioLum2DPSF = []
for rd, rb, magd, magb, i, pa, r in zip(Rd, Rb, magD, magB, inc, PA, r1):
   ratioLum1D.append(ratioLuminosities1D(r2, r, [1, 4], [rd, rb], listMag=[magd, magb], listOffset=[offset, offset], analytical=False))
   ratioLum2D.append(ratioLuminosities2D(r2, r, rd, rb, where=['sky', 'galaxy'], noPSF=[True, True], fineSampling=81,
                                         magD=magd, magB=magb, offsetD=offset, offsetB=offset, inclination=i, PA=pa, arcsecToGrid=arcsecToGrid))
   ratioLum2DPSF.append(ratioLuminosities2D(r2, r, rd, rb, where=['sky', 'galaxy'], noPSF=[False, True], fineSampling=81,
                                         magD=magd, magB=magb, offsetD=offset, offsetB=offset, inclination=i, PA=pa, arcsecToGrid=arcsecToGrid))
   print("%.2f%%" %(len(ratioLum2D)/len(Rd)*100))

where = np.argmax((np.array(ratioLum2D)-np.array(ratioLum1D))/np.array(ratioLum1D))
print(Table(table)[where]['COSMOS_Group_Number', 'ID', 'R_d_GF', 'R_b_GF', 'Mag_d_GF', 'Mag_b_GF', 'b_a_d_GF', 'pa_d_GF'])

plt.figure(figsize=(12, 12))
ax, ret = asManyPlots2(111, [r1*arcsecToGrid], [(np.array(ratioLum2D)-np.array(ratioLum1D))/np.array(ratioLum1D)],
                       dataProperties={'linestyle':['none'], 'markerSize':[5], 'order':[2], 'color':[ratioLum1D], 'type':['scatter']},
                       colorbarProperties={'cmap':'spring', 'label':'ratio Flux 1D'},
                       generalProperties={'gridZorder':0},
                       axesProperties={'xlabel':'2.2Rd (arcsec)', 'ylabel':'(2D-1D)/1D'})

plt.figure(figsize=(12, 12))
ax, ret = asManyPlots2(111, [ratioLum2D], [np.array(ratioLum2DPSF)/np.array(ratioLum2D)-1], #[r1*arcsecToGrid], [ratioLum2D],
                       dataProperties={'linestyle':['none'], 'markerSize':[5], 'order':[2], 'color':[r1*arcsecToGrid], 'type':['scatter']},
                       colorbarProperties={'cmap':'spring', 'label':'R22 (arcsec)'},
                       generalProperties={'gridZorder':0},
                       axesProperties={'xlabel':'Flux2D(R=1.5")/Flux2D(R22)', 'ylabel':'(2D with PSF - 2D)/2D'})
#bounds = ax.get_xlim()
#ax.plot(bounds, [1, 1], 'k--')
#ax.set_xlim(bounds)
plt.show()
