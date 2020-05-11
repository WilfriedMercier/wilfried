# Mercier Wilfried - IRAP

import numpy              as     np
from   astropy.io.votable import parse
from   wilfried.galaxy    import ratioLuminosities2D, ratioLuminosities1D

# Loading data
path          = '/home/wilfried/Thesis/catalogues/VOtables/perso/field_gals/newCatalogues/withMorpho/Valentina_paper/'
name          = 'fieldsGals_CONFID_2_3_withLaigle+16_withFAST_withPLATEFIT_jan20_withFOF_jan19_withGALFIT_jan20_removedNoMorpho.vot'
file          = path + name
table         = parse(file).get_first_table().array

offset        = 30
Rd            = table['R_d_GF']
Rb            = table['R_b_GF']
magD          = table['Mag_d_GF']
magB          = table['Mag_b_GF']
inc           = np.arccos(table['b_a_d_GF'])
PA            = table['pa_d_GF']

# First radius = 2.2*disk scale length
r1            = 2.2*Rd/1.67835

# Second radius = 1.5" -> must be converted in pixels
arcsecToGrid  = 0.03
r2            = 1.5/arcsecToGrid

ratioLum1D    = []
ratioLum2D    = []
for rd, rb, magd, magb, i, pa, r in zip(Rd, Rb, magD, magB, inc, PA, r1):

   # Ratio here is Flux(1.5")/Flux(R22)
   ratioLum1D.append(ratioLuminosities1D(r2, r, [1, 4], [rd, rb], listMag=[magd, magb], listOffset=[offset, offset]))

   # where is ['sky', 'galaxy'] because we compute Flux(1.5") on sky plane, and Flux(R22) on galaxy plane
   # noPSF is [False, True] because we perform PSF convolution for the projected model (1.5", on sky), but not for the model on galaxy plane (R22)
   ratioLum2D.append(ratioLuminosities2D(r2, r, rd, rb, where=['sky', 'galaxy'], noPSF=[False, True],
                                         magD=magd, magB=magb, offsetD=offset, offsetB=offset, inclination=i, PA=pa, arcsecToGrid=arcsecToGrid))
   print("%.2f%%" %(len(ratioLum2D)/len(Rd)*100))
