from   matplotlib.colors   import LogNorm, Normalize
from   matplotlib          import rc
from   matplotlib.gridspec import GridSpec
from   astropy.io          import fits
from   wilfried.galaxy     import models as mod
import matplotlib.pyplot   as     plt
import matplotlib          as     mpl
import numpy               as     np

##############################
#     Generate 2D models     #
##############################

X, Y, bulge1 = mod.bulge2D(100, 100, 15, mag=21, offset=30, noPSF=True,  fineSampling=81, samplingZone={'where':'centre', 'dx':5, 'dy':5})
X, Y, bulge2 = mod.bulge2D(100, 100, 15, mag=21, offset=30, noPSF=False, fineSampling=81, samplingZone={'where':'centre', 'dx':5, 'dy':5},
                          PSF={'name':'Gaussian2D', 'FWHMX':0.087, 'FWHMY':0.087, 'unit':'arcsec'}, arcsecToGrid=0.03)

#####################################
#     Load GALFIT output images     #
#####################################

with fits.open('/home/wilfried/wilfried_libs/galaxy/.test/bulge_withoutPSF.fits') as hdul:
   bulge_nopsf = hdul[0].data

with fits.open('/home/wilfried/wilfried_libs/galaxy/.test/bulge_withPSF.fits') as hdul:
   bulge_psf   = hdul[0].data

##########################
#       Plot parts       #
##########################

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'

f            = plt.figure(figsize=(9, 6))
gs           = GridSpec(2, 3, figure=f, wspace=0, hspace=0, left=0.1, right=0.9, top=0.95, bottom=0.15)

axs          = []
for i in gs:
   axs.append(f.add_subplot(i))

axs[0].set_ylabel(r'without PSF', size=15)
axs[0].set_title(r'bulge2D',      size=15)
axs[1].set_title(r'Galfit bulge', size=15)
axs[2].set_title(r'Residuals',    size=15)
axs[3].set_ylabel(r'with PSF',    size=15)


for a in axs:
   a.set_xticklabels([])
   a.set_yticklabels([])

   a.tick_params(axis='x', which='both', direction='in')
   a.tick_params(axis='y', which='both', direction='in')
   a.yaxis.set_ticks_position('both')
   a.xaxis.set_ticks_position('both')

# First line
mmax  = np.nanmax([bulge1, bulge2, bulge_nopsf, bulge_psf])
mmin  = np.nanmin([bulge1, bulge2, bulge_nopsf, bulge_psf])

diff1 = 100 - 100*bulge_nopsf/bulge1
diff2 = 100 - 100*bulge_psf/bulge2
dmax  = np.nanmax([diff1, diff2])
dmin  = np.nanmin([diff1, diff2])

ret11 = axs[0].imshow(bulge1,      origin='lower', cmap='plasma', norm=LogNorm(  vmin=mmin, vmax=mmax))
ret12 = axs[1].imshow(bulge_nopsf, origin='lower', cmap='plasma', norm=LogNorm(  vmin=mmin, vmax=mmax))
ret13 = axs[2].imshow(diff1,       origin='lower', cmap='viridis',    norm=Normalize(vmin=dmin, vmax=dmax))

# Second line
ret21 = axs[3].imshow(bulge2,      origin='lower', cmap='plasma', norm=LogNorm(  vmin=mmin, vmax=mmax))
ret22 = axs[4].imshow(bulge_psf,   origin='lower', cmap='plasma', norm=LogNorm(  vmin=mmin, vmax=mmax))
ret23 = axs[5].imshow(diff2,       origin='lower', cmap='viridis',    norm=Normalize(vmin=dmin, vmax=dmax))

# Add colorbar 1
cb_ax1 = f.add_axes([0.1, 0.1, 0.53, 0.025])
cbar1  = f.colorbar(ret11, cax=cb_ax1, orientation='horizontal')
cbar1.set_label(r'Surface brightness [arbitrary unit]', size=15)
cbar1.ax.tick_params(labelsize=15)

# Add colorbar 2
cb_ax2 = f.add_axes([0.64, 0.1, 0.26, 0.025])
cbar2  = f.colorbar(ret13, cax=cb_ax2, orientation='horizontal')
cbar2.set_label(r'Relative difference (\%)', size=15)
cbar2.ax.tick_params(labelsize=15)

plt.show()