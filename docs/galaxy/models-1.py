from   matplotlib.colors   import LogNorm
from   matplotlib          import rc
from   matplotlib.gridspec import GridSpec
from   wilfried.galaxy     import models as mod
import matplotlib.pyplot   as     plt
import matplotlib          as     mpl

# Bulge model using fine sampling and without PSF
X, Y, bulge1 = mod.bulge2D(100, 100, 35, mag=20, offset=30, noPSF=True)

# Bulge mode with fine sampling but without PSF
X, Y, bulge2 = mod.bulge2D(100, 100, 35, mag=20, offset=30, noPSF=True, fineSampling=81)

# Bulge model with fine sampling and with PSF convolution (FWHM=0.8 arcsec)
X, Y, bulge3 = mod.bulge2D(100, 100, 35, mag=20, offset=30, noPSF=False, fineSampling=81,
                          PSF={'name':'Gaussian2D', 'FWHMX':0.8, 'FWHMY':0.8, 'unit':'arcsec'}, arcsecToGrid=0.03)

###############################
#          Plot part          #
###############################

# Setup figure and axes
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'

f            = plt.figure(figsize=(9, 3))
gs           = GridSpec(1, 3, figure=f, wspace=0, hspace=0, left=0.2, right=0.8, top=0.9, bottom=0.3)

ax1          = f.add_subplot(gs[0])
ax2          = f.add_subplot(gs[1])
ax3          = f.add_subplot(gs[2])

ax1.set_title(r'No fine sampling',      size=15)
ax2.set_title(r'Fine sampling = $9^2$', size=15)
ax3.set_title(r'Fine sampling \& PSF',  size=15)

for a in [ax1, ax2, ax3]:
   a.set_xticklabels([])
   a.set_yticklabels([])
   a.tick_params(axis='x', which='both', direction='in')
   a.tick_params(axis='y', which='both', direction='in')
   a.yaxis.set_ticks_position('both')
   a.xaxis.set_ticks_position('both')

# Show bulges
ret1  = ax1.imshow(bulge1, origin='lower', norm=LogNorm(), cmap='plasma')
ret2  = ax2.imshow(bulge2, origin='lower', norm=LogNorm(), cmap='plasma')
ret3  = ax3.imshow(bulge3, origin='lower', norm=LogNorm(), cmap='plasma')

# Add colorbar
cb_ax = f.add_axes([0.2, 0.2, 0.6, 0.025])
cbar  = f.colorbar(ret1, cax=cb_ax, orientation='horizontal')
cbar.set_label(r'Surface brightness [arbitrary unit]', size=15)
cbar.ax.tick_params(labelsize=15)

plt.show()