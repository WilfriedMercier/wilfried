#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: B. Epinat - LAM
modified by W. Mercier - IRAP

Create recap morpho-kinematics plots for MUSE \& HST modelling.
'''

import copy
import numpy                 as np
import astropy.wcs           as wcs
import astropy.io.fits       as fits
import matplotlib            as     mpl
import matplotlib.pyplot     as     plt
from   matplotlib.patches    import Circle
from   matplotlib.colors     import LogNorm
from   matplotlib            import rc
from   matplotlib            import gridspec
from   matplotlib.ticker     import AutoMinorLocator
from   scipy.ndimage.filters import gaussian_filter

# Setup figure parameters
mpl.style.use('classic')
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
rc('figure', figsize=(6.0, 4.5))

def plot_hst(gs, hstmap, xc, yc, pix, xminh, xmaxh, yminh, ymaxh, xch, xmah1, xmah2, ych, ymah1, ymah2, pixh, ticklabels, vmin, vmax, pa=False, title=False, fluxcont=[0]):
    '''
    General plot given a 2D array.
    
    Author : B. Epinat - LAM
    
    Mandatory parameters
    --------------------
        hstmap : numpy 2D array
            2D image data
        gs : gridspec element
            matplotlib GridSpec grid element
        pix : float
            MUSE pixel scale in arcsec/pixel
        pixh : float
            HST pixel scale in arcsec/pixel
        ticklabels : numpy array
            labels for the ticks in arcsec
        xc : int/float
            MUSE centre x-axis position
        xch : int/float
            HST centre x-axisposition
        xmaxh : int/float
            x-axis HST maximum value
        xmah1 : int/float
            x-axis HST PA minimum value 
        xmah2 : int/float
            x-axis HST PA maximum value
        xminh : int/float
            x-axis HST minimum value
        yc : int/float
            MUSE centre y-axis position
        ych : int/float
            HST centre y-axis position
        ymaxh : int/float
            y-axis HST maximum value
        ymah1 : int/float
            y-axis HST PA minimum value
        ymah2 : int/float
            y-axis HST PA maximum value
        yminh : int/float
            y-axis HST minimum value
        vmax : int/float
            maximum data value used to set the logarithmic scale
        vmin : int/float
            minimum data value used to clip data and set the logarithmic scale
            
    Optional parameters
    -------------------
        fluxcont : list/numpy array
            MUSE flux values used to show as contours on top of HST image.
        pa : bool
            whether to draw the HST PA line or not. Default is to not draw it.
        title : bool
            whether to show a title or not as well as the x-axis label on top. Default is to not shown any title.
            
    Return the axis and plot objects.
    '''
    
    # Generate axis
    axhstmap              = plt.subplot(gs)
    axhstmap.set_xlim(xminh, xmaxh)
    axhstmap.set_ylim(yminh, ymaxh)
    
    # Clipping and plotting (logarithmic scale)
    hstmap[hstmap < vmin] = vmin
    imfluxmap             = axhstmap.imshow(hstmap, cmap=plt.cm.gray_r, origin='lower', interpolation='nearest')
    
    ##################################################################
    #                  Setting x and y ticks labels                  #
    ##################################################################
    
    # Converting ticks values from arcsec to HST pixel and centre on HST centre
    xticks  = ticklabels/pixh + xch
    axhstmap.set_xticks(xticks)
    axhstmap.set_xticklabels(-ticklabels)
    
    yticks  = ticklabels/pixh + ych
    axhstmap.set_yticks(yticks)
    axhstmap.set_yticklabels(ticklabels)
    
    # Setting minor ticks number between successive major ticks to 4
    ml      = AutoMinorLocator(4)
    axhstmap.xaxis.set_minor_locator(ml)
    axhstmap.yaxis.set_minor_locator(ml)
    
    axhstmap.set_ylabel(r"$\Delta \delta~('')$")
    axhstmap.tick_params(labelbottom=False, labelleft=True, labelright=False, labeltop=False)
    
    ########################################################
    #              Additionnal optional plots              #
    ########################################################
    
    if pa:
        axhstmap.plot([xch], [ych], 'g+', mew=1)
        axhstmap.plot([xmah1, xmah2], [ymah1, ymah2], 'g')
    if title:
        axhstmap.set_xlabel(r"$\Delta \alpha~('')$")
        axhstmap.tick_params(labelbottom=False, labelleft=True, labelright=False, labeltop=True)
        axhstmap.xaxis.set_label_position("top")
        axhstmap.set_title(r'HST-ACS F814W')
    if len(fluxcont) > 1:
        
        cont_levels = np.logspace(np.log10(250), np.log10(8000), 6)
        
        # Scale factor to convert from HST pixel to MUSE pixel
        scalex      = pix / pixh
        scaley      = scalex
        
        # Generate a transformation to apply onto contours to have them scale correctly with HST pixels (remove MUSE centre coordinate to be centred on position (0,0), apply pixel scale transform to go into HST pixels, and centre into HST centre position)
        scaletrans  = mpl.transforms.Affine2D().translate(-xc, -yc).scale(scalex, scaley).translate(xch, ych)
        trans_data  = scaletrans + axhstmap.transData
        axhstmap.contour(np.log10(fluxcont), levels=np.log10(cont_levels), cmap='brg', transform=trans_data)

    return axhstmap, imfluxmap


def paper_map(hst, flux, snr, vf, vfm, vfr, sig, sigm, sigr, name, z, xc, yc, vsys, pa, r22, rc, lsf, psf, pathout, deltapa=-42, offset=[0,0], 
              MUSE_smooth_sigma=1.0, withSNR=False, withFlux=False, hstInput=None):
    '''
    Create the morpho-kinematics recap maps.
    
    Mandatory parameters
    --------------------
        flux : list of str
            list of MUSE flux maps file names
        hst : str
            file name of the GALFIT output fits file containing the image as the 1st extension, the model as the 2nd extension and the residuals as the 3rd
        lsf : float
            MUSE LSF FWHM at the galaxy redshift
        name : str
            galaxy name
        pa : int/float
            position angle in degrees
        pathout : str
            path for the output pdf file
        psf : float
            MUSE PSF FWHM at the galaxy redshift
        r22 : float
            2.2*disk scalelength in HST pixels
        rc : float
            last radius in MUSE velocity field in MUSE pixels
        sig : str
            velocity dispersion map file name
        sigm : str
            velocity dispersion map model file name
        sigr : str
            velocity dispersion map residuals file name
        snr : str
            MUSE snr map file name
        vf : str
            CAMEL velocity field file name
        vfm : str
            velocity field model file name
        vfr : str
            velocity field 1residuals file name
        vsys : float
            galaxy systemic velocity in km/s
        xc : int/float
            x-axis centre position in MUSE pixels
        yc : int/float
            y-axis centre position in MUSE pixels
        z : float
            galaxy redshift
            
    Optional parameters
    -------------------
        deltapa : int/float
            offset to apply to the galaxy position angle. Default is -42°. WHY ?
        hstInput : str
            name of the HST input file name used to replace the image located in the galfit output file. Default is None so that the Galfit output image is used.
        MUSE_smooth_sigma : float
            smoothing value (Gaussian std) used for Gaussian smoothing the MUSE flux data. Default is 1.0 MUSE pixel.
        offset : list of two int/float
            offsets to subtract from the x and y reference pixel coordinates (in the same pixel coordinates as in the fits files). Default is 0 for both.
        withFlux : bool
            whether to add a (combination of) MUSE flux map(s) or not. Default is False.
        withSNR : bool
            whether to add a MUSE SNR map or not. Default is False.
    '''
    
    ##################################################
    #                  Getting data                  #
    ##################################################
    
    # HST data
    with fits.open(hst) as hsthdu:
        hstimhdu         = hsthdu[1]
        hstmodhdu        = hsthdu[2]
        hstreshdu        = hsthdu[3]
    
        # HST image
        hstimmap             = hstimhdu.data
        hstimhdr             = hstimhdu.header
        
        # Galfit HST model
        hstmodmap            = hstmodhdu.data
        hstmodhdr            = copy.deepcopy(hstimhdr)
        
        # Galfit HST residuals
        hstresmap            = hstreshdu.data
        hstreshdr            = copy.deepcopy(hstimhdr)
        
    # If HST image is given, we replace the one found in Galfit output file
    if hstInput is not None:
        with fits.open(hstInput) as hsthdu:
            hstimmap         = hsthdu[0].data
    
    # Get and apply x and y offsets
    offstr               = hstimhdr['OBJECT']
    boundaries           = offstr.replace('[', ' ').replace(',', ' ').replace(']', ' ').replace(':', ' ').split()
    offx                 = np.int(boundaries[0]) - 1
    offy                 = np.int(boundaries[2]) - 1
    
    hstimhdr['CRPIX1']  += offset[1] - offx
    hstimhdr['CRPIX2']  += offset[0] - offy
    
    # Opening and combining MUSE flux maps
    if isinstance(flux, str):
        flux             = [flux]
    
    with fits.open(flux[0]) as fluxhdu:
        fluxmap          = fluxhdu[0].data * 0
        fluxhdr          = fluxhdu[0].header
        
    for file in flux:
        with fits.open(file) as temphdu:
            fluxmap     += temphdu[0].data
            
    pixarea              = (fluxhdr['CD1_1'] * 3600) ** 2  # arcsec**2 
    fluxmap             /= pixarea
    
    # SNR file
    with fits.open(snr) as snrhdu:
        snrmap           = snrhdu[0].data
        snrhdr           = snrhdu[0].header
        
    # Velocity field (CAMEL map)
    with fits.open(vf) as vfhdu:
        vfmap            = vfhdu[0].data
        vfhdr            = vfhdu[0].header
    
    # Velocity field model
    with fits.open(vfm) as vfmhdu:
        vfmmap           = vfmhdu[0].data
    
    # Velocity field residuals
    with fits.open(vfr) as vfrhdu:
        vfrmap           = vfrhdu[0].data
    
    # Velocity dispersion map (CAMEL)
    with fits.open(sig) as sighdu:
        sigmap           = sighdu[0].data
        sighdr           = sighdu[0].header

    # Velocity dispersion map model    
    with fits.open(sigm) as sigmhdu:
        sigmmap          = sigmhdu[0].data

    # Velocity dispersion map residuals  
    with fits.open(sigr) as sigrhdu:
        sigrmap          = sigrhdu[0].data
    
    
    #########################################################
    #                 Setting up parameters                 #
    #########################################################
    
    # Defining wcs
    fluxhdr['WCSAXES']   = 2
    vfhdr['WCSAXES']     = 2
    sighdr['WCSAXES']    = 2
    snrhdr['WCSAXES']    = 2
    hstimhdr['WCSAXES']  = 2
    hstmodhdr['WCSAXES'] = 2
    hstreshdr['WCSAXES'] = 2
    fluxwcs              = wcs.WCS(fluxhdr)
    snrwcs               = wcs.WCS(snrhdr)
    
    hstimwcs             = wcs.WCS(hstimhdr)
    
    # Get MUSE and HST pixel scales in arcsec as the geometric mean of the x and y pixel scales
    pix                  = np.sqrt(fluxhdr['CD1_1']**2  + fluxhdr['CD1_2'] **2) * 3600
    pixh                 = np.sqrt(hstimhdr['CD1_1']**2 + hstimhdr['CD1_2']**2) * 3600
    
    # Smoothing data
    fluxcont             = gaussian_filter(fluxmap,  MUSE_smooth_sigma)
    
    # Major axis (in MUSE pixel)
    r22m                 = r22 * pixh / pix
    
    # Rlast in arcsec
    rcas                 = rc * pix
    
    # X and y positions for the major axis, taking into account PA, given in MUSE pixel coordinates
    xma1                 = xc + r22m * np.sin(np.radians(pa+deltapa))
    yma1                 = yc - r22m * np.cos(np.radians(pa+deltapa))
    xma2                 = xc - r22m * np.sin(np.radians(pa+deltapa))
    yma2                 = yc + r22m * np.cos(np.radians(pa+deltapa))
    
    # Get world coordinates (ra, dec pairs) for the centre and the two end points of the PA line
    ra0, dec0            = fluxwcs.wcs_pix2world([xc, xma1, xma2], [yc, yma1, yma2], 0)
    
    # Defining the size of images and ticks labels in arcsec
    if rcas < 2.5:
        szim       = 2.8
        ticklabels = np.arange(-2, 3, 1)
    else:
        szim       = 3.1
        ticklabels = np.arange(-3, 4, 1)
    
    # Converting ticks into MUSE pixel values
    xticks         = ticklabels/pix + xc
    yticks         = ticklabels/pix + yc
    
    # Defining lower and upper axes bounds in MUSE pixels (used for MUSE images)
    xmin           = xc - szim / pix
    xmax           = xc + szim / pix
    ymin           = yc - szim / pix
    ymax           = yc + szim / pix
    
    
    ##########################################
    #                Plotting                #
    ##########################################
    
    # Initializing figure ratio
    figx          = 7
    width_ratios  = [1,1,1]
    height_ratios = [1,1,1,0.075]
    figy          = figx * (np.sum(height_ratios) + 0.075) / np.sum(width_ratios)
    fig           = plt.figure(figsize=(figx, figy))  # sums of width in gridspec + intervals in subplots_adjust, eventually multiply by some factor
            
    plt.figtext(0.2, 1.01, name.replace('_', '-'),        fontsize=16)
    plt.figtext(0.4, 1.01, r'$z={:.5f}$'.format(z), fontsize=16)
    
    # Setting up the grid
    nlines        = 4
    ncols         = 3
    gs            = gridspec.GridSpec(nlines, ncols, width_ratios=width_ratios, height_ratios=height_ratios)
    
    ## HST values
    vmin0         = 3e1
    hstimmap     += vmin0
    hstmodmap    += vmin0
    hstresmap    += vmin0
    vmin          = 1.5e1
    vmax          = 1.5e2
    
    # Recover the MUSE centre position and PA endpoints but in HST pixel coordinates this time
    xh, yh        = hstimwcs.wcs_world2pix(ra0, dec0, 0)
    xch           = xh[0]
    ych           = yh[0]
    
    xmah1         = xh[1]
    xmah2         = xh[2]
    ymah1         = yh[1]
    ymah2         = yh[2]
    
    # Axes limits in HST pixels
    xminh         = xch - szim / pixh
    xmaxh         = xch + szim / pixh
    yminh         = ych - szim / pixh
    ymaxh         = ych + szim / pixh


    #############################################################################
    #                    HST plots (image, model, residuals)                    #
    #############################################################################
    
    axim, imim    = plot_hst(gs[0],             hstimmap,  xc, yc, pix, xminh, xmaxh, yminh, ymaxh, xch, xmah1, xmah2, ych, ymah1, ymah2, pixh, ticklabels, vmin, vmax, pa=True,  title=True, fluxcont=fluxcont)
    axmod, immod  = plot_hst(gs[0 + ncols],     hstmodmap, xc, yc, pix, xminh, xmaxh, yminh, ymaxh, xch, xmah1, xmah2, ych, ymah1, ymah2, pixh, ticklabels, vmin, vmax, pa=False, title=False)
    axres, imres  = plot_hst(gs[0 + 2 * ncols], hstresmap, xc, yc, pix, xminh, xmaxh, yminh, ymaxh, xch, xmah1, xmah2, ych, ymah1, ymah2, pixh, ticklabels, vmin, vmax, pa=False, title=False)
    
    # HST colorbar
    axcbhst       = plt.subplot(gs[0 + 3 * ncols])
    pos_axres     = axres.get_position()
    pos_axcbhst   = axcbhst.get_position()
    axcbhst.set_position([pos_axres.bounds[0], pos_axcbhst.bounds[1], pos_axres.width, pos_axcbhst.height])

    cb            = plt.colorbar(imres, orientation='horizontal', cax=axcbhst)
    cb.set_label(r'arbitrary (log)')
    cb.ax.set_xticklabels([], rotation=0)
    cb.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    
    
    ######################################################################
    #                              Flux map                              #
    ######################################################################
    
    if withFlux:
        # Setting axes limits
        axflux                = plt.subplot(gs[0 + 2 * ncols])
        fluxmap[fluxmap == 0] = np.nan
        vmin                  = np.nanmin(fluxmap)
        vmax                  = np.nanmax(fluxmap)
        axflux.set_xlim(xmin, xmax)
        axflux.set_ylim(ymin, ymax)
        
        imfluxmap             = axflux.imshow(fluxmap, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.gray_r, origin='lower', interpolation='nearest')
        
        # Setting ticks and ticks labels
        axflux.set_xticks(xticks)
        axflux.set_xticklabels(-ticklabels)
        axflux.set_yticks(yticks)
        axflux.set_yticklabels(ticklabels)
        
        ml                    = AutoMinorLocator(4)
        axflux.xaxis.set_minor_locator(ml)
        axflux.yaxis.set_minor_locator(ml)
        axflux.set_xlabel(r"$\Delta \alpha~('')$")
        
        axflux.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=True)
        axflux.xaxis.set_label_position("top")
        axflux.set_title(r'[O\textsc{ ii}] flux')
        
        # Flux colorbar
        axcbfl                = plt.subplot(gs[3 * ncols])
        pos_axfl              = axflux.get_position()
        pos_axcbfl            = axcbfl.get_position()
        axcbfl.set_position([pos_axfl.bounds[0], pos_axcbfl.bounds[1], pos_axfl.width, pos_axcbfl.height])
        cb                    = plt.colorbar(imfluxmap, orientation='horizontal', cax=axcbfl)
        cb.set_label(r'arbitrary (log)')
        cb.ax.set_xticklabels([], rotation=45)
        cb.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        
    
    #########################################################################
    #                                SNR map                                #
    #########################################################################
    
    if withSNR:
        axsnr = plt.subplot(gs[2], projection=snrwcs)
        
        vmin      = np.nanmin(snrmap)
        vmax      = np.nanmax(snrmap)
        axsnr.set_xlim(xmin, xmax)
        axsnr.set_ylim(ymin, ymax)
        
        # Logarithmic scale
        axsnr.imshow(snrmap, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.gray_r, origin='lower', interpolation='nearest')
        ra        = axsnr.coords['ra']
        dec       = axsnr.coords['dec']
        ra.set_ticklabel_visible( False)
        dec.set_ticklabel_visible(False)

    
    ########################################################################
    #                                VF map                                #
    ########################################################################
    
    axvf        = plt.subplot(gs[1])
    val         = np.max([np.abs(np.nanmax(vfmmap - vsys)), np.abs(np.nanmin(vfmmap - vsys))])
    
    if val < 32.5:
        val     = 32.5
    
    # Setting axes limits
    vmin        = -val
    vmax        = val
    axvf.set_xlim(xmin, xmax)
    axvf.set_ylim(ymin, ymax)
    
    # Plotting PSF on the bottom left corner
    print('PSF FWHM: ', psf)
    circle      = Circle(xy=(xmin + psf*1.4, ymin + psf*1.4), radius=psf, edgecolor='0.8', fc='0.8', lw=0.5, zorder=0)
    axvf.add_patch(circle)
    
    axvf.imshow(vfmap - vsys, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, origin='lower', interpolation='nearest')
    
    # Setting ticks and ticks labels
    axvf.set_xticks(xticks)
    axvf.set_xticklabels(-ticklabels)
    ml          = AutoMinorLocator(4)
    axvf.xaxis.set_minor_locator(ml)
    axvf.set_xlabel(r"$\Delta \alpha~('')$")
    
    axvf.set_yticks(yticks)
    axvf.set_yticklabels(ticklabels)
    axvf.yaxis.set_minor_locator(ml)
    
    axvf.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=True)
    axvf.xaxis.set_label_position("top")
    axvf.set_title(r'Velocity field')
    
    # Plotting flux contours in log scale
    cont_levels = np.logspace(np.log10(250), np.log10(8000), 6)
    axvf.contour(np.log10(fluxcont), levels=np.log10(cont_levels), colors='0.5')
    
    # Plotting centre and PA
    axvf.plot([xc], [yc], 'g+', mew=1)
    axvf.plot([xma1, xma2], [yma1, yma2], 'g')
    
    
    ##############################################################
    #                          VF model                          #
    ##############################################################
    
    axvfm       = plt.subplot(gs[1 + ncols])
    
    # Axes limits
    axvfm.set_xlim(xmin, xmax)
    axvfm.set_ylim(ymin, ymax)
    
    imvfm       = axvfm.imshow(vfmmap - vsys, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, origin='lower', interpolation='nearest')
    
    # Set ticks and ticks labels
    axvfm.set_xticks(xticks)
    axvfm.set_xticklabels(-ticklabels)
    axvfm.xaxis.set_minor_locator(ml)
    
    axvfm.set_yticks(yticks)
    axvfm.set_yticklabels(ticklabels)
    axvfm.yaxis.set_minor_locator(ml)
    
    axvfm.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    
    
    ######################################################################
    #                            VF residuals                            #
    ######################################################################
    
    axvfr       = plt.subplot(gs[1 + 2 * ncols])
    
    # Axes limits
    axvfr.set_xlim(xmin, xmax)
    axvfr.set_ylim(ymin, ymax)
    
    axvfr.imshow(vfrmap, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, origin='lower', interpolation='nearest')
    
    # Set ticks and ticks labels
    axvfr.set_xticks(xticks)
    axvfr.set_xticklabels(-ticklabels)
    axvfr.xaxis.set_minor_locator(ml)
    
    axvfr.set_yticks(yticks)
    axvfr.set_yticklabels(ticklabels)
    axvfr.yaxis.set_minor_locator(ml)
    
    axvfr.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    
    
    #######################################################
    #                     VF colorbar                     #
    #######################################################
    
    axcbvf      = plt.subplot(gs[1 + 3 * ncols])
    
    pos_axvfr    = axvfr.get_position()
    pos_axcbvf   = axcbvf.get_position()
    axcbvf.set_position([pos_axvfr.bounds[0], pos_axcbvf.bounds[1], pos_axvfr.width, pos_axcbvf.height])
    
    # Setting colorbar ticks
    stepvf       = np.floor((vmax)/2/5) * 5
    while (vmax >= 40) & ((2 * stepvf) > (0.8 * vmax)):
        stepvf  -= 5
    ticksvf      = np.arange(-4 * stepvf, vmax + 1, stepvf)
    
    cb           = plt.colorbar(imvfm, orientation='horizontal', cax=axcbvf, ticks=ticksvf)
    cb.set_label(r'km s$^{-1}$')
    cb.ax.set_xticklabels(np.int64(ticksvf), rotation=0)


    #################################################################
    #                      Velocity dispersion                      #
    #################################################################
    
    axsig        = plt.subplot(gs[2])
    
    # Only keep velocity dispersion residuals for pixels with a dispersion larger than the lsf FWHM
    vmin         = 0
    cond         = np.logical_not(np.isnan(np.sqrt(sigmap**2 - lsf**2)))
    ss           = np.sqrt(sigmap**2 - lsf**2)[cond]
    ss.sort()
    
    # Define maximum value
    try:
        vmax     = ss[-3]
    except:
        vmax     = 50
    
    if vmax <= 63.5:
        vmax     = 63.5
    if (vmax >= 75) & (vmax < 79):
        vmax     = 70
    if (vmax >= 100) & (vmax < 106):
        vmax     = 106
    if (vmax >= 125) & (vmax < 132):
        vmax     = 132
    
    # Axes limits
    axsig.set_xlim(xmin, xmax)
    axsig.set_ylim(ymin, ymax)
    
    axsig.imshow(np.sqrt(sigmap**2 - lsf**2), vmin=vmin, vmax=vmax, cmap=plt.cm.CMRmap, origin='lower', interpolation='nearest')
    
    # Setting ticks and ticks labels
    axsig.set_xticks(xticks)
    axsig.set_xticklabels(-ticklabels)
    axsig.xaxis.set_minor_locator(ml)
    axsig.set_xlabel(r"$\Delta \alpha~('')$")
    
    axsig.set_yticks(yticks)
    axsig.set_yticklabels(ticklabels)
    axsig.yaxis.set_minor_locator(ml)
    
    axsig.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=True)
    axsig.xaxis.set_label_position("top")
    axsig.yaxis.set_label_position("right")
    axsig.set_title(r'Velocity dispersion')
    
    # Plotting centre and PA
    axsig.plot([xc], [yc], 'g+', mew=1)
    axsig.plot([xma1, xma2], [yma1, yma2], 'g')
    
    # Flux contours in log scale
    cont_levels  = np.logspace(np.log10(250), np.log10(8000), 6)
    axsig.contour(np.log10(fluxcont), levels=np.log10(cont_levels), colors='0.5')


    #########################################################################
    #                       Velocity dispersion model                       #
    #########################################################################
    
    axsigm       = plt.subplot(gs[2 + 1 * ncols])
    
    # Axes limits
    axsigm.set_xlim(xmin, xmax)
    axsigm.set_ylim(ymin, ymax)
    
    imsigm       = axsigm.imshow(np.sqrt(sigmmap**2), vmin=vmin, vmax=vmax, cmap=plt.cm.CMRmap, origin='lower', interpolation='nearest')
    
    # Ticks and ticks labels
    axsigm.set_xticks(xticks)
    axsigm.set_xticklabels(-ticklabels)
    axsigm.xaxis.set_minor_locator(ml)
    
    axsigm.set_yticks(yticks)
    axsigm.set_yticklabels(ticklabels)
    axsigm.yaxis.set_minor_locator(ml)
    
    axsigm.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    axsigm.yaxis.set_label_position("right")
    
    
    #########################################################################
    #                     Velocity dispersion residuals                     #
    #########################################################################
    
    axsigr       = plt.subplot(gs[2 + 2 * ncols])
    
    # Axes limits
    axsigr.set_xlim(xmin, xmax)
    axsigr.set_ylim(ymin, ymax)
    
    axsigr.imshow(sigrmap, vmin=vmin, vmax=vmax, cmap=plt.cm.CMRmap, origin='lower', interpolation='nearest')
    
    # Ticks and ticks labels
    axsigr.set_xticks(xticks)
    axsigr.set_xticklabels(-ticklabels)
    axsigr.xaxis.set_minor_locator(ml)
    
    axsigr.set_yticks(yticks)
    axsigr.set_yticklabels(ticklabels)
    axsigr.yaxis.set_minor_locator(ml)
    
    axsigr.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    axsigr.yaxis.set_label_position("right")
    
    
    ################################################################
    #                        Sigma colorbar                        #
    ################################################################
    
    axcbsig      = plt.subplot(gs[2 + 3 * ncols])
    
    pos_axsigr   = axsigr.get_position()
    pos_axcbsig  = axcbsig.get_position()
    axcbsig.set_position([pos_axsigr.bounds[0], pos_axcbsig.bounds[1], pos_axsigr.width, pos_axcbsig.height])
    
    # Defining the number of ticks on the colorbar
    stepsig      = np.floor((vmax)/4/5) * 5
    while ((4 * stepsig) > (0.95 * vmax)):
        stepsig -= 5
        
    tickssig     = np.arange(0, vmax + 1, stepsig)
    cb           = plt.colorbar(imsigm, orientation='horizontal', cax=axcbsig, ticks=tickssig)
    cb.set_label(r'km s$^{-1}$')
    cb.ax.set_xticklabels(np.int64(tickssig),rotation=0)
    
    # Spacing left+right vs top+bottom must be identical to keep good aspect, otherwize change figure size
    fig.subplots_adjust(hspace=0.05, wspace=0.05, left=0.18, right=0.98, top=0.9, bottom=0.1)
    
    # Saving figure
    plt.show()
    #fig.savefig(pathout + grid + '-' + num + '.pdf', bbox_inches='tight')
    plt.close()