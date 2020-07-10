#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: B. Epinat - LAM
modified by W. Mercier - IRAP

Create recap morpho-kinematics plots for MUSE \& HST modelling.
'''

import glob
import copy
import numpy                 as np
import astropy.wcs           as wcs
import astropy.io.fits       as fits
import astropy.io.ascii      as asci
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
    axhstmap = plt.subplot(gs)
    axhstmap.set_xlim(xminh, xmaxh)
    axhstmap.set_ylim(yminh, ymaxh)
    
    # Clipping and plotting (logarithmic scale)
    hstmap[hstmap < vmin] = vmin
    imfluxmap             = axhstmap.imshow(hstmap, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.gray_r, origin='lower', interpolation='nearest')
    
    ##################################################################
    #                  Setting x and y ticks labels                  #
    ##################################################################
    
    # Converting ticks values from arcsec to HST pixel and centre on HST centre
    xticks  = ticklabels/pixh + xch
    axhstmap.set_xticks(xticks)
    axhstmap.set_xticklabels(-ticklabels)
    
     yticks = ticklabels/pixh + ych
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
        ctr         = axhstmap.contour(np.log10(fluxcont), levels=np.log10(cont_levels), cmap='brg', transform=trans_data)

    return axhstmap, imfluxmap


def paper_map(hst, flux, snr, vf, vfm, vfr, sig, sigm, sigr, name, z, xc, yc, vsys, pa, r22, rc, lsf, psf, pathout, deltapa=-42, offset=[0,0], HST_smooth_sigma=3.0, MUSE_smooth_sigma=1.0):
    '''
    Create the morpho-kinematics recap maps.
    
    Mandatory parameters
    --------------------
        flux : list of str
            list of MUSE flux maps file names
        hst : str
            file name of the GALFIT output fits file containing the image as the 1st extension, the model as the 2nd extension and the residuals as the 3rd
        offset : list of two int/float
            offsets to subtract from the x and y reference pixel coordinates
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
            velocity field residuals file name
            
    Optional parameters
    -------------------
        HST_smooth_sigma : float
            smoothing value (Gaussian std) used for Gaussian smoothing the HST image. Default is 3.0 HST pixels.
        MUSE_smooth_sigma : float
            smoothing value (Gaussian std) used for Gaussian smoothing the MUSE flux data. Default is 1.0 MUSE pixel.
    '''
    
    # HST data
    with fits.open(hst) as hsthdu:
        hstimhdu  = hsthdu[1]
        hstmodhdu = hsthdu[2]
        hstreshdu = hsthdu[3]
    
    hstimmap      = hstimhdu.data
    hstimhdr      = hstimhdu.header
    
    # Get and apply x and y offsets
    offstr        = hstimhdr['OBJECT']
    boundaries    = offstr.replace('[', ' ').replace(',', ' ').replace(']', ' ').replace(':', ' ').split()
    offx          = np.int(boundaries[0]) - 1
    offy          = np.int(boundaries[2]) - 1
    
    hstimhdr['CRPIX1'] += offset[1] - offx
    hstimhdr['CRPIX2'] += offset[0] - offy
    
    hstmodmap     = hstmodhdu.data
    hstmodhdr     = copy.deepcopy(hstimhdr)
    
    hstresmap     = hstreshdu.data
    hstreshdr     = copy.deepcopy(hstimhdr)
    
    # Opening and combining MUSE flux maps
    if isinstance(flux, str):
        flux      = [flux]
    
    with fits.open(flux[0]) as fluxhdu:
        fluxmap   = fluxhdu[0].data * 0
        fluxhdr   = fluxhdu[0].header
        
    for file in flux:
        with fits.open(file) as temphdu:
            fluxmap += temphdu[0].data
            
    pixarea       = (fluxhdr['CD1_1'] * 3600) ** 2  # arcsec**2 
    fluxmap      /= pixarea
    
    # SNR file
    with fits.open(snr) as snrhdu:
        snrmap    = snrhdu[0].data
        snrhdr    = snrhdu[0].header
        
    # Velocity field (CAMEL map)
    with fits.open(vf) as vfhdu:
        vfmap     = vfhdu[0].data
        vfhdr     = vfhdu[0].header
    
    # Velocity field model
    with fits.open(vfm) as vfmhdu:
        vfmmap    = vfmhdu[0].data
        vfmhdr    = vfmhdu[0].header
    
    # Velocity field residuals
    with fits.open(vfr) as vfrhdu:
        vfrmap    = vfrhdu[0].data
        vfrhdr    = vfrhdu[0].header
    
    # Velocity dispersion map (CAMEL)
    with fits.open(sig) as sighdu:
        sigmap = sighdu[0].data
        sighdr = sighdu[0].header

    # Velocity dispersion map model    
    with fits.open(sigm) as sigmhdu:
        sigmmap = sigmhdu[0].data
        sigmhdr = sigmhdu[0].header

    # Velocity dispersion map residuals  
    with fits.open(sigr) as sigrhdu:
        sigrmap = sigrhdu[0].data
        sigrhdr = sigrhdu[0].header
    
    # Defining wcs
    fluxhdr['WCSAXES']   = 2
    vfhdr['WCSAXES']     = 2
    sighdr['WCSAXES']    = 2
    snrhdr['WCSAXES']    = 2
    hstimhdr['WCSAXES']  = 2
    hstmodhdr['WCSAXES'] = 2
    hstreshdr['WCSAXES'] = 2
    fluxwcs              = wcs.WCS(fluxhdr)
    vfwcs                = wcs.WCS(vfhdr)
    sigwcs               = wcs.WCS(sighdr)
    snrwcs               = wcs.WCS(snrhdr)
    
    hstimwcs             = wcs.WCS(hstimhdr)
    facim                = 1
    
    hstmodwcs            = wcs.WCS(hstmodhdr)
    facmod               = 1
    
    hstreswcs            = wcs.WCS(hstreshdr)
    facres               = 1
    
    # Get MUSE and HST pixel scales in arcsec as the geometric mean of the x and y pixel scales
    pix                  = np.sqrt(fluxhdr['CD1_1']**2 + fluxhdr['CD1_2']**2) * 3600
    pixh                 = np.sqrt(hstimhdr['CD1_1']**2 + hstimhdr['CD1_2']**2) * 3600
    
    # Smoothing data
    data                 = gaussian_filter(hstimmap, HST_smooth_sigma)
    fluxcont             = gaussian_filter(fluxmap, MUSE_smooth_sigma)
    
    # positions of center and major axis
    
    r22m = r22 * pixh / pix  # in Muse pixels
    rcas = rc * pix
    
    xma1 = xc + r22m * np.sin(np.radians(pa+deltapa))
    yma1 = yc - r22m * np.cos(np.radians(pa+deltapa))
    xma2 = xc - r22m * np.sin(np.radians(pa+deltapa))
    yma2 = yc + r22m * np.cos(np.radians(pa+deltapa))
    
    #print(xma1, xma2, yma1, yma2, pa)
    
    ra0, dec0 = fluxwcs.wcs_pix2world([xc, xma1, xma2], [yc, yma1, yma2], 0)
    
    # Def of size of images
    #if rcas < 2.0:
        #szim = 2.3  # arcsec
        #ticklabels = np.arange(-2, 3, 1)
    if rcas < 2.5:
        szim = 2.8  # arcsec
        ticklabels = np.arange(-2, 3, 1)
    else:
        szim = 3.3  # arcsec
        ticklabels = np.arange(-3, 4, 1)
    #else:
        #szim = 3.8  # arcsec
        #ticklabels = np.arange(-3, 4, 1)
    
    #ticklabels = np.arange(-2, 3, 1)
    
    xticks = ticklabels/pix + xc
    yticks = ticklabels/pix + yc
    
    xmin = xc - szim / pix
    xmax = xc + szim / pix
    
    ymin = yc - szim / pix
    ymax = yc + szim / pix
    
    # Initializing figure
    figx = 7
    width_ratios = [1,1,1]
    height_ratios = [1,1,1,0.075]
    figy = figx * (np.sum(height_ratios) + 0.075) / np.sum(width_ratios)
    fig = plt.figure(figsize=(figx, figy))  # sums of width in gridspec + intervals in subplots_adjust, eventually multiply by some factor
    
    num = name.split('_')[-2]
    grid = name.split('_')[0].split('-')[0]
    if 'B' in name:
        grid += 'b'
    #print(name, num, grid)
            
    #plt.figtext(0.2, 0.25, 'ID ' + name.split('_o2')[0].split('_')[-1])
    plt.figtext(0.2, 1.01, grid + '-' + num, fontsize=16)
    plt.figtext(0.4, 1.01, r'$z={:.5f}$'.format(z), fontsize=16)
    
    nlines = 4
    ncols = 3
    gs = gridspec.GridSpec(nlines, ncols, width_ratios=width_ratios, height_ratios=height_ratios)
    
    ## Valeurs HST
    print('std res: ', np.std(hstresmap))
    # std ~ 30
    vmin0 = 3e1
    hstimmap += vmin0
    hstmodmap += vmin0
    hstresmap += vmin0
    vmin = 1.5e1
    vmax = 1.5e3
    
    xh, yh = hstimwcs.wcs_world2pix(ra0, dec0, 0)
    
    xch = xh[0]
    ych = yh[0]
    
    xmah1 = xh[1]
    xmah2 = xh[2]
    ymah1 = yh[1]
    ymah2 = yh[2]
    
    xminh = xch - szim / pixh
    xmaxh = xch + szim / pixh
    
    yminh = ych - szim / pixh
    ymaxh = ych + szim / pixh

    axim, imim = plot_hst(gs[0], hstimmap, xc, yc, pix, xminh, xmaxh, yminh, ymaxh, xch, xmah1, xmah2, ych, ymah1, ymah2, pixh, ticklabels, vmin, vmax, pa=True, title=True, fluxcont=fluxcont)
    axmod, immod = plot_hst(gs[0 + ncols], hstmodmap, xc, yc, pix, xminh, xmaxh, yminh, ymaxh, xch, xmah1, xmah2, ych, ymah1, ymah2, pixh, ticklabels, vmin, vmax, pa=False, title=False)
    axres, imres = plot_hst(gs[0 + 2 * ncols], hstresmap, xc, yc, pix, xminh, xmaxh, yminh, ymaxh, xch, xmah1, xmah2, ych, ymah1, ymah2, pixh, ticklabels, vmin, vmax, pa=False, title=False)
    
    #-------------
    # HST colorbar
    #-------------
    axcbhst = plt.subplot(gs[0 + 3 * ncols])
    
    pos_axres = axres.get_position()
    pos_axcbhst = axcbhst.get_position()
    axcbhst.set_position([pos_axres.bounds[0], pos_axcbhst.bounds[1], pos_axres.width, pos_axcbhst.height])
    
    #stepvf = np.ceil(vmax/3/10) * 10
    #stepvf = np.ceil(vmax/3/15) * 15
    #ticksvf = np.arange(-4 * stepvf,vmax + 1, stepvf)
    
    cb = plt.colorbar(imres, orientation='horizontal', cax=axcbhst)
    cb.set_label(r'arbitrary (log)')
    #cb.ax.set_xticklabels(np.int64(ticksvf),rotation=45)
    #print(cb.ax.get_xticklabels()[:])
    #cb.ax.set_xticklabels([], rotation=45)
    cb.ax.set_xticklabels([], rotation=0)
    
    cb.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    
    ##-------------
    ## Flux map
    ##-------------
    #axflux = plt.subplot(gs[0 + 2 * ncols])
    
    #fluxmap[fluxmap == 0] = np.nan
    #vmin = np.nanmin(fluxmap)
    #vmax = np.nanmax(fluxmap)
    
    #axflux.set_xlim(xmin, xmax)
    #axflux.set_ylim(ymin, ymax)
    
    #imfluxmap = axflux.imshow(fluxmap, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.gray_r, origin='lower', interpolation='nearest')
    
    #axflux.set_xticks(xticks)
    #axflux.set_xticklabels(-ticklabels)
    #ml = AutoMinorLocator(4)
    #axflux.xaxis.set_minor_locator(ml)
    #axflux.set_xlabel(r"$\Delta \alpha~('')$")
    ##axfluxt.set_xlabel(r"$\Delta \alpha~('')$")
    
    #axflux.set_yticks(yticks)
    #axflux.set_yticklabels(ticklabels)
    #ml = AutoMinorLocator(4)
    #axflux.yaxis.set_minor_locator(ml)
    ##axflux.set_ylabel(r"$\Delta \delta~('')$")
    
    #axflux.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=True)
    #axflux.xaxis.set_label_position("top")
    ##axflux.yaxis.set_label_position("right")
    #axflux.set_title(r'[O\textsc{ ii}] flux')
    
    ##plt.rcParams['xtick.labelbottom'] = True
    
    ##-------------
    ## Flux colorbar
    ##-------------
    #axcbfl = plt.subplot(gs[3 * ncols])
    
    #pos_axfl = axflux.get_position()
    #pos_axcbfl = axcbfl.get_position()
    #axcbfl.set_position([pos_axfl.bounds[0], pos_axcbfl.bounds[1], pos_axfl.width, pos_axcbfl.height])
    #print(pos_axcbfl.height, pos_axcbfl.bounds[1])
    
    #cb = plt.colorbar(imfluxmap, orientation='horizontal', cax=axcbfl)
    #cb.set_label(r'arbitrary (log)')
    ##cb.ax.set_xticklabels(np.int64(ticksvf),rotation=45)
    ##print(cb.ax.get_xticklabels()[:])
    #cb.ax.set_xticklabels([], rotation=45)
    
    #cb.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    
    ##-------------
    ## SNR map
    ##-------------
    
    #axsnr = plt.subplot(gs[2], projection=snrwcs)
    ##axsnr = fig.add_subplot(3, 5, 3, projection=snrwcs)
    
    #vmin = np.nanmin(snrmap)
    #vmax = np.nanmax(snrmap)
    
    #axsnr.set_xlim(xmin, xmax)
    #axsnr.set_ylim(ymin, ymax)
    
    #imsnrmap = axsnr.imshow(snrmap, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.gray_r, origin='lower', interpolation='nearest')  # Logarithmic scale
    #ra = axsnr.coords['ra']
    #dec = axsnr.coords['dec']
    ##dec.set_ticklabel(rotation=90)
    #ra.set_ticklabel_visible(False)
    #dec.set_ticklabel_visible(False)
    ##dec.set_axislabel('Declination (J2000)')
    ##ra.set_axislabel('Right ascention (J2000)', minpad=0.4)
    
    ##cb = plt.colorbar(imsnrmap, orientation='horizontal', fraction=0.04, pad=0.)
    ##cbarC.set_label(---)

    
    #-------------
    # VF map
    #-------------
    axvf = plt.subplot(gs[1])
    val = np.max([np.abs(np.nanmax(vfmmap - vsys)), np.abs(np.nanmin(vfmmap - vsys))])
    
    if val < 32.5:
        val = 32.5
    
    vmin = -val
    vmax = val
    
    axvf.set_xlim(xmin, xmax)
    axvf.set_ylim(ymin, ymax)
    
    imvf = axvf.imshow(vfmap - vsys, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, origin='lower', interpolation='nearest')
    
    axvf.set_xticks(xticks)
    axvf.set_xticklabels(-ticklabels)
    ml = AutoMinorLocator(4)
    axvf.xaxis.set_minor_locator(ml)
    axvf.set_xlabel(r"$\Delta \alpha~('')$")
    
    axvf.set_yticks(yticks)
    axvf.set_yticklabels(ticklabels)
    ml = AutoMinorLocator(4)
    axvf.yaxis.set_minor_locator(ml)
    #axvf.set_ylabel(r"$\Delta \delta~('')$")
    
    axvf.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=True)
    axvf.xaxis.set_label_position("top")
    #axvf.yaxis.set_label_position("right")
    axvf.set_title(r'Velocity field')
    
    #trans = axvf.get_transform(hstreswcs)
    #axvf.contour(data, levels=np.logspace(-3, 0., 7), colors='k', transform=trans)
    #cont_levels = np.logspace(np.log10(300), np.log10(8000), 5)
    cont_levels = np.logspace(np.log10(250), np.log10(8000), 6)
    #ctr = axvf.contour(np.log10(fluxcont), levels=np.log10(cont_levels), cmap='brg')
    ctr = axvf.contour(np.log10(fluxcont), levels=np.log10(cont_levels), colors='0.5')
    
    axvf.plot([xc], [yc], 'g+', mew=1)
    axvf.plot([xma1, xma2], [yma1, yma2], 'g')
    
    print('PSF FWHM: ',  psf)
    circle = Circle(xy=(xmin + psf*1.4, ymin + psf*1.4), radius=psf, edgecolor='0.8', fc='0.8', lw=0.5)
    axvf.add_patch(circle)
    
    #-------------
    # VF model
    #-------------
    axvfm = plt.subplot(gs[1 + ncols])
    
    axvfm.set_xlim(xmin, xmax)
    axvfm.set_ylim(ymin, ymax)
    
    imvfm = axvfm.imshow(vfmmap - vsys, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, origin='lower', interpolation='nearest')
    
    axvfm.set_xticks(xticks)
    axvfm.set_xticklabels(-ticklabels)
    ml = AutoMinorLocator(4)
    axvfm.xaxis.set_minor_locator(ml)
    #axvfm.set_xlabel(r"$\Delta \alpha~('')$")
    
    axvfm.set_yticks(yticks)
    axvfm.set_yticklabels(ticklabels)
    ml = AutoMinorLocator(4)
    axvfm.yaxis.set_minor_locator(ml)
    #axvfm.set_ylabel(r"$\Delta \delta~('')$")
    
    axvfm.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    
    #-------------
    # VF residuals
    #-------------
    axvfr = plt.subplot(gs[1 + 2 * ncols])
    #axvfr = plt.subplot(gs[2 + 2 * ncols], projection=vfwcs)
    #axvfr = fig.add_subplot(3, 5, 14, projection=vfwcs)
    
    axvfr.set_xlim(xmin, xmax)
    axvfr.set_ylim(ymin, ymax)
    
    imvfr = axvfr.imshow(vfrmap, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, origin='lower', interpolation='nearest')
    
    axvfr.set_xticks(xticks)
    axvfr.set_xticklabels(-ticklabels)
    ml = AutoMinorLocator(4)
    axvfr.xaxis.set_minor_locator(ml)
    #axvfr.set_xlabel(r"$\Delta \alpha~('')$")
    
    axvfr.set_yticks(yticks)
    axvfr.set_yticklabels(ticklabels)
    ml = AutoMinorLocator(4)
    axvfr.yaxis.set_minor_locator(ml)
    #axvfr.set_ylabel(r"$\Delta \delta~('')$")
    
    axvfr.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    
    #-------------
    # VF colorbar
    #-------------
    axcbvf = plt.subplot(gs[1 + 3 * ncols])
    
    pos_axvfr = axvfr.get_position()
    pos_axcbvf = axcbvf.get_position()
    axcbvf.set_position([pos_axvfr.bounds[0], pos_axcbvf.bounds[1], pos_axvfr.width, pos_axcbvf.height])
    
    #stepvf = np.ceil(vmax/3/10) * 10
    #stepvf = np.ceil(vmax/3/15) * 15
    stepvf = np.floor((vmax)/2/5) * 5
    while (vmax >= 40) & ((2 * stepvf) > (0.8 * vmax)):
        stepvf -= 5
    #if vmax >= 30:
        #stepvf = np.floor((vmax-20)/2/5) * 5
    #else:
        #stepvf = 10
    #elif vmax > 15:
        #stepvf = 10
    #else:
        #stepvf = 5
    print(vmax, stepvf)
    ticksvf = np.arange(-4 * stepvf, vmax + 1, stepvf)
    
    cb = plt.colorbar(imvfm, orientation='horizontal', cax=axcbvf, ticks=ticksvf)
    cb.set_label(r'km s$^{-1}$')
    #cb.ax.set_xticklabels(np.int64(ticksvf),rotation=45)
    cb.ax.set_xticklabels(np.int64(ticksvf),rotation=0)
    #print(cb.ax.get_xticklabels()[:])
    #cb.ax.set_xticklabels(cb.ax.get_xticklabels()[:], rotation=45)

    #-------------
    # Velocity dispersion
    #-------------
    axsig = plt.subplot(gs[2])
    
    vmin = 0
    cond = np.logical_not(np.isnan(np.sqrt(sigmap**2 - lsf**2)))
    ss = np.sqrt(sigmap**2 - lsf**2)[cond]
    ss.sort()
    try:
        vmax = ss[-3]
    except:
        vmax = 50
    #print(vmax)
    if vmax <= 63.5:
        vmax = 63.5
    if (vmax >= 75) & (vmax < 79):
        vmax = 70
    if (vmax >= 100) & (vmax < 106):
        vmax = 106
    if (vmax >= 125) & (vmax < 132):
        vmax = 132
    
    axsig.set_xlim(xmin, xmax)
    axsig.set_ylim(ymin, ymax)
    
    imsig = axsig.imshow(np.sqrt(sigmap**2 - lsf**2), vmin=vmin, vmax=vmax, cmap=plt.cm.CMRmap, origin='lower', interpolation='nearest')
    
    axsig.set_xticks(xticks)
    axsig.set_xticklabels(-ticklabels)
    ml = AutoMinorLocator(4)
    axsig.xaxis.set_minor_locator(ml)
    axsig.set_xlabel(r"$\Delta \alpha~('')$")
    
    axsig.set_yticks(yticks)
    axsig.set_yticklabels(ticklabels)
    ml = AutoMinorLocator(4)
    axsig.yaxis.set_minor_locator(ml)
    #axsig.set_ylabel(r"$\Delta \delta~('')$")
    
    axsig.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=True)
    axsig.xaxis.set_label_position("top")
    axsig.yaxis.set_label_position("right")
    axsig.set_title(r'Velocity dispersion')
    
    axsig.plot([xc], [yc], 'g+', mew=1)
    axsig.plot([xma1, xma2], [yma1, yma2], 'g')
    
    #cont_levels = np.logspace(np.log10(300), np.log10(8000), 5)
    cont_levels = np.logspace(np.log10(250), np.log10(8000), 6)
    #ctr = axvf.contour(np.log10(fluxcont), levels=np.log10(cont_levels), cmap='brg')
    ctr = axsig.contour(np.log10(fluxcont), levels=np.log10(cont_levels), colors='0.5')

    #-------------
    # Velocity dispersion model
    #-------------
    axsigm = plt.subplot(gs[2 + 1 * ncols])
    
    axsigm.set_xlim(xmin, xmax)
    axsigm.set_ylim(ymin, ymax)
    
    imsigm = axsigm.imshow(np.sqrt(sigmmap**2), vmin=vmin, vmax=vmax, cmap=plt.cm.CMRmap, origin='lower', interpolation='nearest')
    
    axsigm.set_xticks(xticks)
    axsigm.set_xticklabels(-ticklabels)
    ml = AutoMinorLocator(4)
    axsigm.xaxis.set_minor_locator(ml)
    #axsigm.set_xlabel(r"$\Delta \alpha~('')$")
    
    axsigm.set_yticks(yticks)
    axsigm.set_yticklabels(ticklabels)
    ml = AutoMinorLocator(4)
    axsigm.yaxis.set_minor_locator(ml)
    #axsigm.set_ylabel(r"$\Delta \delta~('')$")
    
    axsigm.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    #axsigm.xaxis.set_label_position("top")
    axsigm.yaxis.set_label_position("right")
    
    #-------------
    # Velocity dispersion residuals
    #-------------
    axsigr = plt.subplot(gs[2 + 2 * ncols])
    
    axsigr.set_xlim(xmin, xmax)
    axsigr.set_ylim(ymin, ymax)
    
    imsigr = axsigr.imshow(sigrmap, vmin=vmin, vmax=vmax, cmap=plt.cm.CMRmap, origin='lower', interpolation='nearest')
    
    axsigr.set_xticks(xticks)
    axsigr.set_xticklabels(-ticklabels)
    ml = AutoMinorLocator(4)
    axsigr.xaxis.set_minor_locator(ml)
    #axsigr.set_xlabel(r"$\Delta \alpha~('')$")
    
    axsigr.set_yticks(yticks)
    axsigr.set_yticklabels(ticklabels)
    ml = AutoMinorLocator(4)
    axsigr.yaxis.set_minor_locator(ml)
    #axsigr.set_ylabel(r"$\Delta \delta~('')$")
    
    axsigr.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    #axsigr.xaxis.set_label_position("top")
    axsigr.yaxis.set_label_position("right")
    
    #-------------
    # Sigma colorbar
    #-------------
    axcbsig = plt.subplot(gs[2 + 3 * ncols])
    
    pos_axsigr = axsigr.get_position()
    pos_axcbsig = axcbsig.get_position()
    axcbsig.set_position([pos_axsigr.bounds[0], pos_axcbsig.bounds[1], pos_axsigr.width, pos_axcbsig.height])
    
    stepsig = np.floor((vmax)/4/5) * 5
    while ((4 * stepsig) > (0.95 * vmax)):
        stepsig -= 5
    #if vmax >= 20:
        #stepsig = np.floor((vmax-5)/4/5) * 5
    #else:
        #stepsig = 5
    tickssig = np.arange(0,vmax + 1, stepsig)
    cb = plt.colorbar(imsigm, orientation='horizontal', cax=axcbsig, ticks=tickssig)
    cb.set_label(r'km s$^{-1}$')
    #cb.ax.set_xticklabels(np.int64(tickssig),rotation=45)
    cb.ax.set_xticklabels(np.int64(tickssig),rotation=0)
    
    #cb = plt.colorbar(imsigm, orientation='horizontal', pad=0.)
    
    fig.subplots_adjust(hspace=0.05, wspace=0.05, left=0.18, right=0.98, top=0.9, bottom=0.1)  # spacing left+right vs top+bottom must be identical to keep good aspect, otherwize chanfe figure size
    
    # Additional map?
    
    #ax32 = fig.add_subplot(3, 5, 6, projection=fluxwcs)
    fig.savefig(pathout + grid + '-' + num + '.pdf', bbox_inches='tight')
    #fig.savefig(pathout + name.split('_Z_')[0] + '.pdf', bbox_inches='tight')
    plt.close()


def main():
    '''
    '''
    #pathhst = '/home/bepinat/Instruments/MUSE/analyse/UDF/data/hst/'
    ##hst = pathhst + 'hlsp_xdf_hst_acswfc-30mas_hudf_f814w_v1_sci.fits'
    ##hst = pathhst + 'hlsp_hudf12_hst_wfc3ir_udfmain_f160w_v1.0_drz.fits'
    
    #hst = {'f105w0':pathhst + 'hlsp_hudf12_hst_wfc3ir_udfmain_f105w_v1.0_drz.fits',
           #'f125w0':pathhst + 'hlsp_hudf12_hst_wfc3ir_udfmain_f125w_v1.0_drz.fits',
           #'f140w0':pathhst + 'hlsp_hudf12_hst_wfc3ir_udfmain_f140w_v1.0_drz.fits',
           #'f160w0':pathhst + 'hlsp_hudf12_hst_wfc3ir_udfmain_f160w_v1.0_drz.fits',
           #'f105w':pathhst + 'hlsp_xdf_hst_wfc3ir-60mas_hudf_f105w_v1_sci.fits',
           #'f125w':pathhst + 'hlsp_xdf_hst_wfc3ir-60mas_hudf_f125w_v1_sci.fits',
           #'f140w':pathhst + 'hlsp_xdf_hst_wfc3ir-60mas_hudf_f140w_v1_sci.fits',
           #'f160w':pathhst + 'hlsp_xdf_hst_wfc3ir-60mas_hudf_f160w_v1_sci.fits',
           #'f435w':pathhst + 'hlsp_xdf_hst_acswfc-30mas_hudf_f435w_v1_sci.fits',
           #'f606w':pathhst + 'hlsp_xdf_hst_acswfc-30mas_hudf_f606w_v1_sci.fits',
           #'f775w':pathhst + 'hlsp_xdf_hst_acswfc-30mas_hudf_f775w_v1_sci.fits',
           #'f814w':pathhst + 'hlsp_xdf_hst_acswfc-30mas_hudf_f814w_v1_sci.fits',
           #'f850lp':pathhst + 'hlsp_xdf_hst_acswfc-30mas_hudf_f850lp_v1_sci.fits'}
    
    #path = '/home/bepinat/Instruments/MUSE/analyse/UDF/data/camel/udf10/'
    ##name = 'udf_mos_c042_e030_1_o2_Z_0.621992'
    
    #file_recap = path + line + '/recap_kinematics_parameters_2_slp_xyi_mclean5.0.txt'
    #file_input = path + line + '/input_fit_o2_v2.txt'
    #file_input = path + line + '/input_fit_o2_v3.txt'
    #pathout = '/home/bepinat/Instruments/MUSE/analyse/UDF/data/pdf/udf10/'
    #deltapa = 0

    # XXX I had to rerun kinemorpho_catalogs.py to create the correct recaps
    # Groups
    path = '/home/bepinat/Bureau/valentina/v08/analyse_be/fig_maps/Data/'
    pathhst = '/home/bepinat/Bureau/valentina/v08/analyse_be/fig_maps/Data/morphology/'
    
    groups = {'CGr32-M1_deep':'HST_CGr32-M1',
              'CGr32-M2_deep':'HST_CGr32-M2',
              'CGr32-M3_deep':'HST_CGr32-M3',
              'CGr79_deep':'HST_CGr79',
              'CGr30_deep':'HST_CGr30/CGr30_z0.7',
              'CGr114_snshot':'HST_CGr114/CGr114_z0.65',
              'CGr34_mid_deep':'HST_CGr34/CGr34_z0.7',
              'CGr28':'HST_CGr28',  # XXX il a fallu que je modifie la première ligne de input_fit_o2
              'CGr84-NorthA':'HST_CGr84-North/CGr84-NorthA',  # XXX il a fallu que je modifie la première ligne de input_fit_o2
              'CGr84-NorthB':'HST_CGr84-North/CGr84-NorthB',  # XXX il a fallu que je modifie la première ligne de input_fit_o2
              'CGr84_mdeep_A':'HST_CGr84/CGr84_z0.68',
              'CGr84_mdeep_B':'HST_CGr84/CGr84_z0.7',
              }
    
    groups_offsets = {'CGr32-M1_deep':[11, -6],
                      'CGr32-M2_deep':[0, 0],
                      'CGr32-M3_deep':[0, 0],
                      'CGr34_mid_deep':[0, 0],
                      'CGr79_deep':[0, 0],
                      'CGr114_snshot':[0, 0],
                      'CGr28':[0, 0],
                      'CGr30_deep':[0, 0],
                      'CGr84-NorthA':[0, 0],
                      'CGr84-NorthB':[0, 0],
                      'CGr84_mdeep_A':[0, 0],
                      'CGr84_mdeep_B':[0, 0],
                      }  # XXX offset [y, x]
    
    morpho_file = {'CGr32-M1_deep':'recap_morpho_params_CGr32-M1.txt',
                   'CGr32-M2_deep':'recap_morpho_params_CGr32-M2.txt',
                   'CGr32-M3_deep':'recap_morpho_params_CGr32-M3.txt',
                   'CGr34_mid_deep':'recap_morpho_params_CGr34.txt',
                   'CGr79_deep':'recap_morpho_params_CGr79.txt',
                   'CGr114_snshot':'recap_morpho_params_CGr114.txt',
                   'CGr28':'recap_morpho_params_CGr28.txt',
                   'CGr30_deep':'recap_morpho_params_CGr30.txt',
                   'CGr84-NorthA':'recap_morpho_params_CGr84-NorthA.txt',
                   'CGr84-NorthB':'recap_morpho_params_CGr84-NorthB.txt',
                   'CGr84_mdeep_A':'recap_morpho_params_CGr84A.txt',
                   'CGr84_mdeep_B':'recap_morpho_params_CGr84B.txt',
                   }
    
    morpho_suff = {'CGr32-M1_deep':'',
                   'CGr32-M2_deep':'',
                   'CGr32-M3_deep':'',
                   'CGr34_mid_deep':'A',
                   'CGr79_deep':'',
                   'CGr114_snshot':'B',
                   'CGr28':'',
                   'CGr30_deep':'B',
                   'CGr84-NorthA':'',
                   'CGr84-NorthB':'',
                   'CGr84_mdeep_A':'A',
                   'CGr84_mdeep_B':'B',
                   }
    
    line = 'o2'
    
    for gr in groups:
        #if 'CGr32' in gr:
            #continue
        file_recap = path + gr + '/'+  line + '/recap_kinematics_parameters_2_slp_xyi_mclean5.0.txt'
        file_input = path + gr + '/input_fit_o2.txt'
        file_minput = path + gr + '/maps_input_o2.txt'
        file_morph = pathhst + groups[gr] + '/' + morpho_file[gr]
        # XXX I had to add titles for the 4 last columns (a b c d)
        pathout = path + 'out/'
        deltapa = 0
        
        cat_recap = ascii.read(file_recap)
        cat_input = ascii.read(file_input)
        cat_minput = ascii.read(file_minput)
        cat_morpho = ascii.read(file_morph)
        #print(cat_morpho)
        
        for obj in cat_recap:
            name = obj['ID']
            print(name)
            
            #z = float(name.split('_Z_')[-1])
            ind, = np.where(cat_input['gal'] == name)
            num = name.split('_')[-2]
            grid = name.split('_')[0]
            print(num, grid)
            
            hst = pathhst + groups[gr] + '/' + num + '_' + grid + morpho_suff[gr] + '_comb_out.fits'
            
            try:
                z = cat_minput['z'][cat_minput['name'] == name][0]
            except:
                print('WARNING, galaxy ID ', name, ' is missing !!!!!')
                z = 0
            
            psf = cat_input['psfx'][ind[0]]  # pixels MUSE (FWHM)
            smooth = cat_input['smooth'][ind[0]]  # pixels MUSE (FWHM)
            psff = np.sqrt(psf**2 + smooth**2)
            #print(psff)
            lsf = cat_input['psfz'][ind[0]]  # km/s  (sigma)
            xc = obj['X']
            yc = obj['Y']
            pa = obj['PA']  # par rapport au nord dans l'image MUSE
            vsys = obj['VS']
            
            rc = obj['RLAST']  # pixels
            nmorph = num + '_' + grid + morpho_suff[gr]
            r22 = 2.2 * cat_morpho[cat_morpho['ID'] == nmorph]['R_d'][0] / 1.67835  # in HST pixels  (re = 1.67835 * rd)
            
            #flux = glob.glob(path + line + '/' + name + '/' + name + '_ssmooth_flux_common_mclean5.0.fits')
            flux = list(set(glob.glob(path + gr + '/' + line + '/' + name + '/' + name + '_ssmooth_flux_common_*.fits')) - set(glob.glob(path + gr + '/' + line + '/' + name + '/' + name + '_ssmooth_flux_common_*clean5.0.fits')))
            
            #snr = path + gr + '/' + line + '/' + name + '/' + name + '_ssmooth_snr_common_mclean5.0.fits'
            snr = path + gr + '/' + line + '/' + name + '/' + name + '_ssmooth_snr_common.fits'
            
            vf = path + gr + '/' + line + '/' + name + '/' + name + '_ssmooth_vel_common_mclean5.0.fits'
            vfm = path + gr + '/' + line + '/' + name + '/' + name + '_modv_slp_xyi_mclean5.0.fits'
            vfr = path + gr + '/' + line + '/' + name + '/' + name + '_resv_slp_xyi_mclean5.0.fits'
            
            sig = path + gr + '/' + line + '/' + name + '/' + name + '_ssmooth_disp_common_mclean5.0.fits'
            sigm = path + gr + '/' + line + '/' + name + '/' + name + '_modd_slp_xyi_mclean5.0.fits'
            sigr = path + gr + '/' + line + '/' + name + '/' + name + '_resd_slp_xyi_mclean5.0.fits'
            
            offhst = groups_offsets[gr]
            
            print('names0: ', gr, name)
            if 'CGr84_mdeep' in gr:
                name = cat_minput['ID'][cat_minput['name'] == name][0] + '_o2'
            if 'CGr84_mdeep_A' in gr:
                name = 'CGr84b_' + name.split('_')[-2] + '_o2'
            print('names0: ', gr, name)
            
            paper_map(hst, flux, snr, vf, vfm, vfr, sig, sigm, sigr, name, z, xc, yc, vsys, pa, r22, rc, lsf, psff, pathout, deltapa=deltapa, offset=offhst)
            #return


if __name__ == "__main__":
    main()