#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Benoit Epinat - LAM
         modified by Wilfried Mercier - IRAP
"""

import astropy.wcs as wcs
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import numpy as np
import copy


def write_stamp(hdr, image, npix=None, xcen=None, ycen=None, pathout='./', outname=None, overwrite=True):
    '''
    Write an hst stamp given as input into a new file.
    
    Parameters
    ----------
        hdr : fits header
            header to write
        image : fits image
            image to write
        npix :int
            half-size of the image in pixels
        outname : str
            name of the output hst stamp
        overwrite : bool
            whether to overwrite the stamp if it already exists or not
        pathout : str
            path of the output hst stamp
        xcen : int
            x position of the image centre in pixels in the orignal image world coordinate system
        ycen : int
            identical to xcen for the y position
    '''
    
    if None in [npix, xcen, ycen, outname]:
        raise ValueError('One of the following variable was not provided: npix, xcen, ycen or outname.')
    
    # Copy header
    hdrc           = copy.deepcopy(hdr)
    hdrc['CRPIX1'] = hdr['CRPIX1'] - xcen + npix
    hdrc['CRPIX2'] = hdr['CRPIX2'] - ycen + npix
    im1            = image[np.int(np.round(ycen - npix)):np.int(np.round(ycen + npix)), np.int(np.round(xcen - npix)):np.int(np.round(xcen + npix))]
    hdu            = fits.PrimaryHDU(im1, hdrc)
    hdulist        = fits.HDUList(hdu)
    hdulist.writeto(pathout + outname, overwrite=overwrite)
    
    return


def extract_stamps_udf(image_file, gal_list, size=2., factor=1., pathout='./', groupNumber=None):
    '''
    This function enables to extract several images centered on galaxies from a large single image
    
    Parameters
    ----------
        image_file: string
            name of the input image (fits format)
        gal_list: string
            name of the galaxy list (columns: ID, z, Flag, RA, DEC, I_AB)
        groupNumber : int
            group number of the galaxies
        size: float
            half size of the small images in arcsec
        factor: float
            factor by which multiplying the data for using GALFIT
        pathout: string
            path where to write the extracted images
    '''
    
    if groupNumber is None:
        raise ValueError('A group number must be provided.')
    
    # Getting data
    print(image_file)
    hdul         = fits.open(image_file)
    im           = hdul[0].data * factor
    hdr          = hdul[0].header
    hdr['UZERO'] = -2.5 * np.log10(1. / factor)
    
    if type(gal_list) is str:
        cat          = ascii.read(gal_list)
    else:
        cat          = gal_list
    
    # Rename columns
    columns      = cat.colnames
    [cat.rename_column(i, i.lower()) for i in columns]
    
    # Get coordinates
    radeg        = cat['ra']
    decdeg       = cat['dec']
    
    # Change pixel size from degree to arcsec
    hst_pix      = np.sqrt(hdr['CD1_1'] ** 2 + hdr['CD1_2'] ** 2) * 3600
    npix         = np.round(size / hst_pix)
    
    # Coordinate system
    w            = wcs.WCS(hdr, hdul)
    xc, yc       = w.wcs_world2pix(radeg, decdeg, 0)
    
    # Go through each galaxy
    for x, y, iid in zip(xc, yc, cat['id']):
        outname  = '%d_CGr%s.fits' %(iid, groupNumber)
        write_stamp(hdr, im, npix=npix, xcen=x, ycen=y, pathout=pathout, outname=outname)
    return
