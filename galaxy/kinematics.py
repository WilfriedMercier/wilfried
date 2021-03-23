#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*Author:* Benoit Epinat - LAM & Wilfried Mercier - IRAP

Fonctions related to kinematical modelling of galaxies.
"""

import os, glob
import numpy             as     np
import astropy.io.ascii  as     asci
import astropy.io.fits   as     fits
import astropy.constants as     ct
from   .symlinks.coloredMessages import *
from   .MUSE                       import compute_lsfw

##########################################################################################################
#                                         Kinematical properties                                         #
##########################################################################################################

def velocityAtR(radius, Vt, Rt, Rlast, verbose=True):
    r'''
    Assuming a linear ramp model, try to compute the velocity at a single radius
    
    .. math::
        
        V(R) = V_t \times R/r_t \ \ \rm{if}\ \ R \leq r_t \ \ \rm{else} \ \ V_t.
    
    *Author:* Wilfried Mercier - IRAP

    :param radius: position where the velocity must be computed (in the same units as Rt and Rlast)
    :type radius: int or float
    :param float Vt: plateau velocity
    :param float Rt: radius of transition between the inner linear slope and the outer plateau
    :param float Rlast: distance from the centre of the furthest pixel used in the fit
    
    :param bool verbose: (**Optional**) whether print error messages or not
    
    :returns: the computed velocity in units of Vt and a boolean value indicating whether the computed value is reliable or not
    :rtype: int or float and bool
    '''

    # If Rt<Rlast or (Rt>Rlast and radius<Rlast), there is no issue, but when Rt>Rlast, the value is much more unconstrained
    if Rt <= Rlast or radius <= Rlast:
        ok           = True
        if radius < Rt:
            velocity = radius/Rt*Vt
        else:
            velocity = Vt
    else:
        if verbose:
            print(errorMessage('Rt > Rlast by %.1f' %(Rt-Rlast)))
        ok           = False
        velocity     = radius/Rt*Vt
        
    return velocity, ok

####################################################################################################################
#                                                Analysis part                                                     #
####################################################################################################################

def apply_mask(mask, image):
    '''
    This function applies a mask to an image and puts nan in the masked area.
    
    *Author:* Benoit Epinat - LAM
    
    *modified by Wilfried Mercier - IRAP*

    :param ndarray mask: array containing the indices of the pixels to be masked
    :param ndarray image: image to be masked
    
    :returns: the masked image
    :rtype: ndarray
    '''
    
    newImage       = image.copy()
    newImage[mask] = np.nan
    return newImage


def clean_galaxy(path, outputpath, name, lsfw, fraction, clean=None, data_mask='snr', thrl=None, thru=None, option='_ssmooth', line='_OII3729'):
    '''
    This function cleans the maps created by CAMEL for a given galaxy.
    
    *Author:* Benoit Epinat - LAM
    
    *modified by Wilfried Mercier - IRAP*

    :param float fraction: fraction for a lower threshold on the velocity dispersion map
    :param float lsfw: spectral resolution in *km/s* (sigma)
    :param str name: name of the galaxy
    :param str ouputpath: path where the ouput data will be stored
    :param str path: path where the input data are stored

    :param str clean: (**Optional**) name of the manually cleaned map. If None, no cleaned map is used to add an additional mask.
    :param data_mask: (**Optional**) basename of the map used for the threshold (e.g. 'snr' to use the signal to noise ratio map)
    :type data_mask: str or list[str]
    :param str line: (**Optional**)  line used (suffixe, e.g. '_Ha')
    :param str option: (**Optional**) option of CAMEL to find the files to clean (e.g. '_ssmooth')
    :param thrl: (**Optional**)  lower threshold for cleaning. Is None, the minimum (apart from nan) will be used.
    :type thrl: float or list[float]
    :param thru: (**Optional**)  upper threshold for cleaning. Is None, the maximum (apart from nan) will be used.
    :type thru: float or list[float]
    
    :returns: None
    '''
    
    #Checking path exists
    if not(os.path.isdir(path)):
        print('clean_galaxy: path % s does not exist' % (str(path)))
        answer      = input('Should we skip this galaxy ? [Y or N] ')
        if answer.lower() in ['y', 'yes']:
            return
        else:
            raise OSError()
 
    #Checking output path exists
    if not(os.path.isdir(outputpath)):
        print('clean_galaxy: path % s does not exist' % (str(outputpath)))
        answer      = input('Should we skip this galaxy ? [Y or N] ')
        if answer.lower() in ['y', 'yes']:
            return
        else:
            raise OSError()
    
    # Dispersion threshold
    smin            = lsfw * fraction
    print('clean_galaxy: dispersion threshold % s' % (str(smin)))
    
    # Correct path name if wrongly provided
    if path[-1] != '/':
        path       += '/'
        
    if outputpath[-1] != '/':
        outputpath += '/'
        
    # Set data_mask to a list if it is only a string
    if isinstance(data_mask, str):
        data_mask   = [data_mask]
        
    if not isinstance(thrl, list):
        thrl        = [thrl]
    
    if not isinstance(thru, list):
        thru        = [thru]
    
    # Get file in path with given name and option + dispersion map + map used for the mask (generally snr)
    files           = glob.glob(path + name + option + '*.fits')
    fim0            = glob.glob(path + name + option + '_disp_*[pn].fits')
    fim1            = []
    for name_mask in data_mask:
        fim1.append(glob.glob(path + name + option + '_' + name_mask + '_*[pn].fits'))
    
    # Try to open the dispersion file (first extension)
    try:
        with fits.open(fim0[0]) as hdul0:
            im0     = hdul0[0].data
        print('clean_galaxy: using % s' % (str(fim0[0])))
    except:
        print('clean_galaxy: % s not found' % (str(path + name + option + '_disp_*[pn].fits')))
        answer      = input('Should we skip this galaxy ? [Y or N] ')
        if answer.lower() in ['y', 'yes']:
            return
        else:
            raise IOError()
        
    # Try to open the map used for the mask (first extension for each map in fim1 list)
    im1             = []
    for fi1, name_mask in zip(fim1, data_mask):
        try:
            with fits.open(fi1[0]) as hdul1:
                im1.append(hdul1[0].data)
            print('clean_galaxy: using % s' % (str(fi1[0])))
        except:
            print('clean_galaxy: % s not found' % (str(path + name + option + '_' + name_mask + '_*[pn].fits')))
            answer  = input('Should we skip this galaxy ? [Y or N] ')
            if answer.lower() in ['y', 'yes']:
                return
            else:
                raise IOError()
    
    ##################################################################
    #                         Creating masks                         #
    ##################################################################
    
    # Dispersion mask: True where im0>=smin
    print('clean_galaxy: making dispersion mask')
    mask0     = create_mask(im0, thrl=smin)
    
    # Provided masks: True where thrl<=im1<=thru
    mask1     = mask0.copy() * False + True
    for im, tl, tu, name_mask in zip(im1, thrl, thru, data_mask):
        print('clean_galaxy: making an additional mask (%s)' %name_mask)
        mask1 = np.logical_and(mask1, create_mask(im, thrl=tl, thru=tu))
    
    # Keep positions where the values are out of bounds
    mask      = np.where((np.logical_not(mask0)) | (np.logical_not(mask1)))
    
    ##############################################################
    #               Using the manually cleaned map               #
    ##############################################################
    
    if clean is not None:
        fcl          = glob.glob(path + clean)
        try:
            # Get data and generate a mask wherever data is not None (neither upper nor lower threshold applied)
            with fits.open(fcl[0]) as hducl:
                imcl = hducl[0].data
                
            print('clean_galaxy: using % s' % (str(fcl[0])) )    
            maskcl   = create_mask(imcl, thrl=None, thru=None)
            mask2    = np.where((np.logical_not(mask0)) | (np.logical_not(mask1)) | (np.logical_not(maskcl)))
        except:
            print('clean_galaxy: % s not found' % (str(path + clean)) )
            clean    = None
    
    #################################################
    #             Updating all the maps             #
    #################################################
    
    for fim in files:
        
        # We skip files which are already cleaned (either manually or automatically). This assumes that cleaned files have a 'clean' keyword in their name
        if 'clean' in fim:
            continue
        
        with fits.open(fim) as hdul:
            im           = hdul[0].data
            
        # We skip datacubes
        if im.ndim == 3:
            continue

        # Setting up the threshold value that will appear in the output file name
        thr              = 0
        notNonethrl      = list(filter(lambda x:x is not None, thrl))
        notNonethru      = list(filter(lambda x:x is not None, thru))
        if  len(notNonethrl) != 0 :
            thr          = min(notNonethrl) 
        if len(notNonethru) != 0:
            thr          = max(notNonethru)

        # Directly modify the file data by applying the master mask (velocity dispersion + any other mask used such as snr)
        if clean is None:
            hdul[0].data = apply_mask(mask, im)
            fimcl        = fim.split('.fits')[0].split('/')[-1] + '_clean%3.1f.fits' %thr
        else:
            hdul[0].data = apply_mask(mask2, im)
            fimcl        = fim.split('.fits')[0].split('/')[-1] + '_mclean%3.1f.fits' %thr
            
        hdul.writeto(outputpath + fimcl, overwrite=True)
        print('output written in %s' %(outputpath+fimcl))
    
    return


def clean_setofgalaxies(path, filename='galsList.input', logFile='folderList.list', fraction=1., data_mask='snr', thrl=None, thru=None, option='_ssmooth', line='_OII3729', clean=None):
    '''
    Clean maps created by camel for a list of galaxies.
    
    *Authors:* Benoit Epinat - LAM
    
    *modified by Wilfried Mercier - IRAP*
        
    .. note::
        **How to use**
        
        Provide:
            - a filename containing two columns: the absolute file name of every galaxy config file (thus ending with .config) and their corresponding spectral resolution in km/s (e.g. /home/wilfried/CGr114_s/o2/CGr114_101_o2.config 30).
            - a keyword for the additional mask (apart from the spectral width criterion), or a list of keywords
            - thresholds if necessary
            - a clean map if there is one. The same name will be used for every galaxy. Best practice would be to use an identical name for all the galaxies (located in different folders), such as 'clean.fits'.
    
    :param str filename: name of the input file containing two columns: the list of galaxies and the associated spectral resolution in *km/s* (sigma)
    :param str path: path where the input file is stored
    :param str logFile: name of of the file containing the list of the output folder names
            
    :param str clean: (**Optional**) name of the manually cleaned map.If None, no clean file is used to add an another manual mask.
    :param data_mask: (**Optional**) basename of the map used for threshold (e.g. 'snr' for signal to noise ratio map)
    :type data_mask: str or list[str]
    :param float fraction: (**Optional**) fraction for a lower threshold on the velocity dispersion map
    :param float thrl: (**Optional**) lower threshold for cleaning. Is None, the minimum (apart from nan) will be used.
    :param float thru: (**Optional**) upper threshold for cleaning. If None, the maximum (apart from nan) will be used.
    :param str option: (**Optional**) option from camel to find the files to clean (e.g. '_ssmooth')
    :param str line: (**Optional**) line used (suffixe, e.g. '_Ha')
    
    :returns: None
    '''
    
    # If the user provides o2 as line we change it to the correct value
    if line == 'o2':
        line = '_OII3729'
    
    cat                     = asci.read(filename)
    
    # Generating a new name if the given file name already exsits
    if os.path.isfile(logFile):
        splt                = logFile.rsplit('.', 1)
        
        if len(splt) == 1:
            end = ''
        else:
            end = '.' + splt[1]
        
        try:
            num             = int(splt[0][-1])
            splt[0]         = splt[0][:-1]
        except ValueError:
            num             = 1
        logFile             = splt[0] + str(num) + end
        
    with open(logFile, "w") as f:
        for l in cat:
            # Get name without the extension at the end and path
            name            = l[0].split('/')[-1].split('.config')[0]
            galPath         = l[0].split(name + '.config')[0]
            
            clean_galaxy(galPath, galPath, name, l[1], fraction, data_mask=data_mask, thru=thru, thrl=thrl, line=l, option=option, clean=clean)
            f.write(galPath.rpartition('/')[0] + "\n")
    return


def compute_velres(z, lbda0, a2=5.835e-8, a1=-9.080e-4, a0=5.983):
    r'''
    Compute the spectral resolution in terms of velocity sigma from the line restframe wavelength, the redshift of the source and from MUSE LSF model
    
    .. math::
        
        {\rm{FWHM}}(\lambda) = a2 \times \lambda^2 + a1 \times \lambda + a0
    
    *Author:* Benoit Epinat - LAM

    :param float lbda0: rest frame wavelength of the line used to infer kinematics in **Angstroms**
    :param float z: redshift of the galaxy

    :param float a0: (**Optional**) lambda ** 0 coefficient of the variation of LSF FWHM with respect to lambda
    :param float a1: (**Optional**) lambda ** 1 coefficient of the variation of LSF FWHM with respect to lambda
    :param float a2: (**Optional**) lambda ** 2 coefficient of the variation of LSF FWHM with respect to lambda
    
    :returns: the observed (redshifted) wavelength, the LSF FWHM in **Angstroms** and the LSF dispersion in **km/s**, assuming a Gaussian shape for the LSF profile.
    :rtype: float and float
    '''
    
    lbda   = lbda0 * (1 + z)
    fwhm   = compute_lsfw(z, lbda0, a2=5.835e-8, a1=-9.080e-4, a0=5.983)
    velsig = fwhm / (lbda * 2 * np.sqrt(2 * np.log(2))) * ct.c.value * 1e-3
    return lbda, fwhm, velsig


def velres_setofgalaxies(inname, outname, lbda0, a2=5.835e-8, a1=-9.080e-4, a0=5.983):
    '''
    Compute the resolution in velocity for a list of galaxies and write it into an output file along the corresponding file name.
    
    *Authors:* Benoit Epinat - LAM
    
    *modified by Wilfried Mercier - IRAP*
        
    .. note::
        
        **How to use**
        
        Provide an input and output file names and the line rest-frame wavelength in Angstroms. The input file shoudl contain the following columns:
            - galaxy file names
            - redshift of the galaxies

    :param str inname: input file name containing the list of galaxies and their redshift. In this version, the redshift is in the name itself, as well as the line used.
    :param str outname: output file name containing the list of galaxies and the spectral resolution

    :param float a0: (**Optional**) lambda ** 0 coefficient of the variation of LSF FWHM with respect to lambda
    :param float a1: (**Optional**) lambda ** 1 coefficient of the variation of LSF FWHM with respect to lambda
    :param float a2: (**Optional**) lambda ** 2 coefficient of the variation of LSF FWHM with respect to lambda
    
    :returns: None
    '''
    
    cat                        = asci.read(inname)
    with open(outname, 'w') as f:
        for line in cat:
            z                  = float(line[1])
            lbda, fwhm, velsig = compute_velres(z, lbda0, a2=a2, a1=a1, a0=a0)
            line               = '{0:100} {1:5.1f} \n'.format(line[0], velsig)
            f.write(line)
    return


def create_mask(image, thrl=None, thru=None):
    '''
    This function creates a mask from an image using a lower and an upper threshold.
    
    *Authors:* Benoit Epinat - LAM
    
    *modified by Wilfried Mercier - IRAP*

    :param ndarray image: image used to create the mask
    
    :param float thrl: (**Optional**) lower threshold
    :param float thru: (**Optional**) upper threshold
    
    :returns: boolean mask (True everywhere thrl<=image<=thru)
    :rtype: ndarray(bool)
    '''
    
    if thrl is None:
        thrl = np.nanmin(image)
    if thru is None:
        try:
            thru = np.nanmax(image)
        except ValueError:
            raise ValueError("\nIt seems that no maximum value can be found. Please check that your .fits file is not corrupted.\n")
            
    print('create_mask: lower threshold % s' % (str(thrl)) )
    print('create_mask: upper threshold % s' % (str(thru)) )
    
    return ((image <= thru) & (image >= thrl))
