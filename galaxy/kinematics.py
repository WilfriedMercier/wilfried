#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:09:08 2019

@author: Benoit Epinat - LAM & Wilfried Mercier - IRAP

Fonctions related to kinematical modelling of galaxies.
"""

import os, glob
import numpy             as     np
import astropy.io.ascii  as     asci
import astropy.io.fits   as     fits
import astropy.constants as     ct
from   ..utilities.coloredMessages import *
from   .MUSE                       import compute_lsfw

##############################################################################################
#                                  Theoretical calculations                                  #
##############################################################################################

class Vprojection:
    '''Class used to project along the line of sight a given velocity profile (function) .'''
    
    def __init__(self, velocity):
        '''
        Parameters
        ----------
            velocity : function
                function describing the velocity profile to project (must be a function of the distance r to the centre)
        '''
        
        self.func = velocity
        
    
    def distance(self, s, D, R, *args, **kwargs):
        '''
        Distance from centre for a point along the major axis for a galaxy at distance D, when this point is at a distance s of us and projected distance R of the centre.
        
        Parameters
        ----------
            s : float/int or array of floats/int
                distance from the point to our location (same unit as L and D)
            D : float
                cosmological angular diameter distance between the galaxy centre and us
            R : float/int
                projected distance of the point along the major axis relative to the galaxy centre
                
        Return as a tuple the foreground distance, the background distance and the angle in radians of the point relative to the centre in this order.
        '''
        
        if not hasattr(s, 'unit'):
            s = u.Quantity(s, unit='pc')
        
        if np.any(s<0) or D<0 or R<0:
            raise ValueError('One of the provided distances is negative.')
            
        if not isinstance(R, (float, int)) or not isinstance(D, (float, int)):
            raise TypeError('Either R or D is neither float nor int.')
            
        if isinstance(s, (float, int)):
            s              = np.array([s])
            
        theta              = np.arctan(R/D)
        dist2              = D**2 + R**2
        
        # Depending on where we are (s in the foreground or background), the formula change
        # So we split between foreground and background to reunite in the end
        whereFor           = np.nonzero(s**2<dist2)
        sFor               = s[whereFor]
        
        whereBac           = np.nonzero(s**2>=dist2)
        sBac               = s[whereBac]
        
        # Compute distances from centre
        dFor               = np.sqrt(D**2 - sFor**2 * np.cos(2*theta))
        dBac               = np.sqrt(D**2 + sBac*(sBac + 2*R*np.sin(theta)))
        
        return dFor, dBac, theta
        
    
    #################################################
    #                  Projections                  #
    #################################################
        
    def projection(self, *args, which='edge-on', **kwargs):
        '''
        Wrapper to call the correct projection depending on the which parameter. *args and **kwargs are assumed to be correct.

        Parameters
        ----------
            which : str
                which type of projection to apply. Default si edge-on galaxy.

        Return the computed projection (see exact method definition for more details).
        '''
        
        if which.lower() == 'edge-on':
            return self.edgeOnDisk(*args, **kwargs)
        else:
            raise ValueError('The given projection %s is not supported (yet). Cheers !')
        
        
    def edgeOnDisk(self, s, D, R, *args, **kwargs):
        '''
        Compute the line of sight projected velocity at a distance s to us, at a projected distance R from the galaxy centre located at a distance D from us (cosmological angular diameter distance) for a galaxy seen edge-on.

        Parameters
        ----------
            s : float/int or array of floats/int
                distance from the point to our location (same unit as L and D)
            D : float
                cosmological angular diameter distance between the galaxy centre and us
            R : float/int
                projected distance of the point along the major axis relative to the galaxy centre

        Return a numpy array with the line of sight projected velocity computed at every distance in R.
        '''
        
        if isinstance(s, (float, int)):
            s              = np.array([s])
            
        # Depending on where we are (s in the foreground or background), the formula change
        # So we split between foreground and background to reunite in the end
        dist2              = D**2 + R**2
        whereFor           = np.nonzero(s**2<dist2)
        sFor               = s[whereFor]
        
        whereBac           = np.nonzero(s**2>=dist2)
        sBac               = s[whereBac]
            
        dFor, dBac, theta  = self.distance(s, D, R)
        stheta             = np.sin(theta)
        
        # Compute velocity at d
        if len(dFor)>0:
            vFor           = self.func(dFor) * np.sin(theta + np.arcsin( sFor*stheta/dFor ))
        else:
            vFor           = 0
        
        if len(dBac)>0:
            vBac           = self.func(dBac) * np.sqrt(1 - ( (sBac + R*stheta - np.sqrt(dist2))/dBac )**2)
        else:
            vBac           = 0
        
        # Reunitting
        velocity           = s.copy()
        velocity[whereFor] = vFor
        velocity[whereBac] = vBac
        
        return velocity
            
        

##########################################################################################################
#                                         Kinematical properties                                         #
##########################################################################################################

def velocityAtR(radius, Vt, Rt, Rlast, verbose=True):
    '''
    Assuming a linear ramp model, try to compute the velocity at a single radius.

    Parameters
    ----------
        radius : float/int
            position where the velocity must be computed (in the same units as Rt and Rlast)
        Vt : float
            plateau velocity
        Rt : float
            radius of transition between the inner linear slope and the outer plateau
        Rlast : float
            distance from the centre of the furthest pixel used in the fit
            
    Optional parameters
    -------------------
        verbose : bool
            whether print error messages or not. Default is True.

    Return the computed velocity in units of Vt and a boolean value indicating whether the computed value is reliable or not.
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
    
    Authors
    -------
        Benoit Epinat - LAM
        modified by Wilfried Mercier - IRAP
    
    Parameters
    ----------
        mask: numpy array
            array containing the indices of the pixels to be masked
        image: numpy array
            image to be masked

    Returns the masked image.
    '''
    
    newImage       = image.copy()
    newImage[mask] = np.nan
    return newImage


def clean_galaxy(path, outputpath, name, lsfw, fraction, clean=None, data_mask='snr', thrl=None, thru=None, option='_ssmooth', line='_OII3729'):
    '''
    This function cleans the maps created by CAMEL for a given galaxy.
    
    Authors
    -------
        Benoit Epinat - LAM
        modified by Wilfried Mercier - IRAP
    
    Mandatory parameters
    --------------------
        fraction: float
            fraction for a lower threshold on the velocity dispersion map
        lsfw: float
            spectral resolution in km/s (sigma)
        name: str
            name of the galaxy
        ouputpath: str
            path where the ouput data will be stored
        path: str
            path where the input data are stored
        
    Optional parameters
    -------------------
        clean: str
            name of the manually cleaned map. Default is None so that no cleaned map is used to add an additional mask.
        data_mask: str/list of str
            basename of the map used for the threshold (e.g. 'snr' to use the signal to noise ratio map). Default is 'snr'.
        line: str
            line used (suffixe, e.g. '_Ha')
        option: str
            option of CAMEL to find the files to clean (e.g. '_ssmooth'). Default is '_ssmooth'.
        thrl: float/list of floats
            lower threshold for cleaning. Default is None so that the minimum (apart from nan) will be used.
        thru: float/list of floats
            upper threshold for cleaning. Default is None so that the maximum (apart from nan) will be used.
        
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
    
    Authors
    -------
        Benoit Epinat - LAM
        modified by Wilfried Mercier - IRAP
        
    How to use
    ----------
    
        Provide:
            - a filename containing two columns: the absolute file name of every galaxy config file (thus ending with .config) and their corresponding spectral resolution in km/s (e.g. /home/wilfried/CGr114_s/o2/CGr114_101_o2.config 30).
            - a keyword for the additional mask (apart from the spectral width criterion), or a list of keywords
            - thresholds if necessary
            - a clean map if there is one. The same name will be used for every galaxy. Best practice would be to use an identical name for all the galaxies (located in different folders), such as 'clean.fits'.
    
    Mandatory parameters
    --------------------
        filename: str
            name of the input file containing two columns: the list of galaxies and the associated spectral resolution in km/s (sigma)
        path: str
            path where the input file is stored
        logFile: str
            name of of the file containing the list of the output folder names
            
    Optional parameters
    -------------------
        clean: str
            name of the manually cleaned map. Default is None so that no clean file is used to add an another manual mask.
        data_mask: str/list of str
            basename of the map used for threshold (e.g. 'snr' for signal to noise ratio map). Default is 'snr'.
        fraction: float
            fraction for a lower threshold on the velocity dispersion map. Default is 1.0, that is the Line Spread Function Width (lsfw).
        thrl: float
            lower threshold for cleaning. Default is None, so that the minimum (apart from nan) will be used.
        thru: float
            upper threshold for cleaning. Default is None, so that the maximum (apart from nan) will be used.
        option: str
            option from camel to find the files to clean (e.g. '_ssmooth'). Default is '_ssmooth'.
        line: str
            line used (suffixe, e.g. '_Ha'). Default is '_OII3729'.
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
    '''
    Compute the spectral resolution in terms of velocity sigma from the line restframe wavelength, the redshift of the source and from MUSE LSF model: FWHM(lbda) = a2 * lbda ** 2 + a1 * lbda + a0
    
    Author
    ------
        Benoit Epinat - LAM
    
    Mandatory parameters
    --------------------
        lbda0: float
            rest frame wavelength of the line used to infer kinematics (in Angstroms)
        z: float
            redshift of the galaxy
    
    Optional parameters
    -------------------    
        a0: float
            lambda ** 0 coefficient of the variation of LSF FWHM with respect to lambda. Default is 5.835e-8 A.
        a1: float
            lambda ** 1 coefficient of the variation of LSF FWHM with respect to lambda. Default is -9.080e-4.
        a2: float
            lambda ** 2 coefficient of the variation of LSF FWHM with respect to lambda. Default is 5.983 A^{-1}.
        
    Return the observed (redshifted) wavelength, the LSF FWHM in Angstroms and the LSF dispersion in km/s, assuming a Gaussian shape for the LSF profile.
    '''
    
    lbda   = lbda0 * (1 + z)
    fwhm   = compute_lsfw(z, lbda0, a2=5.835e-8, a1=-9.080e-4, a0=5.983)
    velsig = fwhm / (lbda * 2 * np.sqrt(2 * np.log(2))) * ct.c.value * 1e-3
    return lbda, fwhm, velsig


def velres_setofgalaxies(inname, outname, lbda0, a2=5.835e-8, a1=-9.080e-4, a0=5.983):
    '''
    Compute the resolution in velocity for a list of galaxies and write it into an output file along the corresponding file name.
    
    Authors
    -------
        Benoit Epinat - LAM
        modified by Wilfried Mercier - IRAP
        
    How to use
    ----------
        Provide an input and output file names and the line rest-frame wavelength in Angstroms. The input file shoudl contain the following columns:
            - galaxy file names
            - redshift of the galaxies
    
    Mandatory parameters
    --------------------
        inname: str
            input file name containing the list of galaxies and their redshift. In this version, the redshift is in the name itself, as well as the line used.
        outname: str
            output file name containing the list of galaxies and the spectral resolution

    Optional parameters
    -------------------    
        a0: float
            lambda ** 0 coefficient of the variation of LSF FWHM with respect to lambda. Default is 5.835e-8 A.
        a1: float
            lambda ** 1 coefficient of the variation of LSF FWHM with respect to lambda. Default is -9.080e-4.
        a2: float
            lambda ** 2 coefficient of the variation of LSF FWHM with respect to lambda. Default is 5.983 A^{-1}.
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
    
    Authors
    -------
        Benoit Epinat - LAM
        modified by Wilfried Mercier - IRAP
    
    Parameters
    ----------
        image: numpy array
            image used to create the mask
        thrl: float
            lower threshold
        thru: float
            upper threshold

    Returns a boolean mask (True everywhere thrl<=image<=thru).
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
