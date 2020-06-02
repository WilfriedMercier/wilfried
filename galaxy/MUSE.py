#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:37:49 2020

@author: wilfried

Functions directly related to MUSE instrument and its observations.
"""

import numpy         as np
import astropy.units as u
import astropy.wcs   as wcs
import os.path       as opath


def centreFromHSTtoMUSE(X, Y, imHST, imMUSE):
    '''
    Convert centre coordinates in pixels in HST image into pixel coordinates in MUSE image/cube.

    Parameters
    ----------
        X : float/int
            x centre position
        Y : float/int
            y centre position
        imHST : str
            file containing the HST image to generate wcs object
        imMUSE : str
            file containing a MUSE cube or image to generate wcs object

    Return the new (MUSE) pixel coordinates.
    '''
    
    # Check files exist first
    if not opath.isfile(imHST):
        raise IOError('HST image %s does not exist.' %imHST)
    
    if not opath.isfile(imMUSE):
        raise IOError('MUSE image/cube %s does not exist.' %imMUSE)
    
    # HST wcs object
    wh      = wcs.WCS(imHST)
    
    # MUSE wcs object
    wm      = wcs.WCS(imMUSE)
    
    # Coordinates from HST
    ra, dec = wh.wcs_pix2world(X-1, Y-1, 0)
     
    # Convert back to MUSE pixels this time
    x, y    = wm.wcs_world2pix(ra, dec, 0)
    
    return x, y

#####################################################################
#                            LSF and PSF                            #
#####################################################################

class ListGroups:
    
    def __init__(self):
        
        # Rest-frame wavelengths in Angstrom
        self.OII  = 3729 
        self.OIII = 5007
        
        self.gaussian = {'114'    : {'OII'  :  3.705,	
                                     'OIII' : 3.315, 
                                     'z'    : 0.598849},
                          '23'    : {'OII'  : 4.28, 
                                     'OIII' : 3.65, 
                                     'z'    : 0.850458}, 
                          '26'    : {'OII'  : 3.68, 
                                     'OIII' : 3.34, 
                                     'z'    : 0.439973}, 
                          '28'    : {'OII'  : 3.62, 
                                     'OIII' : 3.26, 
                                     'z'    : 0.950289},
                          '30_d'  : {'OII'  : 3.485,
                                     'OIII' : 1, 
                                     'z'    : 0.809828}, 
                          '30_bs' : {'OII'  : 3.185,	
                                     'OIII' : 2.815, 
                                     'z'    : 0.809828},
                          '32-M1' : {'OII'  : 2.975,	
                                     'OIII' : 2.58,  
                                     'z'    : 0.753319}, 
                          '32-M2' : {'OII'  : 3.16,	
                                     'OIII' : 2.54, 
                                     'z'    : 0.753319}, 
                          '32-M3' : {'OII'  : 3.61,	
                                     'OIII' : 3.3, 
                                     'z'    : 0.753319},
                          '34_d'  : {'OII'  : 3.31,	
                                     'OIII' : 2.995, 
                                     'z'    : 0.857549}, 
                          '34_bs' : {'OII'  : 3.3,	
                                     'OIII' : 3.003, 
                                     'z'    : 0.85754},
                          '51'    : {'OII'  : 3.75, 
                                     'OIII' : 3.28, 
                                     'z'    : 0.386245}, 
                          '61'    : {'OII'  : 3.915,	
                                     'OIII' : 3.34, 
                                     'z'    : 0.364009}, 
                          '79'    : {'OII'  : 3.29,	
                                     'OIII' : 2.695, 
                                     'z'    : 0.780482},
                          '84'    : {'OII'  : 3.24,	
                                     'OIII' : 3.055, 
                                     'z'    : 0.731648}, 
                          '84-N'  : {'OII'  : 2.89,	
                                     'OIII' : 2.58, 
                                     'z'    : 0.727755}
                         }
        
        self.moffat = {'23'   : {'OII'  : 3.97, 
                                 'OIII' : 3.29, 
                                 'z'    : 0.850458}, 
                       '26'   : {'OII'  : 3.16, 
                                 'OIII' : 2.9, 
                                 'z'    : 0.439973}, 
                       '28'   : {'OII'  : 3.18, 
                                 'OIII' : 3.13, 
                                 'z'    : 0.950289},
                      '32-M1' : {'OII'  : 2.46, 
                                 'OIII' : 1.9, 
                                 'z'    : 0.753319}, 
                      '32-M2' : {'OII'  : 2.52, 
                                 'OIII' : 2.31, 
                                 'z'    : 0.753319}, 
                      '32-M3' : {'OII'  : 2.625, 
                                 'OIII' : 2.465, 
                                 'z'    : 0.753319},
                      '51'    : {'OII'  : 3.425, 
                                 'OIII' : 2.95, 
                                 'z'    : 0.386245}, 
                      '61'    : {'OII'  : 3.2, 
                                 'OIII' : 3.02, 
                                 'z'    : 0.364009}, 
                      '79'    : {'OII'  : 2.895, 
                                 'OIII' : 2.285, 
                                 'z'    : 0.780482}, 
                      '84-N'  : {'OII'  : 2.49, 
                                 'OIII' : 2.21, 
                                 'z'    : 0.727755}, 
                      '30_d'  : {'OII'  : 2.995, 
                                 'OIII' : 2.68, 
                                 'z'    : 0.809828}, 
                      '30_bs' : {'OII'  : 2.745, 
                                 'OIII' : 2.45, 
                                 'z'    : 0.809828},
                      '84'    : {'OII'  : 2.835, 
                                 'OIII' : 2.715, 
                                 'z'    : 0.731648}, 
                      '34_d'  : {'OII'  : 2.89, 
                                 'OIII' : 2.695, 
                                 'z'    : 0.857549}, 
                      '34_bs' : {'OII'  : np.nan, 
                                 'OIII' : np.nan, 
                                 'z'    : 0.85754},
                      '114'   : {'OII'  : 3.115, 
                                 'OIII' : 2.81, 
                                 'z'    : 0.598849}
                     }

def compute_lsfw(z, lambda0, a2=5.835e-8, a1=-9.080e-4, a0=5.983):
    '''
    Compute the MUSE Line Spread Fonction FWHM given the rest-frame wavelength of the line and the redshift of the corresponding object.

    Mandatory parameters
    --------------------
        lambda0 : float/int
            rest-frame wavelength in Angstroms
        z : float/int
            redshift of the object
        
    Optional parameters
    -------------------    
        a0: float
            lambda ** 0 coefficient of the variation of LSF FWHM with respect to lambda. Default is 5.835e-8 A.
        a1: float
            lambda ** 1 coefficient of the variation of LSF FWHM with respect to lambda. Default is -9.080e-4.
        a2: float
            lambda ** 2 coefficient of the variation of LSF FWHM with respect to lambda. Default is 5.983 A^{-1}.

    Return the LSF FWHM in Angstroms.
    '''
    
    if np.any(z<0):
        raise ValueError('Redshift must be positive valued.')
    
    # If the user uses astropy quantities, we convert to the right one    
    if isinstance(lambda0, u.Quantity):
        lambda0.to('Angstrom')
    
    lbda = lambda0 * (1 + z)
    return a2 * lbda ** 2 + a1 * lbda + a0


def computeFWHM(wavelength, field, model='Gaussian'):
    '''
    Compute the FWHM at a given observed wavelength assuming a linearly decreasing relation for the FWHM with wavelength (calibrated on OII and OIII measurements at different redshifts).
    Only MUSE fields in the COSMOS field are considered here.
    
    Mandatory parameters
    --------------------
        field : str
            the group for each desired wavelength
        wavelength : int
            the wavelength for which we want to compute the FWHM in Angstrom
            
    Optional parameters
    -------------------
        model : 'Moffat' or 'Gaussian'
            model to use. Default is Gaussian.
        
    Return the computed FWHM.
    '''
    
    # The FWHM were computed for two wavelengths at some group redshift
    groups            = ListGroups()
    if model.lower() == "gaussian":
        try:
            psfValues = groups.gaussian[field]
        except KeyError:
            raise KeyError('Given field name %s is not correct. Correct values are %s.' %(field, str(list(groups.gaussian()))[1:-1]))
    elif model.lower() == 'moffat':
        try:
            psfValues = groups.moffat[field]
        except KeyError:
            raise KeyError('Given field name %s is not correct. Correct values are %s.' %(field, str(list(groups.gaussian()))[1:-1]))
    else:
        raise Exception("Model %s not recognised. Available values are %s." %(model, "Moffat or Gaussian"))
    
    # Wavelength difference
    deltaLambda = groups.OIII - groups.OII
    
    #A factor of (1+z) must be applied to deltaLambda and OII lambda
    
    slope  = (psfValues['OIII'] - psfValues['OII']) / (deltaLambda*(1+psfValues['z']))
    offset = psfValues['OII']   - slope*groups.OII*(1+psfValues['z'])
    
    FWHM = slope*wavelength+offset
            
    return FWHM