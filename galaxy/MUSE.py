#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Functions directly related to MUSE instrument and its observations.
"""

import enum
import numpy         as     np
import astropy.units as     u
import astropy.wcs   as     wcs
import os.path       as     opath
from   astropy.io    import fits
from   typing        import Union
from   astropy.units import Quantity

class PSFModels(enum.Enum):
    
    GAUSSIAN_OLD = enum.auto()
    MOFFAT       = enum.auto()
    GAUSSIAN_NEW = enum.auto()

def centreFromHSTtoMUSE(X, Y, imHST, imMUSE, extHST=0, extMUSE=0, noError=False):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Convert centre coordinates in pixels in HST image into pixel coordinates in MUSE image/cube.

    :param X: x centre position
    :type X: int or float
    :param Y: y centre position
    :type Y: int or float
    :param str imHST: file containing the HST image to generate wcs object
    :param str imMUSE: file containing a MUSE cube or image to generate wcs object
        
    :param int extHST: extension to open in the HST fits file
    :param int extMUSE: extension to open in the MUSE file
    :param bool noError: whether to not throw an error or not when HST or MUSE file(s) is/are missing. If True, a tuple (np.nan, np.nan) is returned.

    :returns: new (MUSE) pixel coordinates
    :rtype: tuple
    
    :raises IOError:
        
        * if **noError** is False and **imHST** is not found
        * if **noError** is False and **imMUSE** is not found
    '''
    
    # Check files exist first
    if not opath.isfile(imHST):
        if not noError:
            raise IOError('HST image %s does not exist.' %imHST)
        print('HST image %s does not exist.' %imHST)
        return np.nan, np.nan
    
    if not opath.isfile(imMUSE):
        if not noError:
            raise IOError('MUSE image/cube %s does not exist.' %imMUSE)
        print('MUSE image/cube %s does not exist.' %imMUSE)
        return np.nan, np.nan
    
    # HST wcs object
    with fits.open(imHST) as hdul:
        wh  = wcs.WCS(hdul[extHST])
    
    # MUSE wcs object
    with fits.open(imMUSE) as hdul:
        wm  = wcs.WCS(hdul[extMUSE])
    
    # Coordinates from HST
    ra, dec = wh.wcs_pix2world(X-1, Y-1, 0)
     
    # Convert back to MUSE pixels this time
    x, y    = wm.wcs_world2pix(ra, dec, 0)
    
    return x, y

#####################################################################
#                            LSF and PSF                            #
#####################################################################

class ListGroups:
    
    '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Keep track of the measurements made to know the PSF evolution as a function of wavelength in each MUSE field.


        .. note::
           
            There are multiple models available:
              
                * self.gaussianOld and self.Moffat have:
                 
                    * FWHM within old fields were measured at two wavelengths ([OII] and [OIII]), so that is why PSF FWHM values are given for these two wavelengths along with the redshift of the group at which the lines were redshifted
                    * For the new fields, FWHM was measured as the median of each star modelled at ~100 different wavelengths. The given values are the slope and zero point
                
                * self.gaussian:
                    
                    * FWHM in old and new fields have been measured by Benoit at different locations. The given values are the slope (in arcsec/microns) and the zero point (in arcsec) at \lambda = 7000A.

        So, for these fields, slope and offset are directly given instead of given two measurements and a z value
    '''
        
    # Rest-frame wavelengths in Angstrom
    OII  = 3729 
    OIII = 5007
    
    #: Old PSF values using the Gaussian model from Valentina
    gaussianOld = {'114'   : {'OII'  : 3.705,	
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
                              'OIII' : 3.11, 
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
                              'z'    : 0.727755},
                      
                   '172'   : {'slope' : -2.99e-04,
                              'offset': 4.891},
                   '35'    : {'slope' : -2.555e-04,
                              'offset': 4.934},
                   '87'    : {'slope' : -2.306e-04,
                              'offset': 4.756},
                  }
    
    #: New PSF values from Benoit using the Gaussian model
    gaussian = {'114'    : {'slope' : -0.419,
                            'offset': 0.686},
                '23'    : {'slope' : -0.328,
                           'offset': 0.732}, 
                '26'    : {'slope' : -0.408,
                           'offset': 0.680}, 
                '28'    : {'slope' : -0.438,
                           'offset': 0.694},
                '30_d'  : {'slope' : -0.338,
                           'offset': 0.682}, 
                '32-M1' : {'slope' : -0.317,
                           'offset': 0.594}, 
                '32-M2' : {'slope' : -0.505,
                           'offset': 0.603,}, 
                '32-M3' : {'slope' : -0.507,
                           'offset': 0.663},
                '34_d'  : {'slope' : -0.299,
                           'offset': 0.657}, 
                '51'    : {'slope' : -0.381,
                           'offset': 0.662}, 
                '61'    : {'slope' : -0.384,
                           'offset': 0.667}, 
                '79'    : {'slope' : -0.441,
                           'offset': 0.608},
                '84'    : {'slope' : -0.303,
                           'offset': 0.662}, 
                '84-N'  : {'slope' : -0.571,
                           'offset': 0.736},
                '172'   : {'slope' : -0.529,
                           'offset': 0.586},
                '35'    : {'slope' : -0.475,
                           'offset': 0.650},
                '87'    : {'slope' : -0.430,
                           'offset': 0.645},
               }
    
    #: PSF values using the Moffat model
    moffat = {'23'    : {'OII'  : 3.97, 
                         'OIII' : 3.29, 
                         'z'    : 0.850458}, 
              '26'    : {'OII'  : 3.16, 
                         'OIII' : 2.9, 
                         'z'    : 0.439973}, 
              '28'    : {'OII'  : 3.18, 
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
                         'z'    : 0.598849},
                  
              'MXDF'  : {'OII'  : 0.64221739,
                         'OIII' : 0.58665217,
                         'z'    : 0}
                 }

def compute_lsfw(z: Union[int, float, np.ndarray], lambda0: Union[int, float], a2: float = 5.835e-8, a1: float = -9.080e-4, a0: float = 5.983):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the MUSE Line Spread Fonction FWHM given the rest-frame wavelength of the line and the redshift of the corresponding object.

    :param lambda0: rest-frame wavelength in Angstroms
    :type lambda0: int or float
    :param z: redshift of the object
    :type z: int or float or ndarray[int] or ndarray[float]
      
    :param float a0: lambda ** 0 coefficient of the variation of LSF FWHM with respect to lambda
    :param float a1: lambda ** 1 coefficient of the variation of LSF FWHM with respect to lambda
    :param float a2: lambda ** 2 coefficient of the variation of LSF FWHM with respect to lambda

    :returns: LSF FWHM in Angstroms
    :rtype: float or ndarray[float]
    
    :raises ValueError: if np.any(z<0)
    '''
    
    if np.any(z<0):
        raise ValueError('Redshift must be positive valued.')
    
    # If the user uses astropy quantities, we convert to the right one    
    if isinstance(lambda0, u.Quantity):
        lambda0.to('Angstrom')
    
    lbda = lambda0 * (1 + z)
    return a2 * lbda ** 2 + a1 * lbda + a0


def computeFWHM(wavelength: Union[int, float], field: str, model: PSFModels = PSFModels.GAUSSIAN_NEW) -> float:
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the FWHM at a given observed wavelength assuming a linearly decreasing relation for the FWHM with wavelength (calibrated on OII and OIII measurements at different redshifts).
    
    Only MUSE fields in the COSMOS field are considered here.

    :param str field: field name for each desired wavelength
    :param wavelength: observed wavelength for which we want to compute the FWHM in **Angstrom**
    :type wavelength: int or float
    
    :param model: model to use. Either GAUSSIAN_OLD (old Valentina's computations), GAUSSIAN_NEW (new Benoit's computations) or MOFFAT (old Valentina's computations)
    :type model: PSFModels
        
    :returns: computed FWHM in MUSE pixel (pixel scale equals 0.2 arcsec/pixel)
    :rtype: float
    '''
    
    # Get values
    if model == PSFModels.GAUSSIAN_OLD:
        psfDict = ListGroups.gaussianOld
    elif model == PSFModels.MOFFAT:
        psfDict = ListGroups.moffat
    elif model == PSFModels.GAUSSIAN_NEW:
        psfDict = ListGroups.gaussian
    else:
        raise Exception(f'Model {model} not recognised')
        
    if field not in psfDict:
        raise KeyError(f'Given field name {field} is not correct. Accepted values are {list(psfDict.keys())}.')
        
    psfValues   = psfDict[field]
        
    ##############################################
    #            Get slope and offset            #
    ##############################################
    
    # Old and Gaussian and model values must be computed in a convoluted way...
    if model in [PSFModels.GAUSSIAN_OLD, PSFModels.MOFFAT]:
        
        if field not in ['172', '35', '87']:
            
            deltaLambda = ListGroups.OIII - ListGroups.OII
            
            # A factor of (1+z) must be applied to deltaLambda and OII lambda
            slope       = (psfValues['OIII'] - psfValues['OII']) / (deltaLambda*(1+psfValues['z']))
            offset      = psfValues['OII']   - slope * ListGroups.OII * (1+psfValues['z'])
        else:
            slope       = psfValues['slope'] 
            offset      = psfValues['offset']
        
        FWHM            = slope*wavelength + offset
        
    elif model == PSFModels.GAUSSIAN_NEW:
        
        slope           = Quantity(psfValues['slope'],  unit='arcsec/micron')
        offset          = Quantity(psfValues['offset'], unit='arcsec')
        pivot           = Quantity(7000,                unit='angstrom')
        
        # Convert wavelength from Angstrom to micron
        wv              = Quantity(wavelength, unit='angstrom').to('micron')
        
        # Convert FWHM to arcsec and then to MUSE pixel
        FWHM            = (slope*(wv - pivot) + offset).to('arcsec').value / 0.2
        
    return FWHM