#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:09:08 2019

@author: Wilfried Mercier - IRAP

Useful functions for galaxy modelling and other related computation
"""

import numpy                              as     np
from   astropy.modeling.functional_models import Sersic2D
from   .misc                              import check_bns, compute_bn, PSFconvolution2D, checkAndComputeIe, intensity_at_re
from   astropy.constants                  import G

# If Planck18 not available we use Planck15
try:
    from astropy.cosmology                import Planck18 as cosmo
except ImportError:
    from astropy.cosmology                import Planck15 as cosmo



####################################################################################################################
#                                           1D profiles                                                            #
####################################################################################################################

def bulge(r, re, b4=None, Ie=None, mag=None, offset=None):
    """
    Computes the value of the intensity of a bulge (Sersic index n=4) at position r.
    
    Mandatory inputs
    ----------------
        r : float
            the position at which the profile is computed
        re : float
            half-light radius
                
    Optional inputs
    ---------------
        b4 : float
            b4 factor appearing in the Sersic profile defined as $2\gamma(8, b4) = 7!$. By default, b4 is None, and its value will be computed. To skip this computation, please give a value to bn when callling the function.
        Ie : float
            intensity at half-light radius
        mag : float
            galaxy total integrated magnitude used to compute Ie if not given
        offset : float
            magnitude offset in the magnitude system used
            
    Returns I(r) for a bulge.
    """
    
    b4, = check_bns([4], [b4])
    Ie  = checkAndComputeIe(Ie, 4, b4, re, mag, offset)
        
    return sersic_profile(r, 4, re, Ie=Ie, bn=b4)


def exponential_disk(r, re, b1=None, Ie=None, mag=None, offset=None):
    """
    Computes the value of the intensity of an expoential at position r.
    
    Mandatory inputs
    ----------------
        r : float
            the position at which the profile is computed
        re : float
            half-light radius
                
    Optional inputs
    ---------------
        b1 : float
            b1 factor appearing in the Sersic profile defined as $\gamma(2, b1) = 1/2$. By default, b1 is None, and its value will be computed. To skip this computation, please give a value to bn when callling the function.
        Ie : float
            intensity at half-light radius
        mag : float
            galaxy total integrated magnitude used to compute Ie if not given
        offset : float
            magnitude offset in the magnitude system used
                
    Return the profile value at position r.
    """
    
    b1, = check_bns([1], [b1])
    Ie  = checkAndComputeIe(Ie, 1, b1, re, mag, offset)
        
    return sersic_profile(r, 1, re, Ie=Ie, bn=b1)


def hernquist(r, a, M):
    '''
    Hernquist profile (mass 3D distribution).

    Parameters
    ----------
        a : float/int
            scale radius
        M : float/int
            total "mass"
        r : float/int or array of floats/int
            radial distance(s) where to compute the Hernquist profile. Unit must be the same as a.

    Return the Hernquist profile evaluated at the given distance(s). Unit is that of M/a^3.
    '''
    
    # Checking dtypes and values
    if isinstance(r, (float, int)):
        if   r<0:
            raise ValueError('r must be positive only. Cheers !')
        elif r==0:
            return np.inf
        
    elif isinstance(r, np.ndarray):
        if np.any(r)<0:
            raise ValueError('r must be positive only. Cheers !')
        elif np.any(r==0):
            mask = r==0
        
    else:
        raise TypeError('r must either be int or float, or a numpy array of the same types. Cheers !')
        
    if not isinstance(M, (int, float)):
        raise TypeError('M must be int or float only. Cheers !')
    if not isinstance(a, (int, float)):
        raise TypeError('a must be int or float only. Cheers !')
    
    if a<= 0:
        raise ValueError('a must be positive only. Cheers !')
        
    if   M==0:
        return 0*r
    elif M<0:
        raise ValueError('M must be positive only. Cheers !')
    
    out        = r*0
    out[mask]  = np.inf
    out[~mask] = (M*a/2/np.pi) / (r*(r+a)**3)
    
    return out


def nfw(r, c, rs):
    '''
    NFW profile (mass 3D distribution).

    Parameters
    ----------
        c : float/int
            halo concentration
        r : float/int or array of floats/int
            radial distance(s) where to compute the Hernquist profile. Unit must be the same as rs.
        rs : float/int
            scale radius
        
    Return the NFW profile evaluated at the given distance. Unit is that of a 3D mass density in SI (i.e. kg/m^3).
    '''
    
    # Checking dtypes and values
    if isinstance(r, (float, int)):
        if   r<0:
            raise ValueError('r must be positive only. Cheers !')
        elif r==0:
            return np.inf
    elif isinstance(r, np.ndarray):
        if np.any(r)<0:
            raise ValueError('r must be positive only. Cheers !')
        elif np.any(r==0):
            mask = r==0
    else:
        raise TypeError('r must either be int or float, or a numpy array of the same types. Cheers !')
        
    if not isinstance(c, (int, float)):
        raise TypeError('c must be int or float only. Cheers !')
    if not isinstance(rs, (int, float)):
        raise TypeError('rs must be int or float only. Cheers !')
    
    if c <= 0:
        raise ValueError('c must be positive only. Cheers !')
        
    if rs<=0:
        raise ValueError('rs must be positive only. Cheers !')
    
    deltaC     = (200/3) * c**3 / (np.log(1+c) - c/(1+c))
    rhoCrit    = 3*cosmo.H(0)/(8*np.pi*G)
    out        = r*0
    out[mask]  = np.inf
    out[~mask] = deltaC*rhoCrit / ((r/rs) * (1 + r/rs)**2)
    
    return out


def sersic_profile(r, n, re, Ie=None, bn=None, mag=None, offset=None):
    """
    General Sersic profile. Either computes it with n, re and Ie, or if Ie is not known but the total magnitude and the offset are, computes Ie from them and then return the profile.
        
    Mandatory inputs
    ----------------
        n : float/int
            Sersic index
        r : float
            the position at which the profile is computed
        re : float
            half-light radius
            
    Optional inputs
    ---------------
        bn : float
            bn factor appearing in the Sersic profile defined as $2\gamma(2n, bn) = \Gamma(2n)$. By default, bn is None, and its value will be computed by the function using the value of n. To skip this computation, please give a value to bn when callling the function.
        Ie : float
            intensity at half-light radius
        mag : float
            galaxy total integrated magnitude used to compute Ie if not given
        offset : float
            magnitude offset in the magnitude system used
                
    Return the profile value at position r.
    """
    
    bn, = check_bns([n], [bn])
    Ie  = checkAndComputeIe(Ie, n, bn, re, mag, offset)
        
    return Ie*np.exp( -bn*((r/re)**(1.0/n) - 1) )


####################################################################################################
#                                      2D modelling                                                #
####################################################################################################

def bulgeDiskOnSky(nx, ny, Rd, Rb, x0=None, y0=None, Id=None, Ib=None, magD=None, magB=None, offsetD=None, offsetB=None, inclination=0, PA=0, combine=True,
                   PSF={'name':'Gaussian2D', 'FWHMX':0.8, 'FWHMY':0.8, 'sigmaX':None, 'sigmaY':None, 'unit':'arcsec'}, noPSF=False, arcsecToGrid=0.03,
                   fineSampling=1, samplingZone={'where':'centre', 'dx':2, 'dy':2}, skipCheck=False, verbose=True):
    '''
    Generate a bulge + (sky projected) disk 2D model (with PSF convolution).
    
    How to use
    ----------
        Apart from the mandatory inputs, it is necessary to provide either an intensity at Re for each profile (Id, Ib), or if not known, a total magnitude value for each profile (magD, magB) and their corresponding magnitude offset (to convert from magnitudes to intensities).
       
    Caution
    -------
        Rd and Rb should be given in pixel units. If you provide them in arcsec, you must update the arcsecToGrid value to 1 (since 1 pixel will be equal to 1 arcsec). 
        
    Infos about sampling
    --------------------
        fineSampling parameter can be used to rebin the data. The shape of the final image will depend on the samplingZone used:
            - if the sampling is performed everywhere ('where' keyword in samplingZone equal to 'all'), the final image will have dimensions (nx*fineSampling, ny*fineSampling)
            - if the sampling is performed around the centre ('where' equal to 'centre'), the central part is over-sampled, but needs to be binned in the end so that pixels have the same size in the central part and around. Thus, the final image will have the dimension (nx, ny).        

    Mandatory inputs
    ----------------
        nx : int
            size of the model for the x-axis
        ny : int
            size of the model for the y-axis
        Rb : float
            bulge half-light radius. Best practice is to provide it in pixels (see Caution section).
        Rd : float
            disk half-light radius. Best practice is to provide it in pixels (see Caution section).
        
    Optional inputs
    ---------------
        arcsecToGrid : float
            pixel size conversion in arcsec/pixel, used to convert the FWHM/sigma from arcsec to pixel. Default is HST-ACS resolution of 0.03"/px.         
        Ib : float
            bulge intensity at (bulge) half-light radius. If not provided, magnitude and magnitude offset must be given instead.
        Id : float
            disk intensity at (disk) half-light radius. If not provided, magnitude and magnitude offset must be given instead.
        inclination : int/float
            disk inclination on sky. Generally given between -90° and +90°. Value must be given in degrees. Default is 0° so that no projection is performed.
        magB : float
            bulge total magnitude
        magD: float
            disk total magnitude
        offsetB : float
            bulge magnitude offset
        offsetD : float
            disk magnitude offset
        noPSF : bool
            whether to not perform PSF convolution or not. Default is to do convolution.
        PA : int/float
            disk position angle (in degrees). Default is 0° so that no rotation if performed.
        fineSampling : positive int
            fine sampling for the pixel grid used to make high resolution models. For instance, a value of 2 means that a pixel will be split into two subpixels. Default is 1.
        PSF : dict
            Dictionnary of the PSF (and its parameters) to use for the convolution. Default is a (0, 0) centred radial gaussian (muX=muY=0 and sigmaX=sigmaY) with a FWHM corresponding to that of MUSE (~0.8"~4 MUSE pixels).
            For now, only 2D Gaussians are accepted as PSF. 
        samplingZone : dict
            where to perform the over sampling. Default is everywhere. Dictionnaries should have the following keys:
                - 'where' : str; either 'all' to perform everywhere or 'centre' to perform around the centre
                - 'dx'    : int; x-axis maximum distance from the centre coordinate. A sub-array with x-axis values within [xpos-dx, xpos+dx] will be selected. If the sampling is performed everywhere, 'dx' does not need to be provided.
                - 'dy'    : int; y-axis maximum distance from the centre coordinate. A sub-array with y-axis values within [ypos-dy, ypos+dy] will be selected. If the sampling is performed everywhere, 'dy' does not need to be provided.
               
        skipCheck : bool
            whether to skip the checking part or not. Default is False.
        x0 : float/int
            x-axis centre position. Default is None so that nx//2 will be used.
        y0 : float/int
            y-axis centre position. Default is None so that ny//2 will be used.
        verbose : bool
            whether to print text on stdout or not. Default is True.
        
    Return X, Y grids and the total (sky projected + PSF convolved) model of the bulge + disk decomposition.
    '''
    
    ##############################################
    #          Checking input parameters         #
    ##############################################
    
    if not skipCheck:
        if not isinstance(samplingZone, dict) or 'where' not in samplingZone:
            print('sampling zone was not provided or syntax was incorrect. Thus, performing sampling (if relevant) on the full array.')
            samplingZone = {'where':'all'}
            
        if samplingZone['where'] not in ['all', 'centre']:
            raise ValueError("'where' keyword in samplingZone dictionnary should be either 'all' or 'centre'. Cheers !")
            
        if samplingZone['where']=='centre':
            if 'dx' not in samplingZone or 'dy' not in samplingZone:
                raise KeyError("'dx' and 'dy' keywords were missing in samplingZone dictionnary with 'where' keyword equal to 'centre'. Please provide values for the sampling box size around the centre. Cheers !")
            else:
                if not isinstance(samplingZone['dx'], (int, np.integer)) or not isinstance(samplingZone['dy'], (int, np.integer)):
                    raise TypeError("At least one of the following keys in samplingZone dictionnary was not given as an integer: 'dx' or 'dy'. Please provide these as int. Cheers !")
        
        if not isinstance(fineSampling, (int, np.integer)):
            raise TypeError('Given fine sampling parameter does not have the following type: int. Please provide integer only fineSampling values. Cheers !')
    
        if fineSampling < 1:
            raise ValueError('Fine sampling cannot be less than 1.')
         
        if any([i<0 for i in [nx, ny, Rd, Rb, arcsecToGrid]]):
            raise ValueError('At least one of the following parameters was provided as a negative number, which is not correct: nx, ny, Rd, Rb, arcsecToGrid.')
        
        # Checking PA
        if PA<-90 or PA>90:
            raise ValueError('PA should be given in the range -90° <= PA <= 90°, counting angles anti clock-wise (0° means major axis is vetically aligned). Cheers !')

    ##################################
    #         Compute models         #
    ##################################

    listn       = [1, 4]
    listbn      = [compute_bn(n) for n in listn]

    # Checking that we have the correct information to model correctly our data    
    if Id is None:
        if magD is not None and offsetD is not None:
            Id  = intensity_at_re(listn[0], magD, Rd, offsetD, bn=listbn[0])
        else:
            raise ValueError("Id is None, but magD or offsetD is also None. If no Id is given, please provide a value for the total magnitude and magnitude offset in order to compute the intensity. Cheers !")
    
    if Ib is None:
        if magB is not None and offsetB is not None:
            Ib  = intensity_at_re(listn[1], magB, Rb, offsetB, bn=listbn[1])
        else:
            raise ValueError("Ib is None, but magB or offsetB is also None. If no Ib is given, please provide a value for the total magnitude and magnitude offset in order to compute the intensity. Cheers !")

    X, Y, model = model2D(nx, ny, listn, [Rd, Rb], x0=x0, y0=y0, listIe=[Id, Ib], listInclination=[inclination, 0], listPA=[PA, 0], fineSampling=fineSampling, samplingZone=samplingZone, combine=combine)
    
    if not noPSF:
        # If we perform fine sampling only in the central part, model2D function rebins the data in the end, so the arcsec to pixel conversion factor does not need to be updated since we do not have a finer pixel scale
        if samplingZone['where'] == 'centre':
            fineSampling   = 1
        
        if combine:
            model          = PSFconvolution2D(model, model=PSF, arcsecToGrid=arcsecToGrid/fineSampling, verbose=verbose)
        else:
            for pos, mod in enumerate(model):
                model[pos] = PSFconvolution2D(mod, model=PSF, arcsecToGrid=arcsecToGrid/fineSampling, verbose=verbose)
                
    return X, Y, model


def model2D(nx, ny, listn, listRe, x0=None, y0=None, listIe=None, listMag=None, listOffset=None, listInclination=None, listPA=None, combine=True, 
            fineSampling=1, samplingZone={'where':'centre', 'dx':5, 'dy':5}, skipCheck=False):
    """
    Generate a (sky projected) 2D model (image) of a sum of Sersic profiles. Neither PSF smoothing, nor projections onto the sky whatsoever are applied here.
    
    How to use
    ----------
        Apart from the mandatory inputs, it is necessary to provide either an intensity at Re for each profile (listIe), or if not known, a total magnitude value for each profile (listMag) and their corresponding magnitude offset (to convert from magnitudes to intensities).
        The 'combine' keywork can be set to False to recover the model of each component separately.
    
    Infos about sampling
    --------------------
        fineSampling parameter can be used to rebin the data. The shape of the final image will depend on the samplingZone used:
            - if the sampling is performed everywhere ('where' keyword in samplingZone equal to 'all'), the final image will have dimensions (nx*fineSampling, ny*fineSampling)
            - if the sampling is performed around the centre ('where' equal to 'centre'), the central part is over-sampled, but needs to be binned in the end so that pixels have the same size in the central part and around. Thus, the final image will have the dimension (nx, ny).
    
    Mandatory inputs
    ----------------
        listn : list of float/int
            list of Sersic index for each profile
        listRe : list of float
            list of half-light radii for each profile
        nx : int
            size of the model for the x-axis
        ny : int
            size of the model for the y-axis
        
    Optional inputs
    ---------------v      
        combine : bool
            whether to combine (sum) all the components and return a single intensity map, or to return each component separately in lists. Default is to combine all the components into a single image.
        listIe : list of floats
            list of intensities at re for each profile
        listInclination : list of float/int
            lsit of inclination of each Sersic component on the sky in degrees. Default is None so that each profile is viewed face-on.
        listMag : list of floats
            list of total integrated magnitudes for each profile
        listOffset : list of floats
             list of magnitude offsets used in the magnitude system for each profile
        listPA : list of float/int
            list of position angle of each Sersic component on the sky in degrees. Generally, these values are given between -90° and +90°. Default is None, so that no rotation is applied to any component.
        fineSampling : positive int
            fine sampling for the pixel grid used to make high resolution models. For instance, a value of 2 means that a pixel will be split into two subpixels. Default is 1.
        samplingZone : dict
            where to perform the sampling. Default is everywhere. Dictionnaries should have the following keys:
                - 'where' : either 'all' to perform everywhere or 'centre' to perform around the centre
                - 'dx' : int, x-axis maximum distance from the centre coordinate. An sub-array with x-axis values within [xpos-dx, xpos+dx] will be selected. If the sampling is performed everywhere, 'dx' does not need to be provided.
                - 'dy' : int, y-axis maximum distance from the centre coordinate. An sub-array with y-axis values within [ypos-dy, ypos+dy] will be selected. If the sampling is performed everywhere, 'dy' does not need to be provided.
          
        skipCheck : bool
            whether to skip the checking part or not. Default is False.
        x0 : float/int
            x-axis centre position. Default is None so that nx//2 will be used.
        y0 : float/int
            y-axis centre position. Default is None so that ny//2 will be used.
        
    Return:
        - if combine is True: X, Y grids and the intensity map
        - if combine is False: X, Y grids and a list of intensity maps for each component
    """
    
    def computeSersic(X, Y, nbModels, listn, listRe, listIe, listInclination, listPA):
        
        # We need not specify a centre coordinate offset, because the X and Y grids are automatically centred on the real centre.
        # If we combine models, we add them, if we do not combine them, we place them into a list
        for pos, n, re, ie, inc, pa in zip(range(nbModels), listn, listRe, listIe, listInclination, listPA):
            
            # We add 90 to PA because we want a PA=0° galaxy to be aligned with the vertical axis
            ell                = 1-np.cos(inc*np.pi/180)
            pa                *= np.pi/180
            theModel           = Sersic2D(amplitude=ie, r_eff=re, n=n, x_0=0, y_0=0, ellip=ell, theta=np.pi/2+pa)
            
            if pos == 0:
                if combine:
                    intensity  = theModel(X, Y)/(1-ell)
                else:
                    intensity  = [theModel(X, Y)/(1-ell)]
            else:
                if combine:
                    intensity += theModel(X, Y)/(1-ell)
                else:
                    intensity += [theModel(X, Y)/(1-ell)]
        return intensity
    
    ##############################################
    #          Checking input parameters         #
    ############################################## 
    
    if not skipCheck:
        if not isinstance(samplingZone, dict) or 'where' not in samplingZone:
            print('sampling zone was not provided or syntax was incorrect. Thus, performing sampling (if relevant) on the full array.')
            samplingZone = {'where':'all'}
            
        if samplingZone['where'] not in ['all', 'centre']:
            raise ValueError("'where' keyword in samplingZone dictionnary should be either 'all' or 'centre'. Cheers !")
            
        if samplingZone['where']=='centre':
            if 'dx' not in samplingZone or 'dy' not in samplingZone:
                raise KeyError("'dx' and 'dy' keywords were missing in samplingZone dictionnary with 'where' keyword equal to 'centre'. Please provide values for the sampling box size around the centre. Cheers !")
            else:
                if not isinstance(samplingZone['dx'], (int, np.integer)) or not isinstance(samplingZone['dy'], (int, np.integer)):
                    raise TypeError("At least one of the following keys in samplingZone dictionnary was not given as an integer: 'dx' or 'dy'. Please provide these as int. Cheers !")
        
        if not isinstance(fineSampling, (int, np.integer)) or not isinstance(nx, (int, np.integer)) or not isinstance(ny, (int, np.integer)):
            raise TypeError('One of the following parameter is not an integer, which is not valid: fineSampling (%s), nx (%s), ny (%s).' %(type(fineSampling), type(nx), type(ny)))
        
        if fineSampling < 1:
            raise ValueError('Fine sampling cannot be less than 1.')
        
        # Checking PA
        if any([pa<-90 for pa in listPA]) or any([pa>90 for pa in listPA]):
            raise ValueError('PA should be given in the range -90° <= PA <= +90°, counting angles anti clock-wise. Cheers !')  

    if listIe is None:
        if listMag is not None and listOffset is not None:
            listIe         = intensity_at_re(np.array(listn), np.array(listMag), np.array(listRe), np.array(listOffset))
        else:
            raise ValueError("listIe is None, but listMag or listOffset is also None. If no listIe is given, please provide a value for the total magnitude and magnitude offset in order to compute the intensities. Cheers !")

    nbModels               = len(listn)
    if listInclination is None:
        listInclination    = [0]*nbModels
    if listPA          is None:
        listPA             = [0]*nbModels
    
    ##################################
    #         Compute models         #
    ##################################

    # Define image centre
    midX                   = nx//2
    midY                   = ny//2
    
    if x0 is None:
        x0                 = midX
    if y0 is None:
        y0                 = midY
        
    # Pixel width is not 1 if we use fineSampling
    pixWidth               = 1.0/fineSampling
    pixHeight              = 1.0/fineSampling
    
    # We centre the coordinate X and Y grids to the given centre coordinates
    # The centre is recentred inside an 'original' pixel because of fine sampling (to not break any symmetry when rebinning)
    if samplingZone['where'] == 'all':
        newX0              = x0 + (1-pixWidth)/2
        newY0              = y0 + (1-pixHeight)/2
        listX              = np.arange(0, nx, pixWidth)  - newX0
        listY              = np.arange(0, ny, pixHeight) - newY0
        X, Y               = np.meshgrid(listX, listY)
        intensity          = computeSersic(X, Y, nbModels, listn, listRe, listIe, listInclination, listPA)/(fineSampling**2)
        
        # Rebinning intensity map in the central part
        '''
        intensity   = intensity.reshape(int(intensity.shape[0] / fineSampling), fineSampling, int(intensity.shape[1] / fineSampling), fineSampling)
        intensity   = intensity.mean(1).mean(2)
        
        listX              = np.arange(0, nx, 1) - x0
        listY              = np.arange(0, ny, 1) - y0
        X, Y               = np.meshgrid(listX, listY)
        '''
        
    else:
        # We generate grids with pixel size of 1x1 (and we centre it on the galaxy centre)
        listX              = np.arange(0, nx, 1) - x0
        listY              = np.arange(0, ny, 1) - y0
        X, Y               = np.meshgrid(listX, listY)
        intensity          = computeSersic(X, Y, nbModels, listn, listRe, listIe, listInclination, listPA)
        
        # We generate a subarray around the centre in the given box, using the given over-sampling factor
        maxX               = samplingZone['dx'] + 0.5 -1.0/(2*fineSampling) # Weird but that's what comes out of a few diagrams
        maxY               = samplingZone['dy'] + 0.5 -1.0/(2*fineSampling)
        listXcenPart       = np.arange(-maxX, maxX+pixWidth, pixWidth)
        listYcenPart       = np.arange(-maxY, maxY+pixHeight, pixHeight)
        
        XcenPart, YcenPart = np.meshgrid(listXcenPart, listYcenPart)
        intensityCenPart   = computeSersic(XcenPart, YcenPart, nbModels, listn, listRe, listIe, listInclination, listPA)

        # Rebinning intensity map in the central part
        intensityCenPart   = intensityCenPart.reshape(int(intensityCenPart.shape[0] / fineSampling), fineSampling, int(intensityCenPart.shape[1] / fineSampling), fineSampling)
        intensityCenPart   = intensityCenPart.sum(1).sum(2)/(fineSampling**2)
        
        # Combining back the central part into the original array
        intensity[y0-samplingZone['dy']:y0+samplingZone['dy']+1, x0-samplingZone['dx']:x0+samplingZone['dx']+1] = intensityCenPart
        
    return X, Y, intensity





