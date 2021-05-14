#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Miscellaneous functions related to galaxy computations.
"""

import scipy.ndimage                      as     nd
import astropy.units                      as     u
import numpy                              as     np
from   numpy.fft                          import fft2, ifft2
from   scipy.special                      import gammaincinv, gamma, gammainc

#####################################################################
#                 Sersic profiles related functions                 #
#####################################################################

def checkAndComputeIe(Ie, n, bn, re, mag, offset, noError=False):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Check whether Ie is provided. If not, but the magnitude and magnitude offset are, it computes it.

    :param bn: bn factor appearing in the Sersic profile (see :py:func:`compute_bn`)
    :type bn: float or ndarray[float]
    :param float Ie: intensity at effective radius
    :type Ie: float or ndarray[float]
    :param mag: total integrated magnitude
    :type mag: float or ndarray[float]
    :param n: Sersic index
    :type n: int or float or list[int] or list[float] or ndarray[int] or ndarray[float]
    :param offset: magnitude offset
    :type offset: float or ndarray[float]
    :param re: half-light/effective radius
    :type re: float or ndarray[float]

    :param bool noError: (**Optional**) whether to not raise an error or not when data is missing to compute the intensity. If True, None is returned.

    :returns: Ie if it could be computed or already existed, or None if **noError** flag is set to True.
    :rtype: float or ndarray[float] or None

    :raises ValueError: if neither Ie, nor mag and offset are provided
    """
    
    if Ie is None:
        if mag is not None and offset is not None:
            return intensity_at_re(n, mag, re, offset, bn=bn)
        else:
            if noError:
                return None
            else:
                raise ValueError("Ie value is None, but mag and/or offset is/are also None. If no Ie is given, please provide a value for the total magnitude and magnitude offset in order to compute the former one. Cheers !")
    else:
        return Ie
    
def intensity_at_re(n, mag, re, offset, bn=None):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the intensity of a given Sersic profile with index n at the position of the half-light radius re. This assumes to know the integrated magnitude of the profile, as well as the offset used for the magnitude definition.
    
    :param mag: total integrated magnitude of the profile
    :type mag: float or ndarray[float]
    :param n: Sersic index of the given profile
    :type n: int or float or list[int] or list[float] or ndarray[int] or ndarray[float]
    :param offset: magnitude offset used in the defition of the magnitude system
    :type offset: float or ndarray[float]
    :param re: effective (half-light) radius
    :type re: float or ndarray[float]

    :param bn: (**Optional**) bn factor appearing in the Sersic profile defined as $2\gamma(2n, bn) = \Gamma(2n)$. If None, its value will be computed from n.
    :type bn: float or ndarray[float]
    :returns: the intensity at re
    :rtype: float or ndarray[float]
    """
 
    if bn is None:
        bn = compute_bn(n)
    
    return 10**((offset - mag)/2.5) * bn**(2*n) / (2*np.pi*n*gamma(2*n) * re**2 * np.exp(bn))


def check_bns(listn, listbn):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Given a list of bn values, check those which are not given (i.e. equal to None), and compute their value using the related Sersic index.

    :param list[float] listbn: list of bn values
    :param listn: list of Sersic indices
    :type listn: list[int] or list[float]
    :returns: list of bn values
    :rtype: list[float]
    """
    
    return [compute_bn(listn[pos]) if listbn[pos] is None else listbn[pos] for pos in range(len(listn))]


def compute_bn(n):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the value of bn used in the definition of a Sersic profile
    
    .. math::
            
        2\gamma(2n, b_n) = \Gamma(2n)

    :param n: Sersic index of the profile
    :type n: int or float or list[int] or list[float] or ndarray[int] or ndarray[float]
    :returns: bn
    :rtype: float
    """
    
    try:
        res = [gammaincinv(2*i, 0.5) for i in n]
    except:
        res = gammaincinv(2*n, 0.5)
    return res


#####################################################
#                       Other                       #
#####################################################

def realGammainc(a, x):
    ''''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Unnormalised lower incomplete gamma function
    
    .. math::
        
        \gamma(a, x) = \int_0^x dt~t^{a-1} e^{-t}
    
    :param a: power of the gamma function
    :type a: int or float
    :param x: upper bound of the integral
    :type x: int or float
    :returns: the incomplete gamma function
    :rtype: float
    '''
    
    return gamma(a) * gammainc(a, x)


def PSFconvolution2D(data, arcsecToGrid=0.03, model={'name':'Gaussian2D', 'FWHMX':0.8, 'FWHMY':0.8, 'sigmaX':None, 'sigmaY':None, 'unit':'arcsec'}, verbose=True):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Convolve using fast FFT a 2D array with a pre-defined (2D) PSF.
    
    .. note::

        **How to use**
        
        Best practice is to provide the PSF model FWHM or sigma in arcsec, and the image pixel element resolution (arcsecToPixel), so that the fonction can convert correctly the PSF width in pixels.
        You can provide either the FWHM or sigma. If both are given, sigma is used.

    :param data: data to be convolved with the PSF
    :type data: 2D ndarray
        
    :param float arcsecToGrid: (**Optional**) conversion factor from arcsec to the grid pixel size, that is the width and height of a single pixel in the X and Y grids. Default is the pixel size of HST-ACS images (0.03"/px).
    :param dict model: (**Optional**) dictionnary of the PSF (and its parameters) to use for the convolution. Default is a (0, 0) centred radial gaussian (muX=muY=0 and sigmaX=sigmaY) with a FWHM corresponding to that of MUSE (~0.8"~4 MUSE pixels). For now, only 2D Gaussians are accepted as PSF.
    :param bool verbose: (**Optional**) whether to print text on stdout or not

    :returns: a new convolved image
    :rtype: 2D ndarray[float]
    
    :raises ValueError: 
        
        * if unit key in **model** dict is not a valid astropy unit 
        * if the PSF is not a 2D Gaussian model
    '''
    
    def setListFromDict(dictionary, keys=None, default=None):
        """
        Fill a list with values from a dictionary or from default ones if the given key is not in the dictionary.
        
        Mandatory inputs
        ----------------
            dictionary : dict
                dictionary to get the keys values from
        
        Optional inputs
        ---------------
            default : list
                list of default values if given key is not in dictionary
            keys : list of str
                list of key names whose values will be appended into the list
                
        Return a list with values retrived from a dictionary keys or from a list of default values.
        """
        
        out = []
        for k, df in zip(keys, default):
            if k in dictionary:
                out.append(dictionary[k])
            else:
                out.append(df)
        return out

    if verbose:
        print('Convoluting using PSF model %s' %model)
    
    # Retrieve PSF parameters and set default values if not provided
    if model['name'].lower() == 'gaussian2d':
        sigmaX, sigmaY, FWHMX, FWHMY, unit = setListFromDict(model, keys=['sigmaX', 'sigmaY', 'FWHMX', 'FWHMY', 'unit'], default=[None, None, 0.8, 0.8, "arcsec"])
        
        # Compute sigma from FWHM if sigma is not provided
        if sigmaX is None:
            sigmaX = FWHMX / (2*np.sqrt(2*np.log(2)))
        if sigmaY is None:
            sigmaY = FWHMY / (2*np.sqrt(2*np.log(2)))
            
        # Convert sigma to arcsec and then go in grid pixel values
        try:
            sigmaX = u.Quantity(sigmaX, unit).to('arcsec').value / arcsecToGrid
            sigmaY = u.Quantity(sigmaY, unit).to('arcsec').value / arcsecToGrid
        except ValueError:
            raise ValueError('Given unit %s is not valid. Angles may generally be given in units of deg, arcmin, arcsec, hourangle, or rad.')
        
        # Perform convolution
        image     = data.copy()
        image     = fft2(image)
        image     = nd.fourier_gaussian(image, sigma=[sigmaX, sigmaY])
        image     = ifft2(image).real
    else:
        raise ValueError('Given model %s is not recognised or implemented yet. Only gaussian2d model is accepted.' %model['name'])
        
    return image
    
def fromStructuredArrayOrNot(gal, magD, magB, Rd, Rb, noStructuredArray):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Store values of the disk magnitude, the bulge magnitude, the disk effective radius and the bulge effective radius either directly from the given values or from a numpy structured array.

    :param gal: structured array with data for all the galaxies. The required column names are 'R_d_GF' (re for the disk component), 'R_b_GF' (re for the bulge component), 'Mag_d_GF' (the total integrated magnitude for the disk component), 'Mag_b_GF' (the total integrated magnitude for the bulge component).
    :type gal: structured ndarray
    :param magB: total magnitude of the bulge component of the galaxies
    :type magB: float or list(float)
    :param magD: total magnitude of the disk component of the galaxies
    :type magD: float or list(float)
    :param Rb: half-light radius of the bulge components of the galaxies
    :type Rb: float or list(float)
    :param Rd: half-light radius of the disk components of the galaxies
    :type Rd: float or list(float)
    :param bool noStructuredArray: if False, the structured array gal will be used. If False, values of the magnitudes and half-light radii of the two components must be given.

    :returns: the disk magnitude, the bulge magnitude, the disk effective radius and the bulge effective radius
    :rtype: float, float, float, float
    """
    #if no structured array is given, check that each necessary variable is given instead
    if noStructuredArray:
        if np.any([i is None for i in [magD, magB, Rd, Rb]]):
            print("ValueError: a None found. One of the inputs in the list magD, magB, Rd, Rb was not provided. If you are not using the structured array feature, you must provide magnitude and effective values for all galaxies and for both profiles (exponential disk and bulge)")
            return None
        else:
            magD     = np.array(magD)
            magB     = np.array(magB)
            Rd       = np.array(Rd)
            Rb       = np.array(Rb)
    #if a structured array is given, retrieve the correct columns
    else:
        magD         = gal['Mag_d_GF']
        magB         = gal['Mag_b_GF']
        Rd           = gal['R_d_GF']
        Rb           = gal['R_b_GF']
        
    return magD, magB, Rd, Rb


####################################################################################################
#                                            Trash                                                 #
####################################################################################################

def mergeModelsIntoOne(listX, listY, listModels, pixWidth, pixHeight, xlim=None, ylim=None):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Sum the contribution of different models with distorted X and Y grids into a single image with a regular grid.
    
    .. deprecated::
        
        This function is deprecated and may be removed soon.
    
    .. note::
    
        **How to use**
        
        Provide the X and Y grids for each model as well as the intensity maps.
        
    .. warning::
        
        pixWidth and pixHeight parameters are quite important as they will be used to define the pixel scale of the new regular grid. Additonally, all data points of every model falling inside a given pixel will be added to this pixel contribution.
        
        In principle, one should give the pixel scale of the original array (before sky projection was applied).

    :param list[2D ndarray] listX: list of X arrays for each model
    :param list[2D ndarray] listY: list of Y arrays for each model
    :param list[2D ndarray] listModels: list of intensity maps for each model
    :param float pixWidth: width (x-axis size) of a single pixel. With numpy meshgrid, it is possible to generate grids with Nx pixels between xmin and xmax values, so that each pixel would have a width of (xmax-xmin)/Nx.
    :param float pixHeight: height (y-axis size) of a single pixel. With numpy meshgrid, it is possible to generate grids with Ny pixels between ymin and ymax values, so that each pixel would have a height of (ymax-ymin)/Ny.

    :param 2-tuple[float] xlim: (**Optional**) lower and upper bounds (in that order) for the x-axis. If None, a symmetric grid in X is done, using the maximum absolute X value found.
    :param 2-tuple[float] ylim: (**Optional**) lower and upper bounds (in that order) for the y-axis. If None, a symmetric grid in Y is done, using the maximum absolute Y value found.
    
    :results: a new X grid, a new Y grid and a new model intensity map with the contribution of every model summed up
    :rtype: 2D ndarray, 2D ndarray, 2D ndarray
    
    :raises TypeError: if xlim or ylim are neither None, nor a tuple, nor a list
    '''
    
    # Generate the new X and Y grid
    if xlim is None:
        maxX = np.max([np.nanmax(listX), -np.nanmin(listX)])
        minX = -maxX
    elif type(xlim) in [tuple, list]:
        minX = min(xlim)
        maxX = max(xlim)
    else:
        raise TypeError('Unvalid type (%s) for xlim. If you provide values for xlim, please give it as a list or a tuple only. Cheers !' %type(xlim))
        
    if ylim is None:
        maxY = np.max([np.nanmax(listY), -np.nanmin(listY)])
        minY = -maxY
    elif type(ylim) in [tuple, list]:
        minY = min(ylim)
        maxY = max(ylim)
    else:
        raise TypeError('Unvalid type (%s) for ylim. If you provide values for ylim, please give it as a list or a tuple only. Cheers !' %type(ylim))

    nx   = 1 + int((maxX-minX)/pixWidth)
    ny   = 1 + int((maxY-minY)/pixHeight)
    x    = np.linspace(minX, maxX, nx)
    y    = np.linspace(minY, maxY, ny)
    X, Y = np.meshgrid(x, y)
    
    # We have roundoff errors in our X and Y grids, so we need to round off to the precision of pixWidth or pixHeight
    rnd  = np.min([-int(("%e" %pixWidth).split('e')[-1]), -int(("%e" %pixHeight).split('e')[-1])])
    X    = np.round(X, rnd)
    Y    = np.round(Y, rnd)
    
    # Combine data (3 loops, not very efficient...)
    Z    = X.copy()*0.0
    shp  = np.shape(X)
    
    rg0  = range(shp[0])
    rg1  = range(shp[1])
    
    for Xmodel, Ymodel, model in zip(listX, listY, listModels):
        Xmodel         = np.round(Xmodel, rnd)
        Ymodel         = np.round(Ymodel, rnd)
        
        for xpos in rg0:
            for ypos in rg1:
                Xmask          = np.logical_and(Xmodel>=X[xpos, ypos], Xmodel<X[xpos, ypos] + pixWidth)
                Ymask          = np.logical_and(Ymodel>=Y[xpos, ypos], Ymodel<Y[xpos, ypos] + pixHeight)
                mask           = np.logical_and(Xmask, Ymask)
                Z[xpos, ypos] += np.sum(model[mask])
    
    return X, Y, Z


def projectModel2D(model, inclination=0, PA=0, splineOrder=3, fillVal=0):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Project onto the sky a 2D model of a galaxy viewed face-on.
    
    .. deprecated::
        
        This function is deprecated and may be removed soon.
    
    .. note::
        
        Sky projection is used with scipy ndimage functions. Two projections are applied (in this order):
            - inclination: the 2D model is rotated along the vertical axis passing through the centre of the image
            - PA rotation: the (inclined) model is rotated clock-wise in the sky plane
        
        If you do not desire to apply one of the following coordinate transform, either do not provide it, or let it be 0.
        By default, no transform whatsoever is applied.
    
    :param model: intensity map of the model
    :type model: 2D ndarray

    :param inclination: (**Optional**) inclination of the galaxy on the sky in **degrees**
    :type inclination: int or float
    :param fillVal: (**Optional**) value used to filled pixels with missing data
    :type fillVal: int or float
    :param PA: (**Optional**) position angle of the galaxy on the sky in degrees. Generally, this number is given between -90° and +90°
    :type PA: int or float
    :param int splineOrder: (**Optional**) order of the spline used to interpolate values at new sky coordinate
    
    :returns: a new image (intensity map) projected onto the sky with interpolated values at new coordinate location
    :rtype: 2D ndarray
    
    :raises ValueError: if PA not in [-90°, +90°] range or if model image is not 2-dimensional
    '''
    
    # Checking PA
    if PA<-90 or PA>90:
        raise ValueError('PA should be given in the range -90° <= PA <= 90°, counting angles anti clock-wise. Cheers !')

    if model.ndim != 2:
        raise ValueError('Model should be 2-dimensional only. Current number of dimensions is %d. Cheers !' %model.ndim)
    
    newModel     = model.copy()
    # We do not use the notation X *= something for the arrays because of cast issues when having X and Y gris with numpy int values instead of numpy floats
    if inclination != 0:
        newModel = tiltGalaxy(newModel, inclination=inclination, splineOrder=splineOrder, fillVal=fillVal)
    
    # PA rotation (positive PA means rotating anti clock-wise)
    if PA != 0:
        newModel = rotateGalaxy(newModel, PA=PA, splineOrder=splineOrder, fillVal=fillVal)
    
    return newModel


def tiltGalaxy(model, inclination=0, splineOrder=3, fillVal=np.nan):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Tilt a galaxy image around the South-North axis (assumed vertical) passing through the image centre.
    
    .. deprecated::
        
        This function is deprecated and may be removed soon.

    :param model: intensity map of the galaxy model
    :type model: 2D ndarray

    :param fillVal: (**Optional**) value to fill pixels which may become empty
    :type fillVal: int or float
    :param bool fitIn: (**Optional**) whether to resize the image to fit it in or not. If False, a new image is generated with the same dimensions.
    :param inclination: (**Optional**) rotation angle to apply (counted clock-wise in degrees). Is 0, no rotation is applied.
    :param int splineOrder: (**Optional**) order of the spline used to compute the intensity value at new pixel location
    
    :returns: a new image of a galaxy tilted by some angle around the vertical axis passing through the image centre
    :rtype: 2D ndarray
    
    :raises ValueError: if model image is not 2-dimensional
    '''
    
    if model.ndim != 2:
        raise ValueError('Model should be 2-dimensional only. Current number of dimensions is %d. Cheers !' %model.ndim)
    
    # Apply coordinate transform on grid
    inclination   *= np.pi/180
    X, Y           = np.indices(model.shape)
    midX           = (np.nanmax(X) + np.nanmin(X))/2.0
    
    if inclination != 0:
        X[X>midX]  = midX + (X[X>midX]-midX)*np.sin(np.pi/2 + inclination)
        X[X<=midX] = midX - (X[X<=midX]-midX)*np.sin(3*np.pi/2 + inclination)
    
    # Apply coordinate transform on image then using the grid
    return nd.map_coordinates(model, [X, Y], order=splineOrder, cval=fillVal)


def rotateGalaxy(model, PA=0, splineOrder=3, fitIn=False, fillVal=np.nan):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Apply PA rotation to a galaxy image.
    
    .. deprecated::
        
        This function is deprecated and may be removed soon.

    :param model: intensity map of the galaxy model
    :type model: 2D ndarray

    :param fillVal: (**Optional**) value to fill pixels which may become empty
    :type fillVal: int or float
    :param bool fitIn: (**Optional**) whether to resize the image to fit it in or not. If False a new image is generated with the same dimensions.
    :param PA: (**Optional**) rotation angle to apply (counted clock-wise in degrees). Default is 0 so that no rotation is applied.
    :type PA: int or float
    :param int splineOrder: (**Optional**) order of the spline used to compute the intensity value at new pixel location
    
    :returns: a new image of a galaxy rotated by a certain angle
    :rtype: 2D ndarray
    
    :raises ValueError: if model image is not 2-dimensional
    '''
    
    if model.ndim != 2:
        raise ValueError('Model should be 2-dimensional only. Current number of dimensions is %d. Cheers !' %model.ndim)
    
    # We count angles anti clock-wise so we need to inverte the value for rotate which counts clock-wise
    return nd.rotate(model, -PA, reshape=fitIn, order=splineOrder, cval=fillVal)
