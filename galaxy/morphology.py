#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Computations relative to galaxies morphology.
"""

import numpy                              as     np
from   astropy.units.quantity             import Quantity
from   scipy.special                      import gammainc
from   scipy.optimize                     import root
from   scipy.integrate                    import quad
from   math                               import factorial, ceil
from   .models                            import sersic_profile, bulgeDiskOnSky
from   .misc                              import check_bns, compute_bn, realGammainc, checkAndComputeIe, intensity_at_re, fromStructuredArrayOrNot
from   .symlinks.coloredMessages          import errorMessage, brightMessage


#################################################################################################################
#                                           Sersic luminosities                                                 #
#################################################################################################################

def analyticFluxFrom0(r, n, re, bn=None, Ie=None, mag=None, offset=None, start=0.0):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Analytically compute the integrated flux from 0 up to radius r for a Sersic profile of index n.
    
    .. note::
    
        If no Ie is given, values for mag and offset must be given instead. 

    :param n: Sersic index of the profile
    :type n: int or float
    :param r: radius up to which the integral is computed. If a list is given, the position will be computed at each radius in the list.
    :type r: float or list[float]
    :param float re: half-light radius
        
    :param float bn: (**Optional**) bn factor appearing in the Sersic profile defined as 
        
        .. math::
            
            2 \gamma(2n, b_n) = \Gamma(2n).
            
    :param float Ie: (**Optional**) intensity at half-light radius
    :param float mag: (**Optional**) total magnitude used to compute Ie if not given
    :param float offset: (**Optional**) magnitude offset in the magnitude system used
    
    :returns: analytical flux from 0 to r (value (value) and its error (err) as the dictionary {'value':value, 'error':err})
    :rtype: dict
    
    :raises ValueError: if **r** is neither int, nor float
    """
    
    # Compute bn and Ie
    bn,       = check_bns([n], [bn])
    Ie        = checkAndComputeIe(Ie, n, bn, re, mag, offset)
    if Ie is None:
        raise ValueError('Either Ie must be given or mag and offset.')
    
    if isinstance(r, (int, float)):
        value = 2*np.pi*n*Ie*re**2 * np.exp(bn) * realGammainc(2*n, bn*(r/re)**(1.0/n)) / (bn**(2*n))
        return {'value':value, 'error':0}
    
    else:
        value     = []
        error     = [0]*len(r)
        for rval in r:
            value.append(2.0*np.pi*n*Ie*re**2 * np.exp(bn) * realGammainc(2*n, bn*(rval/re)**(1.0/n)) / (bn**(2*n)))
        
        return {'value':value, 'error':error}
    

def BoverD(r, rd, rb, b1=None, b4=None, Ied=None, Ieb=None, magD=None, magB=None, offsetD=None, offsetB=None, noError=False):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the ratio of the bulge flux (B) over the disk one (D) for a bulge-disk galaxy up to radius r.
    
    .. note::
    
        If no Ie is given, values for mag and offset must be given instead. 
    
    :param r: radius up to which the integral is computed. If a list is given, the position will be computed at each radius in the list.
    :type r: float or list[float]
    :param float rb: half-light radius of the bulge
    :param float rd: half-light radius of the disk
                
    :param float b1: (**Optional**) b1 factor appearing in the disk profile defined as 
        
        .. math::
            
            2 \gamma(2, b_1) = 1.
            
    :param float b4: (**Optional**) b4 factor appearing in the disk profile defined as 
        
        .. math::
            
            2 \gamma(8, b_4) = 7!.

    :param float Ied: (**Optional**) intensity of the disk at half-light radius
    :param float Ied: (**Optional**) intensity of the bulge at half-light radius
    :param float magD: (**Optional**) total magnitude of the disk used to compute Ied if not given
    :param float magB: (**Optional**) total magnitude of the bulge used to compute Ieb if not given
    :param bool noError: (**Optional**) whether to not raise an error or not if one of the Ie values could not be computed correctly. If set to True, np.nan is returned.
    :param float offsetB: (**Optional**) magnitude offset in the magnitude system used for the bulge component
    :param float offsetD: (**Optional**) magnitude offset in the magnitude system used for the disk component
            
    :returns: B/D ratio at all the given positions or np.nan if one of the intensities could not be computed correctly
    :rtype: float or list[float]
    """
    
    #compute b1 and b4 if not given
    b1, b4 = check_bns([1, 4], [b1, b4])
    Ied    = checkAndComputeIe(Ied, 1, b1, rd, magD, offsetD, noError=noError)
    Ieb    = checkAndComputeIe(Ieb, 4, b4, rb, magB, offsetB, noError=noError)
    
    if None in [Ieb, Ied]:
        return np.nan
        
    return fluxSersic(r, 4, rb, bn=b4, Ie=Ieb)['value'] / fluxSersic(r, 1, rd, bn=b1, Ie=Ied)['value']    


def BoverT(r, rd, rb, b1=None, b4=None, Ied=None, Ieb=None, magD=None, magB=None, offsetD=None, offsetB=None, noError=False):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Computes the ratio of the bulge flux (B) over the total one (T=D+B) for a bulge-disk galaxy up to radius r.
    
    .. note::
    
        If no Ie is given, values for mag and offset must be given instead. 
    
    :param r: radius up to which the integral is computed. If a list is given, the position will be computed at each radius in the list.
    :type r: float or list[float]
    :param float rb: half-light radius of the bulge
    :param float rd: half-light radius of the disk
                
    :param float b1: (**Optional**) b1 factor appearing in the disk profile defined as 
        
        .. math::
            
            2 \gamma(2, b_1) = 1.
            
    :param float b4: (**Optional**) b4 factor appearing in the disk profile defined as 
        
        .. math::
            
            2 \gamma(8, b_4) = 7!.

    :param float Ied: (**Optional**) intensity of the disk at half-light radius
    :param float Ied: (**Optional**) intensity of the bulge at half-light radius
    :param float magD: (**Optional**) total magnitude of the disk used to compute Ied if not given
    :param float magB: (**Optional**) total magnitude of the bulge used to compute Ieb if not given
    :param bool noError: (**Optional**) whether to not raise an error or not if one of the Ie values could not be computed correctly. If set to True, np.nan is returned.
    :param float offsetB: (**Optional**) magnitude offset in the magnitude system used for the bulge component
    :param float offsetD: (**Optional**) magnitude offset in the magnitude system used for the disk component
            
    :returns: B/T ratio at all the given positions or np.nan if one of the intensities could not be computed correctly
    :rtype: float or list[float]
    """
    
    #compute b1 and b4 if not given
    b1, b4 = check_bns([1, 4], [b1, b4])
    Ied    = checkAndComputeIe(Ied, 1, b1, rd, magD, offsetD, noError=noError)
    Ieb    = checkAndComputeIe(Ieb, 4, b4, rb, magB, offsetB, noError=noError)
    
    if None in [Ied, Ieb]:    
        return np.nan
        
    return fluxSersic(r, 4, rb, bn=b4, Ie=Ieb)['value'] / fluxSersics(r, [1, 4], [rd, rb], listbn=[b1, b4], listIe=[Ied, Ieb])['value']    


def DoverT(r, rd, rb, b1=None, b4=None, Ied=None, Ieb=None, magD=None, magB=None, offsetD=None, offsetB=None):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Computes the ratio of the disk flux (D) over the total one (T=D+B) for a bulge-disk galaxy up to radius r.
    
    .. note::
    
        If no Ie is given, values for mag and offset must be given instead. 
    
    :param r: radius up to which the integral is computed. If a list is given, the position will be computed at each radius in the list.
    :type r: float or list[float]
    :param float rb: half-light radius of the bulge
    :param float rd: half-light radius of the disk
                
    :param float b1: (**Optional**) b1 factor appearing in the disk profile defined as 
        
        .. math::
            
            2 \gamma(2, b_1) = 1.
            
    :param float b4: (**Optional**) b4 factor appearing in the disk profile defined as 
        
        .. math::
            
            2 \gamma(8, b_4) = 7!.

    :param float Ied: (**Optional**) intensity of the disk at half-light radius
    :param float Ied: (**Optional**) intensity of the bulge at half-light radius
    :param float magD: (**Optional**) total magnitude of the disk used to compute Ied if not given
    :param float magB: (**Optional**) total magnitude of the bulge used to compute Ieb if not given
    :param bool noError: (**Optional**) whether to not raise an error or not if one of the Ie values could not be computed correctly. If set to True, np.nan is returned.
    :param float offsetB: (**Optional**) magnitude offset in the magnitude system used for the bulge component
    :param float offsetD: (**Optional**) magnitude offset in the magnitude system used for the disk component
            
    :returns: D/T ratio at all the given positions or np.nan if one of the intensities could not be computed correctly
    :rtype: float or list[float]
    """
    
    #compute b1 and b4 if not given
    b1, b4 = check_bns([1, 4], [b1, b4])
    Ied    = checkAndComputeIe(Ied, 1, b1, rd, magD, offsetD)
    Ieb    = checkAndComputeIe(Ieb, 4, b4, rb, magB, offsetB)
    
    if None in [Ied, Ieb]:    
        return np.nan
        
    return fluxSersic(r, 4, rb, bn=b4, Ie=Ieb)['value'] / fluxSersics(r, [1, 4], [rd, rb], listbn=[b1, b4], listIe=[Ied, Ieb])['value']   

    
def fluxSersic(r, n, re, bn=None, Ie=None, mag=None, offset=None, start=0.0):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the flux of a single Sersic profile of index n up to radius r using raw integration.
    
    .. note::
    
        If no Ie is given, values for mag and offset must be given instead. 
    

    :param n: Sersic index of the profile
    :type n: int or float
    :param r: radius up to which the integral is computed. If a list is given, the position will be computed at each radius in the list.
    :type r: float or list[float]
    :param float re: half-light radius
        
    :param float bn: (**Optional**) bn factor appearing in the Sersic profile defined as 
        
        .. math::
            
            2 \gamma(2n, b_n) = \Gamma(2n).
            
    :param float Ie: (**Optional**) intensity at half-light radius
    :param float mag: (**Optional**) total magnitude used to compute Ie if not given
    :param float offset: (**Optional**) magnitude offset in the magnitude system used
    :param float start: (**Optional**) starting point of the integration
            
    :returns: integrated flux up to radius r (value) and an estimation of its absolute error (err) as the dictionary {'value':value, 'error':err}
    :rtype: dict
    """
    
    # The integral we need to compute to have the flux
    def the_integral(r, n, re, Ie=None, bn=None, mag=None, offset=None):
        return 2*np.pi*sersic_profile(r, n, re, Ie=Ie, bn=bn, mag=mag, offset=offset)*r
    
    bn, = check_bns([n], [bn])
    Ie = checkAndComputeIe(Ie, n, bn, re, mag, offset)
    if Ie is None:
        return None
        
    if isinstance(r, (int, float)):
        integral, error = quad(the_integral, start, r, args=(n, re, Ie, bn, mag, offset))
        return {'value':integral, 'error':error}
    
    integral = []
    error    = []
    for pos in range(len(r)):
        inte,  err = quad(the_integral, start, r[pos], args=(n, re, Ie, bn, mag, offset))
        integral.append(inte)
        error.append(err)
        
    return {'value':integral, 'error':error}


def fluxSersics(r, listn, listRe, listbn=None, listIe=None, listMag=None, listOffset=None, analytical=False):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the flux of a sum of Sersic profiles up to radius r (starting from 0).

    :param listn: list of Sersic index for each profile
    :type listn: list[int] or list[float]
    :param r: position at which the profiles are integrated. If a list if given, the position will be computed at each radius in the list.
    :type r: float or list[float]
    :param list[float] listRe: list of half-light radii for each profile
    
    :param bool analytical: (**Optional**) whether to use the analytical solution or integrate the profile
    :param list[float] listbn: (**Optional**) list of bn factors appearing in Sersic profiles
    :param list[float] listIe: (**Optional**) list of intensities at re for each profile
    :param list[float] listMag: (**Optional**) list of total integrated magnitudes for each profile
    :param list[float] listOffset: (**Optional**) list of magnitude offsets used in the magnitude system for each profile
         
    :returns: integrated flux of the sum of all the given Sersic profiles (value) and an estimation of the error (err) as the dictionary {'value':value, 'error'err}
    :rtype: dict
    
    :raises ValueError: if **listIe** and **listMag** and **listOffset** are None
    """
    
    # If no list of bn values is given, compute them all
    if listbn is None:
        listbn        = [compute_bn(n) for n in listn]
    listbn            = check_bns(listn, listbn)
    
    if listIe is None:
        if listMag is not None and listOffset is not None:
            listIe    = intensity_at_re(np.array(listn), np.array(listMag), np.array(listRe), np.array(listOffset), bn=np.array(listbn))
        else:
            raise ValueError("listIe is None, but listMag or listOffset is also None. If no listIe is given, please provide a value for the total magnitude and magnitude offset in order to compute the intensities.")
    
    res         = 0
    err         = 0
    for n, re, ie, bn in zip(listn, listRe, listIe, listbn):
        if not analytical:
            lum = fluxSersic(r, n, re, bn=bn, Ie=ie)
        else:
            lum = analyticFluxFrom0(r, n, re, bn=bn, Ie=ie)
        res    += lum['value']
        err    += lum['error']
        
    return {'value':np.asarray(res), 'error':np.asarray(err)}


def ratioFlux1D(r1, r2, listn, listRe, listbn=None, listIe=None, listMag=None, listOffset=None, analytical=True):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the ratio of the flux of the sum of different Sersic profiles for a single galaxy at two different positions in the galaxy plane only.
    
    This function computes the ratio from the 1D profiles, either integrating (analytical=False) or via an analytical solution (analytical=True).
    
    .. note::
        
        **How to use**
    
        Easiest way is to provide two radii for r1 and r2, and then lists of Sersic profiles parameters. 
        
        For instance, a ratio at radii 1" and 3" for a disk (n=1, Re=10") + bulge (n=4, Re=20") decomposition would give something like
        
            >>> ratioFlux1D(1, 3, [1, 4], [10, 20], listMag=[25, 30], listOffset=[30, 30])
            
        Radii should be given with the same unit as the effective radii.

    :param listn: list of Sersic index for each profile
    :type listn: list[int] or list[float]

    :param float r1: first radius where the flux is computed
    :param float r2: second radius where the flux is computed
    :param list[float] listRe: list of half-light radii for each profile

    :param bool analytical: (**Optional**) whether to use the analytical solution or integrate the profile
    :param list[float] listbn: (**Optional**) list of bn factors appearing in Sersic profiles
    :param list[float] listIe: (**Optional**) list of intensities at re for each profile
    :param list[float] listMag: (**Optional**) list of total integrated magnitudes for each profile
    :param list[float] listOffset: (**Optional**) list of magnitude offsets used in the magnitude system for each profile
     
    :returns: ratio of fluxes at the two different positions
    :rtype: float
    
    :raises ValueError:
        
        * if **listIe** and **listMag** and **listOffset** are None
        * if the 2nd computed flux is 0
        
    """    
    
    # If no list of bn values is given, compute them all
    if listbn is None:
        listbn         = [compute_bn(n) for n in listn]
    listbn             = check_bns(listn, listbn)
    
    if listIe is None:
        if listMag is not None and listOffset is not None:
            listIe     = intensity_at_re(np.array(listn), np.array(listMag), np.array(listRe), np.array(listOffset), bn=np.array(listbn))
        else:
            raise ValueError("listIe is None, but listMag or listOffset is also None. If no listIe is given, please provide a value for the total magnitude and magnitude offset in order to compute the intensities.")
    
    lum1           = fluxSersics(r1, listn, listRe, listbn=listbn, listIe=listIe, analytical=analytical)['value']
    lum2           = fluxSersics(r2, listn, listRe, listbn=listbn, listIe=listIe, analytical=analytical)['value']

    if lum2 == 0:
        raise ValueError("The luminosity computed at radius %f is 0. This is unlikely and the ratio cannot be computed." %lum2)
    
    return lum1/lum2


def ratioFlux2D(r1, r2, Rd, Rb, where=['galaxy', 'galaxy'], noPSF=[False, False], 
                        Id=None, Ib=None, magD=None, magB=None, offsetD=None, offsetB=None, inclination=0.0, PA=0.0,
                        arcsecToGrid=0.03, fineSampling=81,
                        PSF={'name':'Gaussian2D', 'FWHMX':0.8, 'FWHMY':0.8, 'sigmaX':None, 'sigmaY':None, 'unit':'arcsec'},
                        verbose=True):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the ratio of the flux of a bulge+disk model between two radii either in the galaxy plane or in the sky plane.
    This function computes the ratio from 2D models (projected on the sky plane or not) with or without PSF convolution.
    
    .. note::
        
        **How to use**
    
        Easiest way is to provide two radii for r1 and r2, and then lists of Sersic profiles parameters.
        
        For instance, a ratio for a radius of 1 pixel (in galaxy plane) over 3 pixels (in sky plane) for a disk (n=1, Re=10 pixels, inclination=23°, PA=40°) + bulge (n=4, Re=20 pixels) decomposition would give something like
            
            >>> ratioFlux2D(1, 3, 10, 20, magD=25, magB=30, offsetD=30, offsetB=30, inclination=23, PA=40, where=['galaxy', 'sky']})
    
   .. warning:
       
       To avoid problems:
        
       * provide all radii in the same pixel unit (e.g. HST or MUSE) 
       * update the **arcsecToGrid** conversion factor for the PSF if necessary.
        
       By default, the **arcsecToGrid** is tuned for HST resolution, so that the default PSF FWHM values (corresponding to MUSE PSF) will be converted into HST pixel values:
        
       * If radii are given in MUSE pixel values, then the MUSE conversion factor must be given for **arcsecToGrid**.
       * If radii are given in arcsec, then we are considering a grid with pixel size = 1", so that the conversion factor should be set to 1.
        
       The PSF FWHM and sigma values can be given in any relevant unit (arcsec, arcmin, degrees, radians, etc.). Please update the 'unit' key in the PSF dictionnary if you are providing values in arcsec.
       Astropy will apply the corresponding conversion from the given unit to pixel values, so it is important to always give **arcsecToGrid** in units of arcsec/pixel and nothing else.
        
    :param float r1: first radius where the flux is computed
    :param flaot r2: second radius where the flux is computed
    :param float Rb: disk half-light radius (same unit as r1)
    :param float Rd: bulge half-light radius (same unit as r1)
    
    :param float arcsecToGrid: (**Optional**) pixel size conversion in arcsec/pixel, used to convert the PSF FWHM (or sigma) from arcsec to pixel       
    :param int(>0) fineSampling: (**Optional**) fine sampling for the pixel grid used to make high resolution models. For instance, a value of 2 means that a pixel will be split into two subpixels.
    :param float Ib: (**Optional**) bulge intensity at Rb
    :param float Id: (**Optional**) disk intensity at Rd
    :param inclination: inclination of the galaxy in degrees
    :type inclination: int or float
    :param float magB: (**Optional**) bulge total magnitude
    :param float magD: (**Optional**) disk total magnitude
    :param [bool, bool] noPSF : (**Optional**) whether to not perform PSF convolution or not
    :param float offsetB: (**Optional**) bulge magnitude offset
    :param float offsetD: (**Optional**) disk magnitude offset
    :param PA: (**Optional**) position angle on sky in degrees
    :type PA: int or float
    :param dict PSF: (**Optional**) Dictionnary for the PSF (and its parameters) to use for the convolution. For now, only 2D Gaussians are accepted as PSF.
    :param bool verbose: (**Optional**) whether to print info on stdout or not
    :param [str, str] where: (**Optional**) where the flux is computed. For each radius two values are possible: 
            
        * 'galaxy' if the flux is to be computed in the galaxy plane
        * 'sky' if it is to be computed in the sky plane. 
        
    :returns: ratio of the two fluxes
    :rtype: float
    
    :raises TypeError: 
        
        * if **where** is neither a list, nor a tuple
        * if **noPSF** is neither a list, nor a tuple
        
    :raises ValueError:
        
        * if **where** is not of length 2
        * if one of the values in **where** is neither 'galaxy', nor 'sky'
        * if **noPSF** is not of length 2
        * if one of the values in **noPSF** is not a bool
        * if **Ib** and **magB** and **offsetB** are None
        * if **Id** and **magD** and **offsetD** are None
        * if the 2nd computed flux is 0
    """
    
    #########################################
    #       Checking input parameters       #
    #########################################
    
    if not isinstance(where, (list, tuple)):
        raise TypeError('where parameter should be either a list or a tuple.')
        
    if len(where) != 2:
        raise ValueError('where parameter should be of length 2 but current length is %d.' %(len(where)))
    
    for value in where:
        if value.lower() not in ['galaxy', 'sky']:
            raise ValueError("At least one of the values in where parameter is neither 'galaxy' nor 'sky'.")
            
    if not isinstance(noPSF, (list, tuple)):
        raise TypeError('noPSF parameter should be either a list or a tuple.')
        
    if len(noPSF) != 2:
        raise ValueError('noPSF list should be of length 2 but current length is %d.' %(len(noPSF)))
    
    for value in noPSF:
        if not isinstance(value, bool):
            raise ValueError("At least one of the values in noPSF parameter is not boolean.")
    
    ##################################
    #       Compute the ratio        #
    ##################################
    
    if Ib is None:
        if magB is not None and offsetB is not None:
            Ib       = intensity_at_re(4, magB, Rb, offsetB)
        else:
            raise ValueError("Ib is None, but magB or offsetB is also None. If no Ib is given, please provide a value for the total magnitude and magnitude offset in order to compute the intensity.")

    if Id is None:
        if magD is not None and offsetD is not None:
            Id       = intensity_at_re(1, magD, Rd, offsetD)
        else:
            raise ValueError("Id is None, but magD or offsetD is also None. If no Id is given, please provide a value for the total magnitude and magnitude offset in order to compute the intensity.")
    
    # Set inclination and PA according to where we compute the luminosity
    inc              = [0, 0]
    pa               = [0, 0]
    for i in [0, 1]:
        if where[i].lower() == 'sky':
            inc[i]   = inclination
            pa[i]    = PA
    
    # We compute both models: we generate models in grids of shape (2r1, 2r1) or (2r2, 2r2) since we will keep pixels within r1 and r2 to compute the flux.
    # We make the size uneven because we want the center to fall exactly on the central pixel (to be sure we are not missing any pixel flux in the sum)
    size             = 2*ceil(r1)
    if size%2 == 0:
        size        += 1
        
    # If the box size (x or y) is below 31 (radius of 15 pixels from the central pixel) we perform the sampling on the whole image, otherwise we just do it in a box of size [2*15+1, 2*15+1]
    if size <= 31:
        samplingZone = {'where':'all'}
    else:
        samplingZone = {'where':'centre', 'dx':15, 'dy':15}
    
    X1, Y1, mod1     = bulgeDiskOnSky(size, size, Rd, Rb, Id=Id, Ib=Ib, inclination=inc[0], PA=pa[0],
                                      fineSampling=fineSampling, PSF=PSF, noPSF=noPSF[0], arcsecToGrid=arcsecToGrid,
                                      samplingZone=samplingZone, verbose=verbose)
    
    size             = 2*ceil(r2)
    if size%2  == 0:
        size        += 1
        
    if size <=31:
        samplingZone = {'where':'all'}
    else:
        samplingZone = {'where':'centre', 'dx':15, 'dy':15}
        
    X2, Y2, mod2     = bulgeDiskOnSky(size, size, Rd, Rb, Id=Id, Ib=Ib, inclination=inc[1], PA=pa[1],
                                      fineSampling=fineSampling, PSF=PSF, noPSF=noPSF[1], arcsecToGrid=arcsecToGrid,
                                      samplingZone=samplingZone, verbose=verbose)
    
    # We compute the fluxes
    where1           = X1**2+Y1**2 <= r1**2
    where2           = X2**2+Y2**2 <= r2**2
    
    lum1             = np.nansum(mod1[where1])
    lum2             = np.nansum(mod2[where2])
    
    if lum2 == 0:
        raise ValueError("The luminosity computed at radius %f is 0. This is unlikely and the ratio cannot be computed." %lum2)
    
    return lum1/lum2
    

def total_flux(mag, offset):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the integrated flux up to infinity. The flux and magnitude are related by the equation
    
    .. math::
        
        m = -2.5 \log_{10} F_{\rm{tot}} + \rm{offset}

    :param offset: magnitude offset
    :type offset: float or list[float] or ndarray[float]
    :param mag: total magnitude
    :type mag: float or list[float] or ndarray[float]
    :returns: total flux
    :rtype: float or ndarray[float]
    """
    
    return 10**((np.asarray(offset)-np.asarray(mag))/2.5)


############################################################################################
#                                 Morphological parameters                                 #
############################################################################################

def computePAs(image, method='minmax', num=100, returnThresholds=False):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute a set of PA values for a galaxy using different minimum threshold values.
    
    .. note::
        
        **How it works**
        
        A set of threshold values are generated between the minimum and the maximum of the image, e.g. 
        
        * [0, 1, 2] for a galaxy with a minimum of 0 and a maximum of 2 (using num=3)
        
        These values are applied one after another onto the image as a minmum threshold, that is data points with value<threshold are masked. We get what we call a 'slice'.
        For each slice, we compute its PA (angle starting from the vertical axis, counting anti clockwise).
        
    .. warning::
    
        * PA angles are given between -90° and +90° so that there is a degeneracy between these two bounds.
        * If the 'minmax' method is used, a galaxy with a PA close to 90° will not have a good PA estimation as different slices will have values oscillating around +90° and -90°, yielding a median value of approximately 0°...
 
    :param im: image of a galaxy
    :type im: 2D ndarray
    :param method: (**Optional**) method to use
        
        - if 'minmax' the PA of each slice is computed as the angle between the min and the max within the slice (not very efficient)
        - if 'furthest' the PA of each slice is computed as the angle between the max and the furthest point relative to it (much more efficient)
        
    :type method: 'minmax' or 'furthest'
    :param int num: (**Optional**) how many slices must be made
    :param bool returnThresholds: (**Optional**) whether to return the threshold values as well as the PAs
        
    :returns: PA list (and the threshold values if returnThresholds is True)
    :rtype: list (and list f returnThresholds is True)
    
    :raises ValueError: if the method is neither 'minmax', nor 'furthest'
    :raises TypeError: if num is not an int, or if returnThresholds is not a bool
    '''
    
    if method.lower() not in ['minmax', 'furthest']:
        raise ValueError('Given method is not correct. Please provide either minmax or furthest. Cheers !')
    else:
        method    = method.lower()
    
    if not isinstance(num, int):
        raise TypeError('Given num parameter should only be an integer.')
        
    if not isinstance(returnThresholds, bool):
        raise TypeError('Given returnThresholds parameter should only be a bool.')
    
    # Get size and generate an X and Y coordinates grid
    sizeY, sizeX  = np.shape(image)
    X, Y          = np.meshgrid(np.arange(sizeX), np.arange(sizeY))
    
    maxi          = {}
    
    # Get min, max and their position
    maxi['val']   = np.nanmax(image)
    maxi['where'] = np.where(image==maxi['val'])
    maxi['x']     = X[maxi['where']]
    maxi['y']     = Y[maxi['where']]
    
    # Compute distance to max
    distance      = np.sqrt((X-maxi['x'])**2 + (Y-maxi['y'])**2)
    
    # Generate an evenly spaced range of values betwwen min and max 
    valRange      = np.linspace(np.nanmin(image), maxi['val'], num)
    theta         = []
    
    # We avoid the first value as there won't be pixels removed below this threshold, as well as the max since there will only remain one pixel, i.e. the max itself (in most cases)
    imCopy        = np.copy(image)
    point2        = {}
    for i in valRange[1:-1]:
        # Set to nan values below the threshold
        whereLess          = np.where(imCopy<i)
        imCopy[whereLess]  = np.nan
        
        if method=='minmax':
            point2['val']        = np.nanmin(imCopy)
            point2['where']      = np.where(imCopy==point2['val'])
        elif method=='furthest':
            distance[whereLess] = np.nan
            point2['where']     = np.where(distance==np.nanmax(distance))
            
        # If there are multiple points, we select the first one
        point2['x']             = X[point2['where']][0]
        point2['y']             = Y[point2['where']][0]
        
        # Set new theta value
        if maxi['x'] != maxi['y']:
            theta.append(np.arctan(-(maxi['x']-point2['x'])/(maxi['y']-point2['y']))[0])
        else:
            theta.append(np.pi/2)
        
    if returnThresholds:
        return theta, valRange
    else:
        return theta
    
    
##########################################################################################################
#                                 Thickness prescription                                                 #
##########################################################################################################

def disk_thickness(z):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Return the thickness of MS disk-like galaxies (see Mercier et al., 2021) prescription as a function of redshift.

    :param z: redshift
    :type z: float or ndarray[float]
    :returns: disk thickness
    :rtype: float or ndarray[float]
    
    :raises TypeError: if z is not an int, float, np.float16, np.float32, np.float64 or a ndarray
    '''
    
    if isinstance(z, (int, float, np.float16, np.float32, np.float64)):
        lq0        = 0.48 if z > 0.85 else 0.48 + 0.4*z
    elif isinstance(z, np.ndarray):
        lq0        = np.zeros(z.shape) + 0.48
        mask       = z <= 0.85
        lq0[mask] += 0.4*z[mask]
    else:
        raise TypeError('z array has type %s but it must either be float or a numpy array' %type(z))
        
    return 10**(-lq0)


def correct_inclination(inc, q0):
    '''
    Correct the inclination using the given disk thickness assuming the mass distribution is an oblate system. Correction is from Bottinelli et al., 1983 and is given by
    
    .. math::
        
        cos^2 i_0 = (q^2 - q_0^2) / (1 - q_0^2)
    
    where :math:`i_0` is the intrinsic inclination of the galaxy, :math:`q = b/a` is the observed axis ratio on the sky and :math:`q_0` is the intrinsic axis ratio of the galaxy.

    :param inc: observed inclination of the galaxy in degrees (assumed to be :math:`\arccos b/a`)
    :type inc: int/float/astropy Quantity object with unit of an angle of ndarray of one of these types
    :param q0: intrinsic axis ratio
    :type q0: int/float or ndarray[int]/ndarray[float]
    
    :returns: corrected inclination in degree
    :rtype: float or ndarray[float]
    '''
    
    if isinstance(inc, AstropyQuantityType):
        inc     = inc.to('rad')
    else:
        inc    *= np.pi/180
    
    q           = np.cos(inc)
    newinc      = np.arccos(np.sqrt((q*q - q0*q0) / (1 - q0*q0))) 
    
    if isinstance(newinc, AstropyQuantityType):
        newinc  = newinc.to('degree').value
    else:
        newinc *= 180/np.pi
        
    return newinc
    

def correct_I0(I0, q0, inc=None, inc0=None):
    '''
    Correct the observed central surface brightness of a double exponential disk when fitted with a single exponential disk due to the effect of finite thickness. 
    
    .. note::
        
        The intrinsic central surface brightness of a double exponential disk is larger than the central surface brightness given by the best-fit single exponential profile.
        
        The reason is that when fitting with  a single exponential profile, the inclination is biased because the intrinsic axis ratio is not deduced from it, which means the value is underestimated with respect to the intrinsic value.
        
        The correction is given by (Mercier et al., 2021)
        
        .. math::
            
            \Sigma(0) / \Sigma_{\rm{RT}} (0) = (q_0 \sin i_0 + \cos i_0) / \sqrt{q_0^2 \sin^2 i_0 + \cos^2 i_0}
            
    .. warning::
        
        Provide inclinations in degree.
    
    :param I: observed central surface brightness from the single exponential profile
    :type I: float or ndarray[float]
    :param q0: intrinsic axis ratio of the galaxy
    :type q0: float or ndarray[float]
    
    :param inc: (**Optional**) observed inclination in degree (not corrected of the galaxy thickness). If inc0 is given, inc0 is used instead.
    :type inc: float or ndarray[float] or astropy Quantity with angle unit
    :param inc0: (**Optional**) intrinsic inclination in degree (corrected of the galaxy thickness)
    :type inc0: float or ndarray[float] or astropy Quantity with angle unit
    
    :returns: corrected central surface brightness
    :rtype: float or ndarray[float]
    
    :raises ValueError: if both **inc** and **inc0** are None
    '''
    
    if inc is None and inc0 is None:
        raise ValueError('At least inc or inc0 must be provided. Cheers !')
    elif inc0 is None:
        inc0 = correct_inclination(inc, q0)
        
    if isinstance(inc0, AstropyQuantityType):
        inc0.to('rad')
    else:
        inc0 *= np.pi/180
    
    cosi0     = np.cos(inc0)
    sini0     = np.sin(inc0)
    
    r0        = (q0 * sini0 + cosi0) / np.sqrt(q0 * q0 * sini0 * sini0 + cosi0 * cosi0)
    
    return r0 * I0
    

#################################################################################################################
#                                 Half-light radius computation                                                 #
#################################################################################################################

def the_re_equation_for_2_Sersic_profiles(re, gal, b1=None, b4=None, noStructuredArray=False, magD=None, magB=None, Rd=None, Rb=None, offsetMagD=None, offsetMagB=None, 
                                          norm=1.0, stretch=1.0):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    A semi-analytical equation whose zero should give the value of the half-light radius for a bulge-disk decomposition defined as
    
    .. math::
        
        \Sigma (r) = I_{\rm{b}} e^{-b_4 \left [ \left (r/R_{\rm{b}} \right ) -1 \right ]} + I_{\rm{d}} e^{-b_1 \left [ \left (r/R_{\rm{d}} \right ) -1 \right ]},
    
    where :math:`R_{\rm{b}}, R_{\rm{d}}` are the bulge and disk effective radii, and :math:`I_{\rm{b}}, I_{\rm{d}}` are the bulge and disk surface brightness at their effective radii, respectively.
    
    .. note::

        This is meant to be used with a zero search algorithm (dichotomy or anything else).

    :param gal: structured array with data for all the galaxies. The required column names are:
        
        * 'R_d_GF' for the effective radius of the disk
        * 'R_b_GF' for the effective radius of the bulge
        * 'Mag_d_GF' for the total integrated magnitude of the disk
        * 'Mag_b_GF' the total integrated magnitude of the bulge
        
    :type gal: structured ndarray
    :param re: value of the half-light radius of the sum of the two components. This is the value which shall be returned by a zero search algorithm.
    :type re: float or list[float]
            
    :param float b1: b1 factor appearing in the Sersic profile of an exponential disk
    :param float b4: b4 factor appearing in the Sersic profile of a bulge
    :param magB: total magnitude of the bulge
    :type magB: float or list[float]
    :param magD: total magnitude of the disk
    :type magD: float or list[float]
    :param float norm: normalisation factor to divide the equation (used to improve convergence)
    :param bool noStructuredArray: if False, the structured array gal will be used. If False, values of the magnitudes and half-light radii of the two components must be given.
    :param offsetMagD: magnitude offset used in the magnitude system for the disk
    :type offsetMagD: int or float
    :param offsetMagB: magnitude offset used in the magnitude system for the bulge
    :type offsetMagB: int or float
    :param Rb: half-light radius of the bulge
    :type Rb: float or list[float]
    :param Rd: half-light radius of the disk
    :type Rd: float or list[float]
    :param float stretch: dilatation factor used to multiply re in order to smooth out the sharp slope around the 0 of the function
        
    :returns: value of the left-hand side of the equation. If re is correct, the returned value should be close to 0.
    :rtype: float
    
    :raises TypeError: if offsetMagD and offsetMagB are given but are neither float, nor int
    """

    b1, b4 = check_bns([1, 4], [b1, b4])
    magD, magB, Rd, Rb = fromStructuredArrayOrNot(gal, magD, magB, Rd, Rb, noStructuredArray)
    
    # Check mag offsets first
    if offsetMagD is not None and offsetMagB is not None:
        if not isinstance(offsetMagD, float) and not isinstance(offsetMagD, int):
            raise TypeError(errorMessage('Disk magnitude is neither float not int. ') + 'Please provide a correct type. Cheers !')
            
        if not isinstance(offsetMagB, float) and not isinstance(offsetMagB, int):
            raise TypeError(errorMessage('Bulge magnitude is neither float not int. ') + 'Please provide a correct type. Cheers !')
    else:
        print(brightMessage('At least one of the mag offset was not given. Assuming both are equal.'))
        offsetMagB     = 0
        offsetMagD     = 0
    
    # Convert strecth factor to float to avoid numpy casting operation errors
    re = re*float(stretch)
    
    try:
        re = [0 if r<0 else r for r in re]
    except:
        if re < 0:
            re = 0
    
    return ( 10**((offsetMagD-magD)/2.5)*(gammainc(2, b1*(re/Rd)) - 0.5) + 10**((offsetMagB-magB)/2.5)*(gammainc(8, b4*(re/Rb)**(1.0/4.0)) - 0.5) ) / norm


def solve_re(gal, guess=None, b1=None, b4=None, noStructuredArray=False, magD=None, magB=None, Rd=None, Rb=None, normalise=True, stretch=5e-2,
             integration=False, Ltot=None, Ie=None, offsetMagD=None, offsetMagB=None, xtol=1e-3, useZeroOrder=True, method='hybr',
             verbose=True):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    This is meant to find the half-light radius of the sum of an exponential disk and a de Vaucouleur bulge, either via a semi-analytical formula, or using numerical integration.
    
    .. note::
        
        **How to use**
        
        There are two ways to use this function: 
            
            * using numerical integration of the light profiles
            * by finding the zero of a specific equation. 

        In both cases, the parameter **gal** is mandatory. This corresponds to a numpy structured array with the following fields: 
            
            * 'Mag_d_GF', 'Mag_b_GF', 'R_d_GF' and 'R_b_GF'
        
        **HOWEVER**, if the flag **noStructuredArray** is True, this array will not be used (so just cast anything into this parameter, it will not matter) but instead, the optional parameters **magD**, **magB**, **Rd** and **Rb** must be provided.
        
        The guess can be ignored, though the result may not converge.
        
        **b1** and **b4** values do not necessarily need to be provided if you only call this function very few times. If not, they will be computed once at the beginning and propagated in subsequent function calls.
            
        Solving methods:
            
            a) **Numerical integration**
    
               This method will find the zero of the following function 
               
               .. math::

                   f(r) = 2\pi \int_0^r dr~r \Sigma (r) - L_{\rm{tot}}, 
                   
              where :math:`L_{\rm{tot}}` is the total integrated luminosity of the sum of the disk and bulge. The **Ltot** parameter is not mandatory, as it will be computed if not provided. 
              
              **However, if not provided, this requires to give magnitude values (this is mandatory in any case) AND a magnitude offset value in order to compute it**.
                
              The **Ie** parameter can be given or can be ignored. In the latter case, it will be computed using the magnitudes and magnitude offset, so this last parameter should be provided as well in this case.
            
            b) **Semi-analytical solution**
    
               .. warning::
                   
                   This is an experimental feature. It follows from analytically computing the equation for re using its definition as well as the sum of an exponential disk and a bulge.
                
               In this case, the **integration** parameter must be set to False.
                
               **THE FOLLOWING PARAMETERS ARE NOT REQUIRED FOR THIS METHOD:** 
               
               * **Ltot**, **Ie** and **offset**
                
        **Additional information**
        
            For only one galaxy, only a scalar value may be provided for each parameter you would like to pass. However, for more than one galaxy, a list must be given instead.
        
    Basically, the simplest way to solve re is to call the function the following way:
         
         >>> solve_re(array)
         
     where array is a numpy structured array with the relevant columns.
    
    :param gal: structured array with data for all the galaxies. The required column names are 'R_d_GF' (re for the disk component), 'R_b_GF' (re for the bulge component), 'Mag_d_GF' (the total integrated magnitude for the disk component), 'Mag_b_GF' (the total integrated magnitude for the bulge component).
    :type gal: structured ndarray

    :param float b1: (**Optional**) b1 factor appearing in the Sersic profile of an exponential disk
    :param float b4: (**Optional**) b4 factor appearing in the Sersic profile of a bulge        
    :param guess: (**Optional**) guess for the value of re for all the galaxies
    :type guess: float or list[float]
    :param Ie: (**Optional**) intensity at half-light radius for all the galaxies (including both profiles)
    :type Ie: float or list[float]
    :param bool integration: (**Optional**) whether to find re integrating the light profiles or not (i.e. solving the re equation)
    :param Ltot: (**Optional**) total luminosity of the galaxies. This parameter is used when finding re using numerical integration of the light profiles. If integration is True and no Ltot is provided, it will be computed using the total magnitude of each component and the offset value.
    :type Ltot: float or list[float]
    :param magB: (**Optional**) total magnitude of the bulge
    :type magB: float or list[float]
    :param magD: (**Optional**) total magnitude of the disk
    :type magD: float or list[float]
    :param str method: (**Optional**) method to use to find the zero of the re equation function or the integral to solve
    :param bool normalise: (**Optional**) whether to normalise the equation or not. It is recommended to do so to improve the convergence.
    :param bool noStructuredArray: (**Optional**) if False, the structured array gal will be used. If False, values of the magnitudes and half-light radii of the two components must be given.
    :param offsetMagD: (**Optional**) magnitude offset used in the magnitude system for the disk
    :type offsetMagD: int or float
    :param offsetMagB: (**Optional**) magnitude offset used in the magnitude system for the bulge
    :type offsetMagB: int or float
    :param Rb: (**Optional**) half-light radius of the bulge
    :type Rb: float or list[float]
    :param Rd: (**Optional**) half-light radius of the disk
    :type Rd: float or list[float]
    :param float stretch: (**Optional**) dilatation factor used to multiply re in order to smooth out the sharp slope around the 0 of the function
    :param bool useZeroOder: (**Optional**) whether to use the zero order analytical solution as a guess. If True, the value of guess will be used by the zero search algorithm.
    :param bool verbose: (**Optional**) whether to print messaged on stdout (True) or not (False)
    :param float xtol: (**Optional**) relative error convergence factor
            
    :returns: value of re, as well as a convergence flag and a debug dict
    :rtype: float or list[float], float or list[float], dict or list[dict]
    """
    
    # To solve numerically we find the zero of the difference between the integral we want to solve and half the total luminosity
    def integral_to_solve(r, listn, listRe, listbn, listIe, listMag, listOffset, Ltot):
        res, err     = fluxSersics(r, listn, listRe, listbn=listbn, listIe=listIe, listMag=listMag, listOffset=listOffset)
        return res-Ltot/2.0
     
        
    # ##########################################################
    #            Compute bn, mag and radii values              #
    ############################################################
    
    # If only a single value is given, transform into numpy arrays
    if isinstance(magD, (float, int)):
        magD = np.array([magD])
    if isinstance(magB, (float, int)):
        magB = np.array([magB])
    if isinstance(Rd, (float, int)):
        Rd   = np.array([Rd])
    if isinstance(Rb, (float, int)):
        Rb   = np.array([Rb])
    
    b1, b4             = check_bns([1, 4], [b1, b4])
    magD, magB, Rd, Rb = fromStructuredArrayOrNot(gal, magD, magB, Rd, Rb, noStructuredArray)
    
    #########################################
    #            Define a guess             #
    #########################################
    
    #use the zero order solution as a guess if the flag if trigerred
    if useZeroOrder:
        md10         = 10**(-magD/2.5)
        mb10         = 10**(-magB/2.5)
        numerator    = md10 + mb10 
        denominator  = md10*(b1/Rd)**2 + (2/factorial(8))*mb10*( (b4**4)/Rb )**2
        guess        = np.sqrt(numerator/denominator)
    else:
        #if no guess given, set default to 10px
        if guess is None:
            guess    = np.copy(Rd)*0+10
    
    #################################################
    #            Declare output arrays              #
    #################################################
    
    #set ouput arrays empty
    solution         = np.array([])
    convFlag         = np.array([], dtype=bool)
    debug            = np.array([])
    
    #compute re from the re equation instead of integrating
    if not integration:
            
        # Check mag offset values
        if offsetMagB is None or offsetMagD is None:
            
            if verbose:
                print(brightMessage('At least one of the mag offset values was not provided. Assuming both are equal.'))
            offsetMagB = 0
            offsetMagD = 0
            
        #define the normalisation of the equation
        if normalise:
            norm     = 0.5*(10**((offsetMagD-magD)/2.5) + 10**((offsetMagB-magB)/2.5))
        else:
            norm     = 1.0
            
        #print(magD, magB, Rd, Rb, guess, norm)
        
        #solve by finding the zero of the function
        for g, md, mb, rd, rb, nm in zip(guess, magD, magB, Rd, Rb, norm):
            sol      = root(the_re_equation_for_2_Sersic_profiles, g, 
                            args=(None, b1, b4, True, md, mb, rd, rb, offsetMagD, offsetMagB, nm, stretch),
                            method=method, options={'xtol':xtol, 'maxfev':200})
            
            # solution tested in the_re_equation is multiplied by the stretch factor, so that the best-solution*stretch gives the_re_equation \approx 0
            # thus the best-solution needs to be multiplied by the stretch factor at the end to recover the true values
            solution = np.append(solution, sol['x']*stretch)
            convFlag = np.append(convFlag, sol['status']==1.0)
            debug    = np.append(debug, sol['message'])
            
    #computes re by integrating the light profiles instead
    else:
        
        #first check that offsets are provided
        if offsetMagB is None or offsetMagD is None:
            print("ValueEror: a None was found. One of the inputs in the list offsetMagD, offsetMagD was not provided. If you are using the integration method, you must provide offset values for all galaxies and for both profiles (exponential disk and bulge)")
            return None
        
        #if no total luminosity given, compute it
        if Ltot is None:
            Ltot     = total_flux(magD, offsetMagD) + total_flux(magB, offsetMagB)
            
        #setting redundant arrays to defaults values
        listn        = [1, 4]
        listbn       = [b1, b4]

        for g, md, mb, rd, rb, offsetd, offsetb, ltot in zip(guess, magD, magB, Rd, Rb, offsetMagD, offsetMagB, Ltot):
            #we do not provide the list of intensities at re but instead the magnitudes and offset for each component
            sol      = root(integral_to_solve, g,
                            args=(listn, [rd, rb], listbn, None, [md, mb], [offsetd, offsetb], ltot),
                            method=method, options={'xtol':xtol, 'maxfev':200})
            
            solution = np.append(solution, sol['x'])
            convFlag = np.append(convFlag, sol['status']==1.0)
            debug    = np.append(debug, sol['message'])
            
    return solution, convFlag, debug


#################################################################################################################
#                                Other related Sersic functions                                                 #
#################################################################################################################
    
def centralIntensity(n, re, Ie=None, mag=None, offset=None):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the central intensity of a given Sersic profile.

    :param n: Sersic index of the profile
    :type n: int or float
    :param float re: half-light radius
    
    :param float Ie: (**Optional**) intensity at Re. If None, values for mag and offset must be given instead.
    :param float mag: (**Optional**) total magnitude. If None, Ie must be given instead.
    :param float offset: (**Optional**) magnitude offset. If None, Ie must be given instead

    :returns: central intensity of the Sersic profile
    :rtype: float
    
    :raises ValueError: if **Ie** and **mag** and **offset** are None
    '''
    
    bn = compute_bn(n)
    
    if Ie is None:
        if mag is not None and offset is not None:
            Ie = intensity_at_re(n, mag, re, offset, bn=bn)
        else:
            raise ValueError("Ie value is None, but mag and/or offset is/are also None. If no Ie is given, please provide a value for the total magnitude and magnitude offset in order to compute the former one. Cheers !")
    
    return Ie*np.exp(bn)


def compute_R22(Red, dRed=None, b1=None):
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute R22 (and its error) given an array of disk effective radii defined as
    
    .. math::
        
        R_{22} = 2.2 \times R_{\rm{d}} / b_1,
        
    where :math:`R_{\rm{d}}` is the disk effective radius.

    :param Red: disk effective radii
    :type Red: float or ndarray[float]
        
    :param float b1: (**Optional**) b1 factor appearing in the exponential disk profile
    :param dRed: (**Optional**) error estimate on the effective radii
    :type dRed: float or ndarray[foat]

    :returns: R22 (and its error)
    :rtype: float or ndarray[float]
    '''
    
    b1, = check_bns([1], [b1])
    
    if dRed is None:
        return 2.2*Red/b1
    else:
        return 2.2*Red/b1, 2.2*dRed/b1