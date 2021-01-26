#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:37:05 2020

@author: wilfried

Computation relative to galaxy morphological information.
"""

import numpy                              as     np
from   scipy.special                      import gammainc
from   scipy.optimize                     import root
from   scipy.integrate                    import quad
from   math                               import factorial, ceil
from   .models                            import sersic_profile, bulgeDiskOnSky
from   .misc                              import check_bns, compute_bn, realGammainc, checkAndComputeIe, intensity_at_re, fromStructuredArrayOrNot
from   ..utilities.coloredMessages        import errorMessage, brightMessage


#################################################################################################################
#                                           Sersic luminosities                                                 #
#################################################################################################################

def analyticFluxFrom0(r, n, re, bn=None, Ie=None, mag=None, offset=None, start=0.0):
    """
    Analytically compute the integrated flux from 0 up to radius r for a Sersic profile of index n.
    
    How to use
    ----------
        If no Ie is given, values for mag and offset must be given instead for the corresponding component. 
    
    Mandatory inputs
    ----------------
        n : float/int
            Sersic index of the profile
        r : float/list of floats
            radius up to the integral will be computed.
        re : float
            half-light radius
                
    Optional inputs
    ---------------
        bn : float
            b1nfactor appearing in the Sersic profile defined as $2 \gamma(2, bn) = \Gamma(2n). By default, bn is None, and its value will be computed. To skip this computation, please give a value to bn when callling the function.
        Ie : float
            intensity at half-light radius
        mag : float
            component total integrated magnitude used to compute Ie if not given
        offset : float
            magnitude offset in the magnitude system used
            
    Return the analytically derived flux from 0 to r.
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
    """
    Computes the ratio of the bulge flux (B) over the disk one (D) of a two Sersic components galaxy up to radius r.
    
    How to use
    ----------
        If no Ie is given, values for mag and offset must be given instead, in order to compute it. 
    
    Mandatory inputs
    ----------------
        r : float/list of floats
            position at which the profile is integrated. If a list is given, the position will be computed at each radius in the list.
        rb : float
            half-light radius of the bulge component
        rd : float
            half-light radius of the disk component
                
    Optional inputs
    ---------------
        b1 : float
            b1 factor appearing in the Sersic profile defined as $\gamma(2, b1) = 1/2$. By default, b1 is None, and its value will be computed. To skip this computation, please give a value to bn when callling the function.
        b4 : float
            b4 factor appearing in the Sersic profile defined as $2\gamma(8, b4) = 7!$. By default, b4 is None, and its value will be computed. To skip this computation, please give a value to bn when callling the function.
        Ieb : float
            bulge intensity at half-light radius
        Ied : float
            disk intensity at half-light radius
        magB : float
            galaxy total integrated magnitude used to compute the bulge component Ie if not given
        magD : float
            galaxy total integrated magnitude used to compute the disk component Ie if not given
        noError : bool
            whether to not raise an error or not if one of the Ie values could not be computed correctly. Default is False. If set to True, np.nan is returned.
        offsetB : float
            magnitude offset in the magnitude system used for the bulge component
        offsetD : float
            magnitude offset in the magnitude system used for the disk component
            
    Returns the B/D ratio at all the given positions or NaN if one of the intensities could not be computed correctly..
    """
    
    #compute b1 and b4 if not given
    b1, b4 = check_bns([1, 4], [b1, b4])
    Ied    = checkAndComputeIe(Ied, 1, b1, rd, magD, offsetD, noError=noError)
    Ieb    = checkAndComputeIe(Ieb, 4, b4, rb, magB, offsetB, noError=noError)
    
    if None in [Ieb, Ied]:
        return np.nan
        
    return fluxSersic(r, 4, rb, bn=b4, Ie=Ieb)['value'] / fluxSersic(r, 1, rd, bn=b1, Ie=Ied)['value']    


def BoverT(r, rd, rb, b1=None, b4=None, Ied=None, Ieb=None, magD=None, magB=None, offsetD=None, offsetB=None, noError=False):
    """
    Computes the ratio of the bulge flux (B) over the total one (T=D+B) of a two Sersic components galaxy up to radius r.
    
    How to use
    ----------
        If no Ie is given, values for mag and offset must be given instead, in order to compute it. 
    
    Mandatory inputs
    ----------------
        r : float/list of floats
            position at which the profile is integrated. If a list if given, the position will be computed at each radius in the list.
        rb : float
            half-light radius of the bulge component
        rd : float
            half-light radius of the disk component
                
    Optional inputs
    ---------------
        b1 : float
            b1 factor appearing in the Sersic profile defined as $\gamma(2, b1) = 1/2$. By default, b1 is None, and its value will be computed. To skip this computation, please give a value to bn when callling the function.
        b4 : float
            b4 factor appearing in the Sersic profile defined as $2\gamma(8, b4) = 7!$. By default, b4 is None, and its value will be computed. To skip this computation, please give a value to bn when callling the function.
        Ieb : float
            bulge intensity at half-light radius
        Ied : float
            disk intensity at half-light radius
        magB : float
            galaxy total integrated magnitude used to compute the bulge component Ie if not given
        magD : float
            galaxy total integrated magnitude used to compute the disk component Ie if not given
        noError : bool
            whether to not raise an error or not if one of the Ie values could not be computed correctly. Default is False. If set to True, np.nan is returned.
        offsetB : float
            magnitude offset in the magnitude system used for the bulge component
        offsetD : float
            magnitude offset in the magnitude system used for the disk component
            
    Returns the B/T ratio at all the given positions or NaN if one of the intensities could not be computed correctly.
    """
    
    #compute b1 and b4 if not given
    b1, b4 = check_bns([1, 4], [b1, b4])
    Ied    = checkAndComputeIe(Ied, 1, b1, rd, magD, offsetD, noError=noError)
    Ieb    = checkAndComputeIe(Ieb, 4, b4, rb, magB, offsetB, noError=noError)
    
    if None in [Ied, Ieb]:    
        return np.nan
        
    return fluxSersic(r, 4, rb, bn=b4, Ie=Ieb)['value'] / fluxSersics(r, [1, 4], [rd, rb], listbn=[b1, b4], listIe=[Ied, Ieb])['value']    


def DoverT(r, rd, rb, b1=None, b4=None, Ied=None, Ieb=None, magD=None, magB=None, offsetD=None, offsetB=None):
    """
    Computes the ratio of the disk flux (D) over the total one (T=D+B) of a two Sersic components galaxy up to radius r.
    
    How to use
    ----------
        If no Ie is given, values for mag and offset must be given instead, in order to compute it. 
    
    Mandatory inputs
    ----------------
        r : float/list of floats
            position at which the profile is integrated. If a list if given, the position will be computed at each radius in the list.
        rb : float
            half-light radius of the bulge component
        rd : float
            half-light radius of the disk component
                
    Optional inputs
    ---------------
        b1 : float
            b1 factor appearing in the Sersic profile defined as $\gamma(2, b1) = 1/2$. By default, b1 is None, and its value will be computed. To skip this computation, please give a value to bn when callling the function.
        b4 : float
            b4 factor appearing in the Sersic profile defined as $2\gamma(8, b4) = 7!$. By default, b4 is None, and its value will be computed. To skip this computation, please give a value to bn when callling the function.
        Ieb : float
            bulge intensity at half-light radius
        Ied : float
            disk intensity at half-light radius
        magB : float
            galaxy total integrated magnitude used to compute the bulge component Ie if not given
        magD : float
            galaxy total integrated magnitude used to compute the disk component Ie if not given
        offsetB : float
            magnitude offset in the magnitude system used for the bulge component
        offsetD : float
            magnitude offset in the magnitude system used for the disk component
            
    Returns the D/T ratio at all the given positions or NaN if one of the intensities could not be computed correctly..
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
    Compute the flux of a single Sersic profile of index n up to radius r using raw integration.
    
    How to use
    ----------
        If no Ie is given, values for mag and offset must be given instead, in order to compute it. 
    
    Mandatory inputs
    ----------------
        n : float/int
            Sersic index of the profile
        r : float/list of floats
            position at which the profile is integrated. If a list if given, the position will be computed at each radius in the list.
        re : float
            half-light radius
                
    Optional inputs
    ---------------
        bn : float
            bn factor appearing in the Sersic profile defined as $2\gamma(2n, bn) = \Gamma(2n)$. By default, bn is None, and its value will be computed. To skip this computation, please give a value to bn when callling the function.
        Ie : float
            intensity at half-light radius
        mag : float
            galaxy total integrated magnitude used to compute Ie if not given
        offset : float
            magnitude offset in the magnitude system used
        start : float
            starting point of the integration
            
    Return the integrated flux up to radius r and an estimation of its absolute error as a dictionary. 
    If a list of radii is given, it returns a list of luminosities and a list of absolute errors.
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
    Compute the flux of a sum of Sersic profiles up to radius r (starting from 0).
    
    Mandatory inputs
    ----------------
        listn : list of float/int
            list of Sersic index for each profile
        r : float/list of floats
            position at which the profiles are integrated. If a list if given, the position will be computed at each radius in the list.
        listRe : list of float
            list of half-light radii for each profile
        
    Optional inputs
    ---------------
        analytical : bool
            whether to use the analytical solution or integrate the profile. Default is to integrate. 
        listbn : list of floats/None
            list of bn factors appearing in Sersic profiles defined as $2\gamma(2n, bn) = \Gamma(2n)$. If no value is given, each bn will be computed according to their respective Sersic index. If you do not want this function to compute the value of one of the bn, provide its value in the list, otherwise put it to None.
        listIe : list of floats
            list of intensities at re for each profile
        listMag : list of floats
            list of total integrated magnitudes for each profile
        listOffset : list of floats
             list of magnitude offsets used in the magnitude system for each profile
         
    Return the integrated flux of the sum of all the given Sersic profiles and an estimation of the error as a dictionary.
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
    Compute the ratio of the flux of the sum of different Sersic profiles for a single galaxy at two different positions in the galaxy plane only.
    This function computes the ratio from the 1D profiles, either integrating (analytical=False) or via an analytical solution (analytical=True).
    
    How to use
    ----------
    
        Easiest way is to provide two radii for r1 and r2, and then lists of Sersic profiles parameters. For instance, a ratio at radii 1" and 3" for a disk (n=1, Re=10") + bulge (n=4, Re=20") decomposition would give something like
            >> ratioFlux1D(1, 3, [1, 4], [10, 20], listMag=[25, 30], listOffset=[30, 30])
        Radii should be given with the same unit as the effective radii.

    Mandatory inputs
    ----------------
        listn : list of float/int
            list of Sersic index for each profile
        r1 : float
            first radius where the luminosity will be computed
        r2 : float
            second radius where the luminoisty will be computed
        listRe : list of float
            list of half-light radii for each profile
        
    Optional inputs
    ---------------
        analytical : bool
            whether to use the analytical solution or integrate the profile. Default is True.
        listbn : list of floats/None
            list of bn factors appearing in Sersic profiles defined as $2\gamma(2n, bn) = \Gamma(2n)$. If no value is given, each bn will be computed according to their respective Sersic index. If you do not want this function to compute the value of one of the bn, provide its value in the list, otherwise put it to None.
        listIe : list of floats
            list of intensities at re for each profile
        listMag : list of floats
            list of total integrated magnitudes for each profile
        listOffset : list of floats
             list of magnitude offsets used in the magnitude system for each profile
         
    Return the ratio of fluxes at the two different positions.
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
    Compute the ratio of the flux of a bulge+disk model between two radii either in the galaxy plane or in the sky plane.
    This function computes the ratio from 2D models (projected on the sky plane or not) with or without PSF convolution.
    
    How to use
    ----------
    
        Easiest way is to provide two radii for r1 and r2, and then lists of Sersic profiles parameters. 
        For instance, a ratio for a radius of 1 pixel (in galaxy plane) over 3 pixels (in sky plane) for a disk (n=1, Re=10 pixels, inclination=23°, PA=40°) + bulge (n=4, Re=20 pixels) decomposition would give something like
            >> ratioFlux2D(1, 3, 10, 20, magD=25, magB=30, offsetD=30, offsetB=30, inclination=23, PA=40, where=['galaxy', 'sky']})
    
   Caution
    -------
        To avoid problems, provide all radii in the same pixel unit (e.g. HST or MUSE) and update the arcsecToGrid conversion factor for the PSF if necessary.
        By default, the arcsecToGrid is tuned for HST resolution, so that the default PSF FWHM values (corresponding to MUSE PSF) will be converted into HST pixel values.
        
        If radii are given in MUSE pixel values, then the MUSE conversion factor must be given for arcsecToGrid.
        If radii are given in arcsec, then we are considering a grid with pixel size = 1", so that the conversion factor should be set to 1.
        
        The PSF FWHM and sigma values can be given in any relevant unit (arcsec, arcmin, degrees, radians, etc.). Please update the 'unit' key in the PSF dictionnary if you are providing values in arcsec.
        Astropy will apply the corresponding conversion from the given unit to pixel values, so it is important to always give arcsecToGrid in units of arcsec/pixel and nothing else.
        
    Mandatory inputs
    ----------------
        r1 : float
            first radius where the luminosity will be computed
        r2 : float
            second radius where the luminoisty will be computed (same unit as r1)
        Rb : float
            disk half-light radius (same unit as r1)
        Rd : float
            bulge half-light radius (same unit as r1)
        
    Optional inputs
    ---------------
        arcsecToGrid : float
            pixel size conversion in arcsec/pixel, used to convert the PSF FWHM (or sigma) from arcsec to pixel. Default is HST-ACS resolution of 0.03"/px.        
        fineSampling : positive int
            fine sampling for the pixel grid used to make high resolution models. For instance, a value of 2 means that a pixel will be split into two subpixels. Default is 1.
        Ib : float
            bulge intensity at Rb. Default is None so that it is computed from the bulge magnitude and magnitude offset.
        Id : float
            disk intensity at Rd. Default is None so that it is computed from the bulge magnitude and magnitude offset.
        inclination : float/int
            inclination of the galaxy in degrees. Default is 0.0.
        magB : float
            bulge total magnitude. If Ib is not provided, it must be given instead.
        magD : float
            disk total magnitude. If Id is not provided, it must be given instead.
        noPSF : list of two bool
            whether to not perform PSF convolution or not. Default is to do convolution for each radius.
        offsetB : float
            bulge magnitude offset. If Ib is not provided, it must be given instead.
        offsetD : float
            disk magnitude offset. If Id is not provided, it must be given instead.
        PA : float/int
            position angle on sky in degrees. Default is 0.0.
        PSF : dict
            Dictionnary of the PSF (and its parameters) to use for the convolution. Default is a (0, 0) centred radial gaussian (muX=muY=0 and sigmaX=sigmaY) with a FWHM corresponding to that of MUSE (~0.8"~4 MUSE pixels).
            For now, only 2D Gaussians are accepted as PSF.
        verbose : bool
            whether to print info on stdout or not. Default is True.
        where : list of 2 str
            where the flux is computed. For each radius two values are possible: 
                - 'galaxy' if the flux is to be computed in the galaxy plane
                - 'sky' if it is to be computed in the sky plane. 
            Default is 'galaxy' for both radii.
         
    Return the ratio of the two fluxes.
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
    """
    Compute the integrated flux up to infinity. The flux and magnitude are related by the equation
    
    mag = -2.5 \log_{10}(F_tot) + offset
    
    Mandatory inputs
    ----------------
        offset : float/list of floats
            magnitude offset
        mag : float/list of floats
            total magnitude
            
    Return the total flux.
    """
    
    return 10**((np.asarray(offset)-np.asarray(mag))/2.5)


############################################################################################
#                                 Morphological parameters                                 #
############################################################################################

def computePAs(image, method='minmax', num=100, returnThresholds=False):
    '''
    Compute a set of PA values for a galaxy using different minimum threshold values.
    
    How it works
    ------------
        A set of threshold values are generated between the minimum and the maximum of the image, e.g. [0, 1, 2] for a galaxy with a minimum of 0 and a maximum of 2 (using num=3).
        These values are applied one after another onto the image as a minmum threshold, that is data points with value<threshold are masked. We get what we call a 'slice'.
        For each slice, we compute its PA (angle starting from the vertical axis, counting anti clockwise).
        
    Warning
    -------
        PA angles are given between -90° and +90° so that there is a degeneracy between these two bounds. 
        If the 'minmax' method is used, a galaxy with a PA close to 90° will not have a good PA estimation as different slices will have values oscillating around +90° and -90°, yielding a median value of approximately 0°...

    Inputs
    ------   
        im : numpy 2D array
            image of a galaxy
        method : 'minmax' or 'furthest'
            - if 'minmax' the PA of each slice is computed as the angle between the min and the max within the slice (not very efficient)
            - if 'furthest' the PA of each slice is computed as the angle between the max and the furthest point relative to it (much more efficient)
        num : int
            how many slices must be made
        returnThresholds : bool
            whether to return the threshold values as well as the PAs
            
    Return the PA list (and the threshold values if returnThresholds is set True).
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
    

#################################################################################################################
#                                 Half-light radius computation                                                 #
#################################################################################################################

def the_re_equation_for_2_Sersic_profiles(re, gal, b1=None, b4=None, noStructuredArray=False, magD=None, magB=None, Rd=None, Rb=None, offsetMagD=None, offsetMagB=None, 
                                          norm=1.0, stretch=1.0):
    """
    A semi-analytical equation whose zero should give the value of the half-light radius when two Sersic profiles (with n=1 and n=4) are combined together.
    This is meant to be used with a zero search algorithm (dichotomy or anything else).
    
    Mandatory inputs
    ----------------
        gal : numpy structured array
            structured array with data for all the galaxies. The required column names are 'R_d_GF' (re for the disk component), 'R_b_GF' (re for the bulge component), 'Mag_d_GF' (the total integrated magnitude for the disk component), 'Mag_b_GF' (the total integrated magnitude for the bulge component).
        re : float/list of floats
            value of the half-light radius of the sum of the two components. This is the value which shall be returned by a zero search algorithm.
            
    Optional inputs
    ---------------
        b1 : float
            b1 factor appearing in the Sersic profile of an exponential disc, defined as $\gamma(2, b1) = 1/2$. By default, b1 is None, and its value will be computed by the function. To skip this computation, please give it a value when callling the function.
        b4 : float
            b4 factor appearing in the Sersic profile of a bulge, defined as $2\gamma(8, b4) = 7!$. By default, b4 is None, and its value will be computed by the function. To skip this computation, please give it a value when callling the function.
        magB : float/list of floats
            total magnitude of the bulge component of the galaxies
        magD : float/list of floats
            total magnitude of the disk component of the galaxies
        norm : float
            normalisation factor to divide the equation (used to improve convergence)
        noStructuredArray : boolean
            if False, the structured array gal will be used. If False, values of the magnitudes and half-light radii of the two components must be given.
        offsetMagD : float/list of floats
            magnitude offset used in the magnitude system for the disk component
        offsetMagB : float/list of floats
            magnitude offset used in the magnitude system for the bulge component
        Rb : float/list of floats
            half-light radius of the bulge components of the galaxies
        Rd : float/list of floats
            half-light radius of the disk components of the galaxies
        stretch : float
            dilatation factor used to multiply re in order to smooth out the sharp slope around the 0 of the function
        
    Return the value of the left-hand side of the equation. If re is correct, the returned value should be close to 0.
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
    """
    This is meant to find the half-light radius of the sum of an exponential disc and a bulge, either via a semi-analitycal formula, or using numerical integration.
    
    How to use
    ----------
        There are two ways to use this function. Either using numerical integration of the light profiles, or by finding the zero of a specific equation. 
        In both cases, the parameter gal is mandatory. This corresponds to a numpy structured array with the following fields: 'Mag_d_GF', 'Mag_b_GF', 'R_d_GF' and 'R_b_GF'. 
        HOWEVER, if the flag noStructuredArray is True, this array will not be used (so just cast anything into this parameter, it will not matter) but instead, the optional parameters magD, magB, Rd and Rb must be provided.
        
        The guess can be ignored, though the result may not converge.
        b1 and b4 values do not necessarily need to be provided if you only call this function very few times. If not, they will be computed once at the beginning and propagated in subsequent function calls.
            
            a) Numerical integration
                This method will find the zero of the following function f(r) = 2\pi \integral_0^r [ exponential_disc(some parameters) + \integral_0^r bulge(some parameters) ] rdr  - L_{tot},
                where L_{tot} is the total integrated luminosity of the sum of the disc and the bulge.
                
                The Ltot parameter is not mandatory, as it will be computed if not provided. However, if not provided, this requires to give magnitude values (this is mandatory in any case) AND a magnitude offset value in order to compute Ltot.
                The Ie parameter can be given or can be ignored. In the latter case, it will be computed using the magnitudes and magnitude offset, so this last parameter should be provided as well in this case.
            
            b) re equation
                This is an experimental feature (which seems to work fine though). It follows from analytically computing the equation for re using its definition as well as the sum of an exponential disc and a bulge.
                
                In this case, the integration parameter must be set False.
                THE FOLLOWING PARAMETERS ARE NOT REQUIRED FOR THIS METHOD: Ltot, Ie, offset
                
                Basically, the simplest way to solve re is to call the function the following way:
                    
                    solve_re(array)
                    
                where array is a numpy structured array with the relevant columns.
                
    Additional information
    ----------------------
        For only one galaxy, only a scalar value may be provided for each parameter you would like to pass. However, for more than one galaxy, a list must be given instead. The parameters which require a list when solving for more than one galaxy are represented by 'type/list', where type can be int, float, bool, etc. (given after the parameter name in the list below).
    
    Mandatory inputs
    ----------------
        gal : numpy structured array
            structured array with data for all the galaxies. The required column names are 'R_d_GF' (re for the disk component), 'R_b_GF' (re for the bulge component), 'Mag_d_GF' (the total integrated magnitude for the disk component), 'Mag_b_GF' (the total integrated magnitude for the bulge component).

    Optional inputs
    ---------------
        b1 : float
            b1 factor appearing in the Sersic profile of an exponential disc, defined as $\gamma(2, b1) = 1/2$. By default, b1 is None, and its value will be computed by the function. To skip this computation, please give it a value when callling the function.
        b4 : float
            b4 factor appearing in the Sersic profile of a bulge, defined as $2\gamma(8, b4) = 7!$. By default, b4 is None, and its value will be computed by the function. To skip this computation, please give it a value when callling the function.
        guess : float/list of floats
            guess for the value of re for all the galaxies
        Ie : floats/list of floats
            intensity at half-light radius for all the galaxies (including both profiles)
        integration : bool
            whether to find re integrating the light profiles or not (i.e. solving the re equation).
        Ltot : float/list of floats
            total luminosity of the galaxies. This parameter is used when finding re using numerical integration of the light profiles. If integration is True and no Ltot is provided, it will be computed using the total magnitude of each component and the offset value.
        magB : float/list of floats
            total magnitude of the bulge component of the galaxies
        magD : float/list of floats
            total magnitude of the disk component of the galaxies
        method : str
            method to use to find the zero of the re equation function or the integral to solve
        normalise : boolean
            whether to normalise the equation or not. It is recommended to do so to improve the convergence.
        noStructuredArray : boolean
            if False, the structured array gal will be used. If False, values of the magnitudes and half-light radii of the two components must be given.
        offsetMagD : float/list of floats
            magnitude offset used in the magnitude system for the disk component
        offsetMagB : float/list of floats
            magnitude offset used in the magnitude system for the bulge component
        Rb : float/list of floats
            half-light radius of the bulge components of the galaxies
        Rd : float/list of floats
            half-light radius of the disk components of the galaxies
        stretch : float
            dilatation factor used to multiply re in order to smooth out the sharp transition around the 0 of the function
        useZeroOder : boolean
            whether to use the zero order analytical solution as a guess. If True, the value of guess will be used by the zero search algorithm.
        verbose : bool
            whether to print messaged on stdout (True) or not (False). Default is True.
        xtol : float
            relative error convergence factor
            
    Return the value of re for all the galaxies, as well as a 
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
    Compute the central intensity of a given Sersic profile.
    
    Mandatory inputs
    ----------------
        n : int/float
            Sersic index of the profile
        re : float
            half-light radius
        Ie : float
            intensity at Re. If None, values for mag and offset must be given instead.
        mag : float
            total magnitude. If None, Ie must be given instead.
        offset : float
            magnitude offset. If None, Ie must be given instead

    Return the central intensity of the Sersic profile.
    '''
    
    bn = compute_bn(n)
    
    if Ie is None:
        if mag is not None and offset is not None:
            Ie = intensity_at_re(n, mag, re, offset, bn=bn)
        else:
            raise ValueError("Ie value is None, but mag and/or offset is/are also None. If no Ie is given, please provide a value for the total magnitude and magnitude offset in order to compute the former one. Cheers !")
    
    return Ie*np.exp(bn)


def compute_R22(Red, dRed=None, b1=None):
    '''
    Compute R22 (and its error) given an array of disk effective radii.

    Mandatory parameter
    -------------------
        Red : float or numpy array of floats
            disk effective radii
            
    Optional parameters
    -------------------
        b1 : float
            Usual b1 factor appearing in exponential disc profiles. If not provided, its value will be computed.
        dRed : float or numpy array of floats
            error estimate on the effective radii

    Return R22 (and its error).
    '''
    
    b1, = check_bns([1], [b1])
    
    if dRed is None:
        return 2.2*Red/b1
    else:
        return 2.2*Red/b1, 2.2*dRed/b1