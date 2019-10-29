#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:09:08 2019

@author: Wilfried Mercier - IRAP

Useful functions for galaxy modelling and other related computation
"""

import numpy as np
from scipy.special import gammaincinv, gammainc, gamma
from scipy.optimize import root
from scipy.integrate import quad
from math import factorial

#################################################################################################################
#                                           Sersic profiles                                                     #
#################################################################################################################

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


#################################################################################################################
#                                           Sersic luminosities                                                 #
#################################################################################################################
    

def BoverD(r, rd, rb, b1=None, b4=None, Ied=None, Ieb=None, magD=None, magB=None, offsetD=None, offsetB=None):
    """
    Computes the ratio of the bulge luminosity (B) over the disk one (D) of a two Sersic components galaxy up to radius r.
    
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
            
    Returns the B/D ratio at all the given positions.
    """
    
    #compute b1 and b4 if not given
    b1, b4 = check_bns([1, 4], [b1, b4])
    Ied    = checkAndComputeIe(Ied, 1, b1, rd, magD, offsetD)
    Ieb    = checkAndComputeIe(Ieb, 4, b4, rb, magB, offsetB)
        
    return luminositySersic(r, 4, rb, bn=b4, Ie=Ieb)[0] / luminositySersic(r, 1, rd, bn=b1, Ie=Ied)[0]    


def BoverT(r, rd, rb, b1=None, b4=None, Ied=None, Ieb=None, magD=None, magB=None, offsetD=None, offsetB=None):
    """
    Computes the ratio of the bulge luminosity (B) over the total one (T=D+B) of a two Sersic components galaxy up to radius r.
    
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
            
    Returns the B/T ratio at all the given positions.
    """
    
    #compute b1 and b4 if not given
    b1, b4 = check_bns([1, 4], [b1, b4])
    Ied    = checkAndComputeIe(Ied, 1, b1, rd, magD, offsetD)
    Ieb    = checkAndComputeIe(Ieb, 4, b4, rb, magB, offsetB)
        
    return luminositySersic(r, 4, rb, bn=b4, Ie=Ieb)[0] / luminositySersics(r, [1, 4], [rd, rb], listbn=[b1, b4], listIe=[Ied, Ieb])[0]    


def DoverT(r, rd, rb, b1=None, b4=None, Ied=None, Ieb=None, magD=None, magB=None, offsetD=None, offsetB=None):
    """
    Computes the ratio of the disk luminosity (D) over the total one (T=D+B) of a two Sersic components galaxy up to radius r.
    
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
            
    Returns the D/T ratio at all the given positions.
    """
    
    #compute b1 and b4 if not given
    b1, b4 = check_bns([1, 4], [b1, b4])
    Ied    = checkAndComputeIe(Ied, 1, b1, rd, magD, offsetD)
    Ieb    = checkAndComputeIe(Ieb, 4, b4, rb, magB, offsetB)
        
    return luminositySersic(r, 4, rb, bn=b4, Ie=Ieb)[0] / luminositySersics(r, [1, 4], [rd, rb], listbn=[b1, b4], listIe=[Ied, Ieb])[0]   

    
def luminositySersic(r, n, re, bn=None, Ie=None, mag=None, offset=None, start=0.0):
    """
    Compute the luminosity of a single Sersic profile of index n up to radius r.
    
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
            
    Returns the integrated luminosity up to radius r and an estimation of its absolute error as a dictionary. If a list of radii is given, it returns a list of luminosities and a list of absolute errors.
    """
    
    #the integral we need to compute to have the luminosity
    def the_integral(r, n, re, Ie=None, bn=None, mag=None, offset=None):
        return 2*np.pi*sersic_profile(r, n, re, Ie=Ie, bn=bn, mag=mag, offset=offset)*r
    
    bn, = check_bns([n], [bn])
    Ie = checkAndComputeIe(Ie, n, bn, re, mag, offset)
    if Ie is None:
        return None
        
    #if r has no length (not a list), simply return the integral and its error
    try:
        lr=len(r)
    except TypeError:
        integral, error = quad(the_integral, start, r, args=(n, re, Ie, bn, mag, offset))
        return {'value':integral, 'error':error}
    
    #else compute for each radius in the list
    integral = np.zeros(lr)
    error    = np.zeros(lr)
    for pos in range(lr):
        integral[pos],  error[pos] = quad(the_integral, start, r[pos], args=(n, re, Ie, bn, mag, offset))
        
    return {'value':integral, 'error':error}
    

def luminositySersics(r, listn, listRe, listbn=None, listIe=None, listMag=None, listOffset=None):
    """
    Compute the luminosity of a sum of Sersic profiles up to radius r.
    
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
        listbn : list of floats/None
            list of bn factors appearing in Sersic profiles defined as $2\gamma(2n, bn) = \Gamma(2n)$. If no value is given, each bn will be computed according to their respective Sersic index. If you do not want this function to compute the value of one of the bn, provide its value in the list, otherwise put it to None.
        listIe : list of floats
            list of intensities at re for each profile
        listMag : list of floats
            list of total integrated magnitudes for each profile
        listOffset : list of floats
             list of magnitude offsets used in the magnitude system for each profile
         
    Returns the integrated luminosity of the sum of all the given Sersic profiles and an estimation of the error
    """
    
    #if no list of bn values is given, compute them all
    if listbn is None:
        listbn         = [compute_bn(n) for n in listn]
    listbn             = check_bns(listn, listbn)
    
    if listIe is None:
        if listMag is not None and listOffset is not None:
            listIe     = intensity_at_re(np.array(listn), np.array(listMag), np.array(listRe), np.array(listOffset), bn=np.array(listbn))
        else:
            print("ValueError: listIe is None, but listMag or listOffset is also None. If no listIe is given, please provide a value for the total magnitude and magnitude offset in order to compute the intensities.")
            return None 
    
    res               = 0
    err               = 0
    for n, re, ie, bn in zip(listn, listRe, listIe, listbn):
        lum           = luminositySersic(r, n, re, bn=bn, Ie=ie)
        res          += lum['value']
        err          += lum['error']
        
    return {'value':res, 'error':err}


def total_luminosity(mag, offset, factor=1.0):
    """
    Gives the total integrated luminosity of galaxies from their magnitude and magnitude offset
    
    Mandatory inputs
    ----------------
        offset : float/list of floats
            magnitude offset used in the definition of the magnitude system
        mag : float/list of floats
            magnitude of the galaxies
            
    Optional inputs
    ---------------
        factor : float/list of floats
            a multiplicative factor before the power of 10. This can be useful as the definition used for the magnitude is -2.5\log_{10} (L_{tot}), but sometimes, the definition -2.5\log_{10} (F_{tot}) with F_{tot} = L_{tot} /(4\pi D^2) (the total flux), and D the luminosity distance, is used instead. In this case, a multiplicative factor of 4\pi D^2 must be present before the power of 10 to compute the real luminosity.
        
    Returns the total luminosity
    """
    
    return np.array(factor) * 10**((np.array(offset)-np.array(mag))/2.5)


#################################################################################################################
#                                 Half-light radius computation                                                 #
#################################################################################################################
    
def jacobian_re_equation(re, gal, b1=None, b4=None, noStructuredArray=False, magD=None, magB=None, Rd=None, Rb=None, norm=1.0, stretch=1.0):
    """
    Compute the jacobian of the re equation function.
    
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
        Rb : float/list of floats
            half-light radius of the bulge components of the galaxies
        Rd : float/list of floats
            half-light radius of the disk components of the galaxies
        stretch : float
            dilatation factor used to multiply re in order to smooth out the sharp transition around the 0 of the function
            
    Return the value of the jacobian of the left-hand side of the equation.
    """
    
    b1, b4 = check_bns([1, 4], [b1, b4])
    magD, magB, Rd, Rb = fromStructuredArrayOrNot(gal, magD, magB, Rd, Rb, noStructuredArray)
    
    #convert dilate to float to avoid numpy casting operation errors
    re = re[0]*float(stretch)
    
    try:
        re = [0 if r<0 else r for r in re]
    except:
        if re < 0:
            re = 0
            
    prefactor = 10**(-magD/2.5) * (b1/Rd)**2 * np.exp(-b1*re/Rd) + 10**(-magB/2.5) * (b4/Rb)**8 * np.exp(-b4*(re/Rb)**(1.0/4.0)) / (4.0*gamma(8))
    return [prefactor*re]

def the_re_equation_for_2_Sersic_profiles(re, gal, b1=None, b4=None, noStructuredArray=False, magD=None, magB=None, Rd=None, Rb=None, norm=1.0, stretch=1.0):
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
        Rb : float/list of floats
            half-light radius of the bulge components of the galaxies
        Rd : float/list of floats
            half-light radius of the disk components of the galaxies
        stretch : float
            dilatation factor used to multiply re in order to smooth out the sharp transition around the 0 of the function
        
    Return the value of the left-hand side of the equation. If re is correct, the returned value should be close to 0.
    """

    b1, b4 = check_bns([1, 4], [b1, b4])
    magD, magB, Rd, Rb = fromStructuredArrayOrNot(gal, magD, magB, Rd, Rb, noStructuredArray)
    
    #convert dilate to float to avoid numpy casting operation errors
    re = re*float(stretch)
    
    try:
        re = [0 if r<0 else r for r in re]
    except:
        if re < 0:
            re = 0
    
    return ( 10**(-magD/2.5)*(gammainc(2, b1*(re/Rd)) - 0.5) + 10**(-magB/2.5)*(gammainc(8, b4*(re/Rb)**(1.0/4.0)) - 0.5) ) / norm


def solve_re(gal, guess=None, b1=None, b4=None, noStructuredArray=False, magD=None, magB=None, Rd=None, Rb=None, normalise=True, stretch=5e-2,
             integration=False, Ltot=None, Ie=None, offsetMagD=None, offsetMagB=None, xtol=1e-3, useZeroOrder=False, method='hybr', jacobian='numerical'):
    """
    This is meant to find the half-light radius of the sum of an exponential disc and a bulge, either via a semi-analitycal formula, or using numerical integration.
    
    How to use
    ----------
        There are two ways to use this function. Either using numerical integration of the light profiles, or by finding the zero of a specific equation. 
        In both cases, the parameter gal is mandatory. This corresponds to a numpy structured array with the following fields: 'Mag_d_GF', 'Mag_b_GF', 'R_d_GF' and 'R_b_GF'. HOWEVER, if the flag noStructuredArray is True, this array will not be used (so just cast whatever into this parameter) and instead, the optional parameters magD, magB, Rd and Rb must be given.
        
        The guess can be ignored, though the result may not converge.
        b1 and b4 values do not necessarily need to be provided if you only call this function very few times. If not, they will be computed once at the beginning and propagated in subsequent function calls.
            
            a) Numerical integration
                This method will find the zero of the following function f(r) = 2\pi \integral_0^r [ exponential_disc(some parameters) + \integral_0^r bulge(some parameters) ] rdr  - L_{tot},
                where L_{tot} is the total integrated luminosity of the sum of the disc and the bulge.
                
                The Ltot parameter is not mandatory, as it will be computed if not provided. However, if not provided, this requires to give magnitude values (this is mandatory in any case) AND a magnitude offset value in order to compute Ltot.
                The Ie parameter can be given or can be ignored. In the latter case, it will be computed using the magnitudes and magnitude offset, so this last parameter should be provided as well in this case.
            
            b) re equation
                This still an experimental feature. It basically follows from analytically computing the equation for re using its definition as well as the sum of an exponential disc and a bulge.
                
                In this case, the integration parameter must be False.
                THE FOLLOWING PARAMETERS ARE NOT REQUIRED FOR THIS METHOD: Ltot, Ie, offset
                
                Basically, the simplest way to solve re is to call the function the following way:
                    
                    solve_re(array)
                    
                where array is a numpy structured array with the relevant columns.
                
    Additional information
    ----------------------
        For only one galaxy, only a scalar value may be provided for each parameter you would like to pass. However, for more than one galaxy, a list must be given instead. The parameters which require a list when solving for more than one galaxy are represented by type/list of type where type can be int, float, bool, etc. after the parameter name in the list below.
    
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
        jacobian : str of function name
            jacobian to use:
                - if 'numerical', the jacobian will be numerically computed
                - if 'analytical', the jacobian will be computed using the analytical derivative given by jacobian_re_equation function
                - if a function name, the given function will be used, but it must take the same arguments as the_re_equation_for_2_sersic_profiles
                
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
            whether to use the zero order analytical solution as a guess. If True, the value of guess will not be used by the zero search algorithm.
        xtol : float
            relative error convergence factor
            
    Return the value of re for all the galaxies, as well as a 
    """
    
    #to solve numerically we find the zero of the difference between the integral we want to solve and half the total luminosity
    def integral_to_solve(r, listn, listRe, listbn, listIe, listMag, listOffset, Ltot):
        res, err     = luminositySersics(r, listn, listRe, listbn=listbn, listIe=listIe, listMag=listMag, listOffset=listOffset)
        return res-Ltot/2.0
        
    b1, b4 = check_bns([1, 4], [b1, b4])
    magD, magB, Rd, Rb = fromStructuredArrayOrNot(gal, magD, magB, Rd, Rb, noStructuredArray)
        
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
    
    #set ouput arrays empty
    solution         = np.array([])
    convFlag         = np.array([], dtype=bool)
    debug            = np.array([])
    
    #compute re from the re equation instead of integrating
    if not integration:
        
        #define the normalisation of the equation
        if normalise:
            norm     = 0.5*(10**(-magD/2.5) + 10**(-magB/2.5))
        else:
            norm     = 1.0
            
        if jacobian == 'numerical':
            jacobian = False
        elif jacobian == 'analytical':
            jacobian = jacobian_re_equation
        
        #solve by finding the zero of the function
        for g, md, mb, rd, rb, nm in zip(guess, magD, magB, Rd, Rb, norm):
            sol      = root(the_re_equation_for_2_Sersic_profiles, g, 
                           args=(None, b1, b4, True, md, mb, rd, rb, nm, stretch), 
                           jac=jacobian,
                           method=method, options={'xtol':xtol, 'maxfev':200})
            
            # solution tested in the_re_equation is multiplied by the stretch factor, so that the best-solution*stretch gives the_re_equation \approx 0
            # thus the best-solution needs to be multiplied by the stretch factor at the end to recover the true valueS
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
            Ltot     = total_luminosity(magD, offsetMagD) + total_luminosity(magB, offsetMagB)
            
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

def check_bns(listn, listbn):
    """
    Given a list of bn values, check those which are not given (i.e. equal to None), and compute their value using the related Sersic index.
    
    Mandatory inputs
    ----------------
        listbn : list of floats
            list of bn values
        listn : list of floats/int
            list of Sersic indices
        
    Return a complete list of bn values.
    """
    
    return [compute_bn(listn[pos]) if listbn[pos] is None else listbn[pos] for pos in range(len(listn))]


def compute_bn(n):
    """
    Compute the value of bn used in the definition of a Sersic profile (defined as $2\gamma(2n, bn) = \Gamma(2n)$).
    
    Mandatory inputs
    ----------------
        n : float/int
            Sersic index of the profile
        
    Returns the value of bn for each n.
    """
    
    try:
        res = [gammaincinv(2*i, 0.5) for i in n]
    except:
        res = gammaincinv(2*n, 0.5)
    return res


def intensity_at_re(n, mag, re, offset, bn=None):
    """
    Computes the intensity of a given Sersic profile with index n at the position of the half-light radius re. This assumes to know the integrated magnitude of the profile, as well as the offset used for the magnitude definition.
    
    Mandatory inputs
    ----------------       
        mag : float
            total integrated magnitude of the profile
        n : float/int
            Sersic index of the given profile
        offset : float
            magnitude offset used in the defition of the magnitude system
        re : float
            effective (half-light) radius

    Optional inputs
    ---------------
        bn : float
            bn factor appearing in the Sersic profile defined as $2\gamma(2n, bn) = \Gamma(2n)$. By default, bn is None, and its value will be computed by the function using the value of n. To skip this computation, please give a value to bn when callling the function.
    
    Return the intensity at re
    """
 
    if bn is None:
        bn = compute_bn(n)
    
    return 10**((offset - mag)/2.5 - 2.0*np.log10(re) + 2.0*n*np.log10(bn) - bn/np.log(10)) / (2.0*np.pi*n*gamma(2.0*n))


def checkAndComputeIe(Ie, n, bn, re, mag, offset):
    """
    Just check if Ie is not given but the magnitude and magnitude offset are, and if so compute it.
    
    Mandatory inputs
    ---------------- 
        bn : float
            bn factor appearing in the Sersic profile defined as $2\gamma(2n, bn) = \Gamma(2n)$ 
        Ie : float
            intensity at effective radius
        mag : float
            total integrated magnitude
        n : float/int
            Sersic index
        offset : float
            magnitude offset
        re : float
            half-light/effective radius
    
    Return Ie if it could be computed or already existed, or None if a value was missing.
    """
    
    if Ie is None:
        if mag is not None and offset is not None:
            return intensity_at_re(n, mag, re, offset, bn=bn)
        else:
            print("ValueError: Ie value is None, but mag and/or offset is/are also None. If no Ie is given, please provide a value for the total magnitude and magnitude offset in order to compute the former one.")
            return None
    else:
        return Ie


####################################################################################################
#                                Miscellanous functions                                            #
####################################################################################################
    
def fromStructuredArrayOrNot(gal, magD, magB, Rd, Rb, noStructuredArray):
    """
    Store values of the disk magnitude, the bulge magnitude, the disk effective radius and the bulge effective radius either directly from the given values or from a numpy structured array.
    
    Mandatory inputs
    ---------------- 
        gal : numpy structured array
            structured array with data for all the galaxies. The required column names are 'R_d_GF' (re for the disk component), 'R_b_GF' (re for the bulge component), 'Mag_d_GF' (the total integrated magnitude for the disk component), 'Mag_b_GF' (the total integrated magnitude for the bulge component).
        magB : float/list of floats
            total magnitude of the bulge component of the galaxies
        magD : float/list of floats
            total magnitude of the disk component of the galaxies
        Rb : float/list of floats
            half-light radius of the bulge components of the galaxies
        Rd : float/list of floats
            half-light radius of the disk components of the galaxies
        noStructuredArray : boolean
            if False, the structured array gal will be used. If False, values of the magnitudes and half-light radii of the two components must be given.
    
    Return the disk magnitude, the bulge magnitude, the disk effective radius and the bulge effective radius.
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