#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:46:39 2020

@author: wilfried

Computations related to angular momenta of galaxies.
"""

import numpy         as     np
from   scipy.special import gammainc, gamma
from   .models       import compute_bn

def sersic_kthMoment(k, vmin, vmax, n=1, Re=10, Ie=10):
    '''
    Compute the kth radial moment for a Sérsic profile: \int_vmin^max \Sigma r^k dr.

    Parameters
    ----------
        k : int/float
            order of the moment
        vmin: int/float
            lower bound to compute the kth moment. Must be greater than 0 and less than vmax. Should be the same unit as Re.
        vmax : int/float
            upper bound to compute the kth moment. Must be greater than 0 and more than vmin. Should be the same unit as Re.
        
    Optional parameters
    -------------------
        Ie : int/float
            flux at Re
        n : int/float
            Sérsic index
        Re : int/float
            effective radius

    Return the kth order moment.
    '''
    
    if vmin < 0 or vmin >= vmax:
        raise ValueError('Either vmin or vmax is wrong.')
        
    bn        = compute_bn(n)
    prod      = n*(k+1)
    prefactor = (Ie*n*Re**(k+1)*np.exp(bn))/(bn**prod)
    gamma1    = gammainc(prod, bn*(vmax/Re)**(1/n))
    gamma2    = gammainc(prod, bn*(vmin/Re)**(1/n))
    
    return prefactor*gamma(prod)*(gamma1 - gamma2)

def momentum(rt, vt, n=1, Re=10, Ie=10, normalise=True):
    '''
    Compute the analytical angular momentum for a single Sérsic profile and a ramp model rotation curve.

    Parameters
    ----------
        rt : int/float
            kinematical transition radius
        vt : int/float
            kinematical plateau velocity (must be positive)
            
    Optional parameters
    -------------------
        Ie : int/float
            flux at Re
        n : int/float
            Sérsic index
        normalise : bool
            whether to normalise by the first order moment of the light distribution
        Re : int/float
            effective radius. Must have the same unit as rt.

    Return the central angular momentum along the vertical axis. When normalised, the unit is that of rt*vt.
    '''
    
    if rt<=0:
        raise ValueError('Transition radius must be positive only. Cheers !')
        
    if vt<=0:
        raise ValueError('Plateau velocity must be positive only. Cheers !')
        
    if normalise:
        norm = sersic_kthMoment(1, 0, np.inf, n=n, Re=Re, Ie=Ie)
    else:
        norm = 1
        
    # The velocity curve is vt * r/rt for r<rt and vt for r>rt
    # Therefore, two parts must be computed:
    #  1. between 0 and rt as a 3rd order moment divided by rt
    #  2. between rt and infinity as a 2nd order moment
    
    inner = sersic_kthMoment(3, 0,  rt,     n=n, Re=Re, Ie=Ie)/rt
    outer = sersic_kthMoment(2, rt, np.inf, n=n, Re=Re, Ie=Ie)
        
    return vt*(inner+outer)/norm