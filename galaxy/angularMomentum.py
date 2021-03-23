#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*Author:* Wilfried Mercier - IRAP

Computations related to angular momenta of galaxies.
"""

import numpy         as     np
from   scipy.special import gammainc, gamma
from   .models       import compute_bn

def sersic_kthMoment(k, vmin, vmax, n=1, Re=10, Ie=10):
    r'''
    Compute the kth radial moment for a Sérsic profile
    
    .. math::
        \int_{v_{\rm{min}}}^{v_{\rm{max}}} dr~r^k \Sigma(r)

    :param k: order of the moment
    :type k: int or float
    :param vmin: lower bound to compute the kth moment. Must be greater than 0 and less than vmax. Should be the same unit as Re.
    :type vmin: int or float
    :param vmax: upper bound to compute the kth moment. Must be greater than 0 and more than vmin. Should be the same unit as Re.
    :type vmax: int or float
    
    :param Ie: (**Optional**) flux at Re
    :type Ie: int or float
    :param n: (**Optional**) Sérsic index
    :type n: int or float
    :param Re: (**Optional**) effective radius
    :type Re: int or float
    
    :returns: kth order moment
    :rtype: int or float
    :raises ValueError: if vmin<0 or vmin >= vmax
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
    r'''
    Compute the analytical angular momentum for a single Sérsic profile and a ramp model rotation curve in the **unnormalised** case
    
    .. math::
        J_z = 2\pi \int_0^\infty dR~R^2 \Sigma (R) V(R),
        
    and, in the **normalised** case
    
    .. math::
        j = \frac{\int_0^\infty dR~R^2 \Sigma (R) V(R)}{\int_0^\infty dR~R \Sigma (R)},
        
    where 
    
    .. math::
        \Sigma(R) &= Ie \times e^{-b_n \left [ (R/R_e)^{1/n} - 1 \right ]}
        
        V(R) &= V_t \times R/r_t \ \ \rm{if}\ \ R \leq r_t \ \ \rm{else} \ \ V_t
        
    :param rt: kinematical transition radius
    :type rt: int or float
    :param vt: kinematical plateau velocity (must be positive)
    :type vt: int or float
    
    :param Ie: (**Optional**) flux at Re
    :type Ie: int or float
    :param n: (**Optional**) Sérsic index
    :type n: int or float
    :param bool normalise: (**Optional**) whether to normalise by the first order moment of the light distribution. Default is True
    :param Re: (**Optional**) effective radius. Must have the same unit as rt. Default is 10.
    :type Re: int or float
    
    :returns: central angular momentum along the vertical axis. When normalised, the unit is that of rt*vt.
    :rtype: int or float
    :raises ValueError: if rt <= 0 or vt <= 0
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