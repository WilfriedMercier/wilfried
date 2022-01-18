#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Functions and classes related to angular momenta of galaxies.
"""

import numpy           as     np
from   scipy.special   import gammainc, gamma
from   scipy.integrate import quad
from   .models         import compute_bn, sersic_profile
from   typing          import Union, List, Tuple, Callable

def sersic_kthMoment(k: Union[int, float], vmin: Union[int, float], vmax: Union[int, float], 
                     n: Union[int, float] = 1, Re: Union[int, float] = 10, Ie: Union[int, float] = 10) -> Union[int, float]:
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

class SersicMomentum:
    '''
    Class which computes the angular momentum for a single Sérsic profile in the **unnormalised** case
    
    .. math::
        J_z = 2\pi \int_0^\infty dR~R^2 \Sigma (R) V(R),
        
    and, in the **normalised** case
    
    .. math::
        j = \frac{\int_0^\infty dR~R^2 \Sigma (R) V(R)}{\int_0^\infty dR~R \Sigma (R)},
        
    where 
    
    :math:`\Sigma(R) &= Ie \times e^{-b_n \left [ (R/R_e)^{1/n} - 1 \right ]}`.
    '''
    
    def __init__(self,  n: Union[int, float] = 1, Re: Union[int, float] = 10, Ie: Union[int, float] = 10) -> None:
        '''
        Init method.
        
        :param Ie: (**Optional**) Surface brightness at Re
        :type Ie: int or float
        :param n: (**Optional**) Sérsic index
        :type n: int or float
        :param Re: (**Optional**) effective radius
        :type Re: int or float
        
        :raises ValueError if :
            * n < 0
            * Re <= 0
            * Ie < 0
        '''
        
        if n < 0:
            raise ValueError(f'Sérsic index n is {n} but it must be positive.')
            
        if Re <= 0:
            raise ValueError(f'Effective radius Re is {Re} but it must be strictly positive.')
            
        if Ie < 0:
            raise ValueError(f'Surface brightness at Re is {Ie} but it must be positive.')
        
        self.n     = n
        self.Re    = Re
        self.Ie    = Ie
        
        #: Normalisation factor, equal to the total flux
        self.norm  = 2*np.pi*sersic_kthMoment(1, 0, np.inf, n=self.n, Re=self.Re, Ie=self.Ie)
        
    def linear_ramp(self, r: Union[int, float], rt: Union[int, float], vt: Union[int, float], 
                    normalise = True) -> Union[int, float]:
        '''
        Compute the angular momentum using a linear ramp model up to radius r whose rotation curve is
        
        .. math::
            V(R) &= V_t \times R/r_t \ \ \rm{if}\ \ R \leq r_t \ \ \rm{else} \ \ V_t
            
        :param r: radius where the angular momentum is computed
        :type r: int or float
        :param rt: kinematical transition radius. Must be of the same unit as Re.
        :type rt: int or float
        :param vt: kinematical plateau velocity (must be positive)
        :type vt: int or float
        
        :param bool normalise: (**Optional**) whether to normalise by the first order moment of the light distribution. Default is True.
        
        :returns: central angular momentum along the vertical axis. When normalised, the unit is that of rt*vt.
        :rtype: int or float
        :raises ValueError: if
            * r < 0
            * rt <= 0 
            * vt <= 0
        '''
        
        if r < 0:
            raise ValueError(f'Radius r is {r} but is must be strictly positive.')
        
        if rt<=0:
            raise ValueError(f'Transition radius is {rt} but it must be positive only.')
            
        if vt<=0:
            raise ValueError(f'Plateau velocity is {vt} but it must be positive only.')
            
        if normalise:
            norm = self.norm
        else:
            norm = 1
            
        # The velocity curve is vt * r/rt for r<rt and vt for r>rt
        # Therefore, two parts must be computed:
        #  1. between 0 and rt as a 3rd order moment divided by rt
        #  2. between rt and r as a 2nd order moment
        
        if r <= rt:
            inner = sersic_kthMoment(3, 0,  r,  n=self.n, Re=self.Re, Ie=self.Ie)/rt
            outer = 0
        else:
            inner = sersic_kthMoment(3, 0,  rt, n=self.n, Re=self.Re, Ie=self.Ie)/rt
            outer = sersic_kthMoment(2, rt, r,  n=self.n, Re=self.Re, Ie=self.Ie)
            
        J         = 2*np.pi*vt*(inner+outer)
        
        return J/norm
    
    def custom_rotation_curve(self, rotation_curve: Callable[..., Union[int, float]],
                              r: Union[int, float], normalise: bool = True, 
                              args: List = []) -> Tuple[float, float]:
        '''
        Compute the angular momentum using a custom rotation curve up to radius r.
        
        :param func rotation_curve: function representing the total rotation curve used to compute the angular momentum. Its first parameter must always be the radial distance r.
        :param r: distance up to which to compute the angular momentum
        :type r: int or float
        
        :param bool normalise: (**Optional**) whether to normalise by the first order moment of the light distribution. Default is True.
        
        :param args: (**Optional**) arguments to pass to the rotation curve function. Default is not argument (empty list).
        
        :returns: central angular momentum along the vertical axis using the custom rotation curve and its error. When normalised the unit is similar to **r** * **rotation_curve**
        :rtype: tuple of int or float
        
        :raises TypeError: if **rotation_curve** is not a callable function
        :raises ValueError: if **r** < 0
        '''
        
        def the_integral(r: Union[int, float], args_sigma: List[Union[int, float]], args_V: List[Union[int, float]]) -> float:
            '''
            Function to integrate.
            
            :param r: radius where the function is computed
            :type r: int or float
            :param list args_sigma: arguments to pass to the surface brightness function
            :param list args_V: arguments to pass to the rotation curve function
            
            :returns: function computed at the given radius and its error from the integration
            :rtype: float, float
            '''
            
            return 2*np.pi * r * r * sersic_profile(r, *args_sigma) * rotation_curve(r, *args_V)
        
        if not callable(rotation_curve):
            raise TypeError(f'rotation_curve is of type {type(rotation_curve)} but it must be a callable function.')
            
        if r<0:
            raise ValueError(f'Radius r is {r} but it must be positive.')
            
        if normalise:
            norm   = self.norm
        else:
            norm   = 1
            
        # Arguments to pass to the function sersic_profile defined in models library
        args_sigma = [self.n, self.Re, self.Ie, None, None, None]

        # Integrate the angular momentum up to raadius r
        J, err     = quad(the_integral, 0, r, args=(args_sigma, args))
        
        return J/norm, err/norm
        
        

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
        norm = 2*np.pi*sersic_kthMoment(1, 0, np.inf, n=n, Re=Re, Ie=Ie)
    else:
        norm = 1
        
    # The velocity curve is vt * r/rt for r<rt and vt for r>rt
    # Therefore, two parts must be computed:
    #  1. between 0 and rt as a 3rd order moment divided by rt
    #  2. between rt and infinity as a 2nd order moment
    
    inner = sersic_kthMoment(3, 0,  rt,     n=n, Re=Re, Ie=Ie)/rt
    outer = sersic_kthMoment(2, rt, np.inf, n=n, Re=Re, Ie=Ie)
        
    return 2*np.pi*vt*(inner+outer)/(norm)