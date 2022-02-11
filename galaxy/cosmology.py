#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

A set of functions to easily compute standard calculations in extragalactic physics using cosmology. 

.. note::
    This relies heavily on a custom, Python 3 adapted version of `cosmolopy <https://roban.github.io/CosmoloPy/>`_.
"""

from   typing              import Union, Dict, Optional, Tuple, List
from   scipy.optimize      import root
from   astropy.coordinates import SkyCoord
import cosmolopy.distance  as     cd
import cosmolopy.constants as     cc

#: Default cosmology
COSMOLOGY = cd.set_omega_k_0({'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.72})

def angular_diameter_size(z: Union[int, float], theta: Union[int, float], 
                          scaleFactor: Union[int, float]=1.0, cosmology: Optional[Dict]=None) -> float:
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the size of an object with extent **theta** at redshift **z** from the angular diameter distance.

    :param z: redshift of the object
    :type z: int or float
    :param theta: angle subtended by the object. Unit should be in radians. If not, provide a scaling factor with **scaleFactor**.
    :type theta: int or float
    
    :param dict cosmology: (**Optional**) parameters for the desired cosmology. See `cosmolopy <https://roban.github.io/CosmoloPy/>`_ for more information.
    :param scaleFactor: (**Optional**)  if theta is not in radian, please provide a correct scale factor following the formula
    
        .. math::
            \theta [{\rm{rad}}] = \theta [{\rm{your~unit}}] \times scaleFactor [{\rm{rad/your~unit}}].
            
    :type scaleFactor: int or float
    
    :returns: physical size in **kpc**
    :rtype: float
    '''
    
    if cosmology is None:
        cosmology = COSMOLOGY
        
    return cd.angular_diameter_distance(z, **cosmology)*theta*scaleFactor*1000
    

def comoving_separation(z: Union[int, float], theta: Union[int, float], 
                        cosmology: Optional[Dict]=None) -> float:
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the tranverse distance (at fixed redshift) between objects separated by an angle **theta** on the sky in comoving units.

    :param z: redshift of the object
    :type z: int or float
    :param theta: angle subtended by the object. Unit should be in radians. If not, provide a scaling factor with **scaleFactor**.
    :type theta: int or float
    
    :param dict cosmology: (**Optional**) parameters for the desired cosmology. See `cosmolopy <https://roban.github.io/CosmoloPy/>`_ for more information.
    
    :returns: comoving separation in **Mpc**
    :rtype: float
    '''
    
    if cosmology is None:
        cosmology = COSMOLOGY
    
    cosmology     = cd.set_omega_k_0(cosmology)
    return cd.comoving_distance_transverse(z, **cosmology)*theta

def comoving_los(z1: Union[int, float], z2: Union[int, float], 
                 cosmology: Optional[Dict]=None) -> float:
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the line of sight comoving distance between objects.

    :param z1: redshift of the 1st object
    :type z1: int or float
    :param z2: redshift of the 2nd object
    :type z2: int or float
    
    :param dict cosmology: (**Optional**) parameters for the desired cosmology. See `cosmolopy <https://roban.github.io/CosmoloPy/>`_ for more information.
    
    :returns: comoving distance in **Mpc**
    :rtype: float
    '''
    
    if cosmology is None:
        cosmology = COSMOLOGY
    
    cosmology     = cd.set_omega_k_0(cosmology)
    return cd.comoving_distance(z1, z0=z2, **cosmology)
 
def dz_from_dage(z: Union[int, float], dage: Union[int, float],
                 cosmology: Optional[Dict]=None) -> Tuple[float, float]:
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the redshift upper and lower bounds around a given redshift such that they are **dage** Gyr apart from the reference redshift.
    
    :param z: reference redshift
    :type z: int or float
    :param dage: age interval in Gyr
    :type dage: int or float
    
    :param dict cosmology: (**Optional**) parameters for the desired cosmology. See `cosmolopy <https://roban.github.io/CosmoloPy/>`_ for more information.
    
    :returns: upper and lower redshift bounds
    :rtype: float, float
    '''
    
    # Function to find the zero for the lower bound
    def func_min(x):
        
        # Make sure redshift do not fall outside interpolation range
        if x<0:
            x        = 0
        elif x>100:
            x        = 100
        
        age_x        = age_func(x)/cc.Gyr_s # in Gyr
        return age_z - age_x - dage
    
    # Function to find the zero for the upper bound
    def func_max(x):
        
        # Make sure redshift do not fall outside interpolation range
        if x<0:
            x        = 0
        elif x>100:
            x        = 100
            
        age_x        = age_func(x)/cc.Gyr_s # in Gyr
        return age_x - age_z - dage
    
    if cosmology is None:
        cosmology = COSMOLOGY
    
    cosmology        = cd.set_omega_k_0(cosmology)
    
    # age_func gives age as a function of z, z_func gives z as a function of age
    age_func, z_func = cd.quick_age_function(zmax=100, return_inverse=True, **cosmology)
    age_z            = age_func(z)/cc.Gyr_s               # in Gyr
    
    # Find lower bound
    guess            = z-1
    sol_low          = root(func_min, guess)
    
    # Find upper bound
    guess            = z+1
    sol_hig          = root(func_max, guess)
    
    return sol_hig['x'][0], sol_low['x'][0]
    

def separation(z: Union[int, float], ra1: float, dec1: float, ra2: float, dec2: float,
               units: Union[str, List[str]] = ['deg', 'deg', 'deg', 'deg']) -> float:
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the comoving separation between two objects at the same redshift given their position.

    :param z: redshift of the two objects
    :type z: int or float

    :param float ra1: right ascension of the first object
    :param float dec1: declination of the first object
    :param float ra2: right ascension of the second object
    :param float dec2: declination of the second object
    
    :param units: units of the different coordinates in this order: ra1, dec1, ra2, dec2
    :type units:  list[float] or list[astropy unit] or str

    :returns: comoving separation in **Mpc**
    :rtype: float
    
    :raises ValueError: if **units** is neither an iterable of length 4 or a str
    '''
    
    if isinstance(units, list):
        units = tuple(units)
    elif isinstance(units, str):
        units = [units]*4
        
    if len(units) != 4:
        raise ValueError('4 units should be given nor a single one only. Cheers !')
    
    # Objects representing the coordinates
    coord1 = SkyCoord(ra1, dec1, unit=units[:2])
    coord2 = SkyCoord(ra2, dec2, unit=units[-2:])
    
    sep   = coord1.separation(coord2)
    
    return comoving_separation(z, sep)
