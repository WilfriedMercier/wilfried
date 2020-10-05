#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:43:49 2020

@author: Wilfried Mercier - IRAP

A set of functions to easily compute standard calculations in extragalactic physics using cosmology. This relies heavily on a custom, Python 3 adapted version of cosmolopy.
"""

import cosmolopy.distance as cd

COSMOLOGY = cd.set_omega_k_0({'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.72})

def angular_diameter_size(z, theta, scaleFactor=1.0, cosmology=None):
    '''
    Compute the size of an object with extent theta at redshift z from the angular diameter distance as soze=theta*angular diameter distance.

    Mandatory parameters
    --------------------
        z : float/int
            redshift of the object
        theta : float/int
            angle subtended by the object. Ultimately unit should be in arcsec. If not, provide a scaling factor with scaleFactor.
    
    Optional parameters
    -------------------
        cosmology : dict
            parameters of the desired cosmology. See cosmolopy help for more information. Default is {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.72}.
        scaleFactor : float/int
            if theta has a different unit, please provide a correct scale factor such that theta in radians = theta*scaleFactor

    Return the physical size in kpc.
    '''
    
    if cosmology is None:
        cosmology = COSMOLOGY
        
    return cd.angular_diameter_distance(z, **cosmology)*theta*scaleFactor*1000
    

def comoving_separation(z, theta, cosmology=None):
    '''
    Compute the tranverse distance (at fixed redshift) between objects separated by an angle theta on the sky in comoving units.

    Mandatory parameters
    --------------------
        z : float/int
            redshift of the two objects (must be identical)
        theta : float/int
            angular separation (in radians only) between the two objects
            
    Optional parameters
    -------------------
        cosmology : dict
            parameters of the desired cosmology. See cosmolopy help for more information. Default is {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.72}.

    Return the comoving separation in Mpc unit.
    '''
    
    if cosmology is None:
        cosmology = COSMOLOGY
    
    cosmology     = cd.set_omega_k_0(cosmology)
    return cd.comoving_distance_transverse(z, **cosmology)*theta

def comoving_los(z1, z2, cosmology=None):
    '''
    Compute the line of sight comoving distance between objects.

    Mandatory parameters
    --------------------
        z1 : float/int
            redshift of the first object
        z2 : float/int
            redshift of the second object
            
    Optional parameters
    -------------------
        cosmology : dict
            parameters of the desired cosmology. See cosmolopy help for more information. Default is {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.72}.

    Return the comoving distance in Mpc unit.
    '''
    
    if cosmology is None:
        cosmology = COSMOLOGY
    
    cosmology     = cd.set_omega_k_0(cosmology)
    return cd.comoving_distance(z1, z0=z2, **cosmology)