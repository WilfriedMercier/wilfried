#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:37:49 2020

@author: wilfried

Functions directly related to MUSE instrument and its observations.
"""

import astropy.units as u

def compute_lsfw(z, lambda0, a2=5.835e-8, a1=-9.080e-4, a0=5.983):
    '''
    Compute the MUSE Line Spread Fonction FWHM given the rest-frame wavelength of the line and the redshift of the corresponding object.

    Mandatory parameters
    --------------------
        lambda0 : float/int
            rest-frame wavelength in Angstroms
        z : float/int
            redshift of the object
        
    Optional parameters
    -------------------    
        a0: float
            lambda ** 0 coefficient of the variation of LSF FWHM with respect to lambda. Default is 5.835e-8 A.
        a1: float
            lambda ** 1 coefficient of the variation of LSF FWHM with respect to lambda. Default is -9.080e-4.
        a2: float
            lambda ** 2 coefficient of the variation of LSF FWHM with respect to lambda. Default is 5.983 A^{-1}.

    Return the LSF FWHM in Angstroms.
    '''
    
    if z<0:
        raise ValueError('Redshift must be positive valued.')
    
    # If the user uses astropy quantities, we convert to the right one    
    if isinstance(lambda0, u.Quantity):
        lambda0.to('Angstrom')
    
    lbda = lambda0 * (1 + z)
    return a2 * lbda ** 2 + a1 * lbda + a0