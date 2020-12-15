#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:50:50 2020

@author: Benoit Epinat (LAM) & Wilfried Mercier (IRAP)

Photometry functions related to galaxies.
"""

import numpy             as np
from   astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def calzetti_law(lbda, rv=4.05, a=2.659, b1=-1.857, c1=1.040, d1=0, e1=0, b2=-2.156, c2=1.509, d2=-0.198, e2=0.011):
    '''
    Attenuation curve from Calzetti et al. (2000). The coefficients can be modified in input. 
    In principle, this law is valid from 0.12 to 2.2 microns.
    
    The relation is k=a*(b+c/lbda+d/lbda**2+e/lbda**3)
    
    Author : Benoit Epinat - LAM
    
    Parameters
    ----------
        lbda: float/numpy array
            wavelength in microns
    
    Optional parameters
    -------------------
        rv: float
            ratio of the total to selective attenuation
        a: float
            multiplicative coefficient
        b1, c1, d1: float
            coefficients for the relation above 0.63 microns
        b2, c2, d2: float
            coefficients for the relation below 0.63 microns
            
    Return the Calzetti attenuation.
    '''
    
    lbda     = np.asarray(lbda)
    
    # Law is different below and beyond this threshold
    cond1    = (lbda > 0.63)
    cond2    = (lbda <= 0.63)
    
    k        = np.zeros(np.shape(lbda))
    k[cond1] = a * (b1 + c1 / lbda[cond1] + d1 / lbda[cond1] ** 2 + e1 / lbda[cond1] ** 3) + rv
    k[cond2] = a * (b2 + c2 / lbda[cond2] + d2 / lbda[cond2] ** 2 + e2 / lbda[cond2] ** 3) + rv
    return k

def dust_attenuation_calzetti(lbda, ebv, rv=4.05):
    '''
    Compute the dust attenuation using the Calzetti attenuation curve (could be more general adding a function name) from the colour excess.
    
    Author : Benoit Epinat - LAM
    
    Parameters
    ----------
        lbda: float/numpy array
            wavelength where the attenuation has to be estimated
        ebv: float
            colour excess
    
    Optional parameters
    -------------------
        rv: float
            ratio of the total to selective attenuation
    
    Return the dust attenuation in mag.
    '''
    
    return ebv * calzetti_law(lbda, rv=rv)

def cardelli_law(lbda, rv=3.1):
    '''
    Attenuation curve from Cardelli et al. (1989). The coefficients can be modified in input. 
    In principle, this law is valid from 0.1 to 3.3 microns. For MW, Rv~3.1.
    
    Author : Benoit Epinat - LAM
    
    Parameters
    ----------
        lbda: float/numpy array
            wavelength in microns
            
    Optional parameters
    -------------------
        rv: float
            ratio of the total to selective attenuation
            
    Return the cardelli attenuation.
    '''
    
    # Need to be sure we work with numpy arrays to have the right behaviour for the bineary operators
    lbda        = np.asarray(lbda)
    
    condfuv     = (lbda <= 1/8.)  & (lbda >= 1/10.)
    condnuv     = (lbda <= 1/3.3) & (lbda >  1/8.)
    condnuv1    = (lbda <= 1/5.9) & (lbda >  1/8.)
    condnuv2    = (lbda <= 1/3.3) & (lbda >  1/5.9)
    condopt     = (lbda <= 1/1.1) & (lbda >  1/3.3)
    condnir     = (lbda <= 1/0.3) & (lbda >  1/1.1)
    
    a           = np.zeros(np.shape(lbda))
    b           = np.zeros(np.shape(lbda))
    y           = np.zeros(np.shape(lbda))
    
    y[condfuv]  = 1 / lbda[condfuv]  - 8
    y[condnuv1] = 1 / lbda[condnuv1] - 5.9
    y[condnuv2] = 0
    y[condopt]  = 1 / lbda[condopt]  - 1.82
    y[condnir]  = 1 / lbda[condnir]
    
    a[condfuv]  = -1.073 - 0.628 * y[condfuv] + 0.137 * y[condfuv] ** 2 - 0.070 * y[condfuv] ** 3
    a[condnuv]  = 1.752 - 0.316 * 1 / lbda[condnuv] - 0.104 / ((1 / lbda[condnuv] - 4.67) ** 2 + 0.341) - 0.04473 * y[condnuv] ** 2 - 0.009779 * y[condnuv] ** 3
    a[condopt]  = 1 + 0.17699 * y[condopt] - 0.50447 * y[condopt] ** 2 - 0.02427 * y[condopt] ** 3 + 0.72085 * y[condopt] ** 4 + 0.01979 * y[condopt] ** 5 - 0.77530 * y[condopt] ** 6 + 0.32999 * y[condopt] ** 7
    a[condnir]  = 0.574 * y[condnir] ** 1.61
    
    b[condfuv]  = 13.670 + 4.257 * y[condfuv] - 0.420 * y[condfuv] ** 2 + 0.374 * y[condfuv] ** 3
    b[condnuv]  = -3.090 + 1.825 * 1 / lbda[condnuv] + 1.206 / ((1 / lbda[condnuv] - 4.62) ** 2 + 0.263) + 0.2130 * y[condnuv] ** 2 + 0.1207 * y[condnuv] ** 3
    b[condopt]  = 1.41338 * y[condopt] + 2.28305 * y[condopt] ** 2 + 1.07233 * y[condopt] ** 3 - 5.38434 * y[condopt] ** 4 - 0.62251 * y[condopt] ** 5 + 5.30260 * y[condopt] ** 6 - 2.09002 * y[condopt] ** 7
    b[condnir]  = -0.527 * y[condnir] ** 1.61
    
    al_av       = a + b / rv
    k           = al_av * rv
    
    return k

def dust_attenuation_cardelli(lbda, ebv, rv=3.1):
    '''
    Compute the dust attenuation using the Cardelli attenuation curve from the colour excess.
    
    Author : Benoit Epinat - LAM
    
    Parameters
    ----------
        lbda: float/numpy array
            wavelength where the attenuation has to be estimated
        ebv: float/numpy array
            colour excess
            
    Optional parameters
    -------------------
        rv: float
            ratio of the total to selective attenuation
            
    Return the dust attenuation in mag.
    '''
    
    return ebv * cardelli_law(lbda, rv=rv) 

def flux_to_lum(floii, z):
    '''
    Convert a flux into a luminosity.
    
    Author : Benoit Epinat - LAM
    
    Parameters
    ----------
        floii: float/numpy array
            flux in erg/s/cm^2. Optionally, can e directly given as an Astropy Quantity with a unit of the kind erg/s/cm^2.
        z : float/numpy array
            redshift
    
    Returns luminosity in erg/s.
    '''
    
    dl   = cosmo.luminosity_distance(z)
    loii = (floii * 4 * np.pi * dl**2).to('erg/s').value
    
    return loii


def correct_extinction(floii, z, av_fast, ebv, lbda=0.3728):
    '''
    Correct for both galactic extinction and internal extinction.
    
    Author : Benoit Epinat - LAM
    
    Parameters
    ----------
        av_fast : float/numpy array
            attenuation from FAST
        ebv : float/numpy array
            colour excess
        floii : float/numpy array
            ionised gas flux
        z : float/numpy array
            redshift
    
    Optional parameters
    -------------------
        lbda : float/numpy array
            rest-frame wavelength in microns
    '''
    
    if hasattr(lbda, 'unit'):
        lbda.to('micron').value
    
    lbda_obs       = lbda * (1 + z)
    
    # Galactic extinction at the observed wavelength
    abs_gal        = dust_attenuation_cardelli(lbda_obs, ebv, rv=3.1)
    
    # Convert the extinction from FAST at the observed wavelength
    ext_fast       = av_fast / calzetti_law(np.array(.550))
    abs_fast       = dust_attenuation_calzetti(lbda, ext_fast, rv=4.05)
    
    # Apply the flux correction from the Milky Way and the galaxy extinction
    flux_corr_gal  = floii * 10 ** (0.4 * abs_gal)
    flux_corr_fast = flux_corr_gal * 10 ** (0.4 * abs_fast)
    
    return flux_corr_fast


####################################################
#             Infer gas masses and SFR             #
#################################################### 

def mass_gas_sfr(sfr, rd):
    '''
    Convert the SFR into a gas mass assuming it is uniformly distributed in 2.2*rd.
    
    Author : Benoit Epinat - LAM
    
    Parameters
    ----------
        sfr : float/numpy array  
            SFR in Msun/yr. Optionally, if given as an astropy Quantity, it will be converted to Msun/yr.
        rd : float/numpy array
            disk scale length in pc. Optionally, if given as an astropy Quantity, it will be converted to pc.
    
    Return the mass gas.
    '''
    
    # Astropy Quantities
    if hasattr(rd, 'unit'):
        rd.to('pc')
    
    if hasattr(sfr, 'unit'):
        sfr.to('solMass/yr')
    
    return (np.pi * (2.2*rd)**2) ** (2/7) * (sfr / 2.5e-10) ** (5/7)


def mass_gas_kennicutt(floii, rd, z):
    '''
    Convert the ionised gas flux into a gas mass assuming it is uniformly distributed in 2.2*rd.
    
    Author : Benoit Epinat - LAM
    
    Parameters
    ----------
        floii : float/numpy array
            ionised gas flux in erg/s/cm2
        rd : float/numpy array
            disk scale length in pc. Optionally, if given as an astropy Quantity, it will be converted to pc.
    
    Return the mass gas using the Kennicutt law and its error.
    '''
    
    if hasattr(rd, 'unit'):
        rd.to('pc')
    
    loii  = flux_to_lum(floii, z)
    mgas  = ((np.pi * (2.2*rd)**2) ** (2/7) * 4.7564e-23 * loii ** (5/7)).value
    dmgas = ((np.pi * (2.2*rd)**2) ** (2/7) * 1.9438e-23 * loii ** (5/7)).value
    
    return mgas, dmgas


def sfr_kennicutt_oii(floii, z):
    '''
    Compute the SFR from the ionised gas flux.
    
    Author : Benoit Epinat - LAM
    
    Parameters
    ----------
        floii : float/numpy array
            ionised gas flux in erg/s/cm2
        z : float/numpy array
            redshift
    
    Return the SFR in Msun/yr and its error.
    '''
    
    loii = flux_to_lum(floii, z)
    sfr  = 1.4e-41 * loii
    dsfr = 0.4e-41 * loii
    
    return sfr, dsfr