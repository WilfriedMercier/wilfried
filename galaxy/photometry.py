#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Epinat Benoit - LAM <benoit.epinat@lam.fr> & Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> & Hugo Plombat - LUPM <hugo.plombat@umontpellier.fr>

Photometry functions related to galaxies.
"""

import numpy             as np
import astropy.units     as u
from   astropy.cosmology import FlatLambdaCDM
from   numpy             import ndarray
from   typing            import Union, Tuple

#: Astropy cosmology used in the functions
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


###############################################
#       Flux and magnitudes conversions       #
###############################################

def countToMag(data: ndarray, err: ndarray, zeropoint: Union[float, ndarray]) -> Tuple[ndarray]:
    r'''
    .. codeauthor:: Hugo Plombat - LUPM <hugo.plombat@umontpellier.fr>
    
    Convert data counts and their associated error into AB magnitudes using the formula
    
    .. math::
        
        d [{\rm{mag}}] = -2.5 \log_{10} d [{\rm{e^{-}/s}}] + {\rm{zpt}}
        
    where :math:`d` is the data and :math:`\rm{zpt}` is the magnitude zeropoint. The error is given by
    
    .. math::
        
        \Delta d [{\rm{mag}}] = 1.08 \Delta d [{\rm{e^{-}/s}}] / d [{\rm{e^{-}/s}}]
    
    :param data: data in electron/s
    :type data: float or ndarray[float]
    :param err: std errors in electron/s
    :type err: float or ndarray[foat]
    :param float zeropoint: zeropoint associated to the data
    
    :returns: AB magnitude and associated error
    :rtype: float or ndarray[float], float or ndarray[float]
    '''

    mag  = -2.5 * np.log10(data) + zeropoint
    emag = 1.08 * err/data
    
    return mag, emag

def countToFlux(data: ndarray, err: ndarray, zeropoint: Union[float, ndarray]) -> Tuple[ndarray]:
    r'''
    .. codeauthor:: Hugo Plombat - LUPM <hugo.plombat@umontpellier.fr>
    
    Convert data counts and their associated error into flux in :math:`\rm{erg/cm^2/s/Hz}`.
    
    :param data: data in :math:`electron/s`
    :type data: float or ndarray[float]
    :param err: std errors in :math:`electron/s`
    :type err: float or ndarray[float]
    :param float zeropoint: zeropoint associated to the data
    
    :returns: AB magnitude and associated error
    :rtype: Astropy Quantity, Astropy Quantity
    '''

    flux  = u.Quantity(data * 10**(-(zeropoint+48.6)/2.5), unit='erg/(s*Hz*cm^2)')
    eflux = u.Quantity(err  * 10**(-(zeropoint+48.6)/2.5), unit='erg/(s*Hz*cm^2)')
    
    return flux, eflux


###############################################
#       Dust extinction and attenuation       #
###############################################

def av_Gilbank(lmstar: Union[float, ndarray], a: float=51.201, b: float=-11.199, c: float=0.615) -> Union[float, ndarray]:
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
    
    Compute and return the intrinsic extinction in the V band from Gilbank et al., 2010 prescription.
    
    :param lmstar: log10 of the stellar mass in solar masses
    :type lmstar: float or ndarray[float]
    
    :param float a: (**Optional**) first coefficient from Gilbank et al., 2010
    :param float b: (**Optional**) second coefficient from Gilbank et al., 2010
    :param float c: (**Optional**) third coefficient from Gilbank et al., 2010
    
    :returns: intrinsic extinction in the V band (Av) from Gilbank et al., 2010 prescription.
    :rtype: float or ndarray[float]
    '''
    
    def prescription(lmstar):
        return 51.201 - 11.199*lmstar + 0.615*lmstar**2
    
    # If an array
    if isinstance(lmstar, np.ndarray):
        
        # Gilbank et al., 2010 say to fit to a constant value below 10^9 solar masses
        aHalpha       = np.full(lmstar.shape, prescription(9))
        mask          = lmstar > 9
        aHalpha[mask] = prescription(lmstar[mask])
    
    elif isinstance(lmstar, (float, np.float16, np.float32, np.float64)):
        
        if lmstar <= 9:
            aHalpha   = prescription(9)
        else:
            aHalpha   = prescription(lmstar)
        
    return aHalpha * calzetti_law(0.550) / calzetti_law(0.656)

def calzetti_law(lbda: Union[float, ndarray], rv: float=4.05, 
                 a: float=2.659, b1: float=-1.857, c1: float=1.040, d1: float=0, e1: float=0, b2: float=-2.156, c2: float=1.509, d2: float=-0.198, e2: float=0.011) -> Union[float, ndarray]:
    r'''
    .. codeauthor:: Epinat Benoit - LAM <benoit.epinat@lam.fr>
    
    Attenuation curve from Calzetti et al. (2000). The coefficients can be modified in input. 
    
    .. note::
        
        In principle, this law is valid from 0.12 to 2.2 microns.
    
    The relation is 
    
    .. math::

        k = a \times (b + c/\lambda + d/\lambda^2 + e/\lambda^3),
        
    with :math:`\lambda` the wavelength.

    :param lbda: wavelength in microns
    :type lbda: float or ndarray[float]

    :param float rv: (**Optional**) ratio of the total to selective attenuation
    :param float a: (**Optional**) multiplicative coefficient
    :param float bi: (**Optional**) coefficients for the relation above 0.63 microns
    :param float bi: (**Optional**) coefficients for the relation below 0.63 microns
        
    :returns: Calzetti attenuation
    :rtype: float or ndarray[float]
    '''
    
    lbda     = np.asarray(lbda)
    
    # Law is different below and beyond this threshold
    cond1    = (lbda > 0.63)
    cond2    = (lbda <= 0.63)
    
    k        = np.zeros(np.shape(lbda))
    k[cond1] = a * (b1 + c1 / lbda[cond1] + d1 / lbda[cond1] ** 2 + e1 / lbda[cond1] ** 3) + rv
    k[cond2] = a * (b2 + c2 / lbda[cond2] + d2 / lbda[cond2] ** 2 + e2 / lbda[cond2] ** 3) + rv
    return k

def dust_attenuation_calzetti(lbda: Union[float, ndarray], ebv: float, rv: float=4.05) -> Union[float, ndarray]:
    '''
    .. codeauthor:: Epinat Benoit - LAM <benoit.epinat@lam.fr>
    
    Compute the dust attenuation using the Calzetti attenuation curve (could be more general adding a function name) from the colour excess.

    :param lbda: wavelength in microns
    :type lbda: float or ndarray[float]
    :param float ebv: colour excess

    :param float rv: (**Optional**) ratio of the total to selective attenuation

    :returns: dust attenuation in mag
    :rtype: float or ndarray[float]
    '''
    
    return ebv * calzetti_law(lbda, rv=rv)

def cardelli_law(lbda: Union[float, ndarray], rv: float=3.1) -> Union[float, ndarray]:
    '''
    .. codeauthor:: Epinat Benoit - LAM <benoit.epinat@lam.fr>
    
    Attenuation curve from Cardelli et al. (1989). The coefficients can be modified in input. 
    
    .. note::
        
        In principle, this law is valid from 0.1 to 3.3 microns. For MW, Rv~3.1.
    
    :param lbda: wavelength in microns
    :type lbda: float or ndarray[float]
    
    :param float rv: (**Optional**) ratio of the total to selective attenuation
            
    :returns: cardelli attenuation
    :rtype: float or ndarray[float]
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

def dust_attenuation_cardelli(lbda: Union[float, ndarray], ebv: float, rv: float=3.1) -> Union[float, ndarray]:
    '''
    .. codeauthor:: Epinat Benoit - LAM <benoit.epinat@lam.fr>
    
    Compute the dust attenuation using the Cardelli attenuation curve from the colour excess.

    :param lbda: wavelength in microns
    :type lbda: float or ndarray[float]
    :param float ebv: colour excess

    :param float rv: (**Optional**) ratio of the total to selective attenuation
            
    :returns: dust attenuation in mag
    :rtype: float or ndarray[float]
    '''
    
    return ebv * cardelli_law(lbda, rv=rv) 


#############################################
#      Flux and luminosity corrections      #
#############################################

def correct_extinction(floii: Union[float, ndarray], z: Union[float, ndarray], av: Union[float, ndarray], ebv: Union[float, ndarray], 
                       lbda: Union[float, ndarray]=0.3728) -> Union[float, ndarray]:
    '''
    .. codeauthor:: Epinat Benoit - LAM <benoit.epinat@lam.fr>
    
    Correct for both galactic extinction and internal extinction. We use the following attenuation curves:
        
        * Cardelli for the Galactic extinction
        * Calzetti for the intrinsic extinction

    :param av: attenuation in V band
    :type av: float or ndarray[float]
    :param ebv: colour excess
    :type ebv: float or ndarray[float]
    :param floii: ionised gas flux
    :type floii: float or ndarray[float]
    :param z: redshift
    :type z: float or ndarray[float]

    :param lbda: (**Optional**) rest-frame wavelength in microns
    :type lbda: float or ndarray[float]
    
    :returns: flux corrected of extinction
    :rtype: float or ndarray[float]
    '''
    
    if hasattr(lbda, 'unit'):
        lbda.to('micron').value
    
    lbda_obs       = lbda * (1 + z)
    
    # Galactic extinction at the observed wavelength
    abs_gal        = dust_attenuation_cardelli(lbda_obs, ebv, rv=3.1)
    
    # Convert the extinction from FAST to the observed wavelength
    EBV_fast       = av / calzetti_law(np.array(.550))
    abs_fast       = dust_attenuation_calzetti(lbda, EBV_fast, rv=4.05)
    
    # Apply the flux correction from the Milky Way and the intrinsic extinction
    flux_corr_gal  = floii * 10 ** (0.4 * abs_gal)
    flux_corr_fast = flux_corr_gal * 10 ** (0.4 * abs_fast)
    
    return flux_corr_fast

def flux_to_lum(floii: Union[float, ndarray], z: Union[float, ndarray], 
                av: Union[float, ndarray]=None, ebv: Union[float, ndarray]=None, lbda: Union[float, ndarray]=0.3728) -> Union[float, ndarray]:
    r'''
    .. codeauthor:: Epinat Benoit - LAM <benoit.epinat@lam.fr>
    
    Convert a flux into a luminosity. If an attenuation and a colour excess are given, the flux is corrected beforehand.

    :param floii: flux in :math:`\rm{erg/s/cm^2}`. Optionally, can directly be given as an Astropy Quantity with a unit of the kind :math:`\rm{erg/s/cm^2}`.
    :type floii: float or ndarray[float]
    :param z: redshift
    :type z: float or ndarray[float]
    
    :param av: (**Optional**) attenuation in V band
    :type av: float or ndarray[float]
    :param ebv: (**Optional**) colour excess
    :type ebv: float or ndarray[float]
    :param lbda: (**Optional**) rest-frame wavelength in microns
    :type lbda: float or ndarray[float]
    
    :returns: luminosity in :math:`\rm{erg/s}`
    :rtype: float or ndarray[float]
    '''
    
    # Make astropy Quantity
    if not hasattr(floii, 'unit'):
        floii              = u.Quantity(floii, unit='erg/(s.cm^2)')
        
    # Correct flux of attenuation
    if av is not None and ebv is not None and lbda is not None:
        
       # If an array
       if isinstance(floii.value, np.ndarray):
               
           # Where there is nan, we do not apply the correction
           mask            = np.isnan(av) | np.isnan(ebv) | np.isnan(lbda)
           
           # Redshift mask
           if isinstance(z, np.ndarray):
               zmask       = z[~mask]
           else:
               zmask       = z
             
           # Av mask
           if isinstance(av, np.ndarray):
               av_mask     = av[~mask]
           else:
               av_mask     = av
    
           # Colour excess mask
           if isinstance(ebv, np.ndarray):
               ebvmask     = ebv[~mask]
           else:
               ebvmask     = ebv
               
           # Rest-frame wavelength mask
           if isinstance(lbda, np.ndarray):
               lbdamask    = lbda[~mask]
           else:
               lbdamask    = lbda
               
           # Extract unit and value to apply the correction
           unit            = floii.unit
           floii           = floii.value
           floii[~mask]    = correct_extinction(floii[~mask], zmask, av_mask, ebvmask, lbda=lbdamask)
           
           # Reconstruct the Quantity object
           floii           = u.Quantity(floii, unit=unit)
           
       # If a float and we can apply a correction
       elif not np.isnan(av) and not np.isnan(ebv) and not np.isnan(lbda):
           floii           = correct_extinction(floii, z, av, ebv, lbda=lbda)
    
    dl                     = cosmo.luminosity_distance(z)
    loii                   = (floii * 4 * np.pi * dl**2).to('erg/s').value
    
    return loii


####################################################
#             Infer gas masses and SFR             #
#################################################### 

def mass_gas_sfr(sfr: Union[float, ndarray], rd: Union[float, ndarray]) -> Union[float, ndarray]:
    r'''
    .. codeauthor:: Epinat Benoit - LAM <benoit.epinat@lam.fr>
    
    Convert the SFR into a gas mass assuming it is uniformly distributed in :math:`2.2 \times R_{\rm{d}}`, with :math:`R_{\rm{d}}` the disk scale length.

    :param sfr: SFR in :math:`M_{\odot}/{\rm{yr}}`. Optionally, if given as an astropy Quantity, it will be converted to :math:`M_{\odot}/{\rm{yr}}`.
    :type sfr: float or ndarray[float]
    :param rd: disk scale length in :math:`\rm{pc}`. Optionally, if given as an astropy Quantity, it will be converted to :math:`\rm{pc}`.
    :type rd: float or ndarray[float]

    :returns: gas mass in :math:`M_{\odot}`
    :rtype: float or ndarray[float]
    '''
    
    # Astropy Quantities
    if hasattr(rd, 'unit'):
        rd.to('pc').value
    
    if hasattr(sfr, 'unit'):
        sfr.to('solMass/yr').value
    
    return (np.pi * (2.2*rd)**2) ** (2/7) * (sfr / 2.5e-10) ** (5/7)


def mass_gas_kennicutt(floii: Union[float, ndarray], rd: Union[float, ndarray], z: Union[float, ndarray], 
                       av: Union[float, ndarray]=None, ebv: Union[float, ndarray]=None, lbda: Union[float, ndarray]=0.3728) -> Union[float, ndarray]:
    r'''
    .. codeauthor:: Epinat Benoit - LAM <benoit.epinat@lam.fr>
    
    Convert the ionised gas flux into a gas mass assuming it is uniformly distributed in 2.2*rd. If an attenuation and a colour excess are given, the flux is corrected beforehand.
    
    :param floii: ionised gas flux in :math:`\rm{erg/s/cm2}`
    :type floii: float or ndarray[float]
    :param rd: disk scale length in :math:`\rm{pc}`. Optionally, if given as an astropy Quantity, it will be converted to :math:`\rm{pc}`.
    :type rd: float or ndarray[float]
    :param z: redshift
    :type z: float or ndarray[float]
    
    :param av: (**Optional**) attenuation in V band
    :type av: float or ndarray[float]
    :param ebv: (**Optional**) colour excess
    :type ebv: float or ndarray[float]
    :param lbda: (**Optional**) rest-frame wavelength in microns
    :type lbda: float or ndarray[float]
    
    :returns: gas mass using the Kennicutt law and its error in :math:`M_{\odot}`
    :rtype: float or ndarray[float], float or ndarray[float]
    '''
    
    if hasattr(rd, 'unit'):
        rd.to('pc').value
    
    loii  = flux_to_lum(floii, z, av=av, ebv=ebv, lbda=lbda)
    mgas  = (np.pi * (2.2*rd)**2) ** (2/7) * 4.7564e-23 * loii ** (5/7)
    dmgas = (np.pi * (2.2*rd)**2) ** (2/7) * 1.9438e-23 * loii ** (5/7)
    
    return mgas, dmgas


def sfr_kennicutt_oii(floii: Union[float, ndarray], z: Union[float, ndarray], 
                      av: Union[float, ndarray]=None, ebv: Union[float, ndarray]=None, lbda: Union[float, ndarray]=0.3728) -> Union[float, ndarray]:
    r'''
    .. codeauthor:: Epinat Benoit - LAM <benoit.epinat@lam.fr>
    
    Compute the SFR from the ionised gas flux. If an attenuation and a colour excess are given, the flux is corrected beforehand.
    
    :param floii: ionised gas flux in :math:`\rm{erg/s/cm^2}`. Optionally, can directly be given as an Astropy Quantity with a unit of the kind :math:`\rm{erg/s/cm^2}`.
    :type floii: float or ndarray[float]
    :param z: redshift
    :type z: float or ndarray[float]
    
    :param av: (**Optional**) attenuation in V band
    :type av: float or ndarray[float]
    :param ebv: (**Optional**) colour excess
    :type ebv: float or ndarray[float]
    :param lbda: (**Optional**) rest-frame wavelength in microns
    :type lbda: float or ndarray[float]
    
    :returns: SFR in :math:`M_{\odot}/{\rm{yr}}` and its error.
    '''
    
    loii = flux_to_lum(floii, z, av=av, ebv=ebv, lbda=lbda)
    sfr  = 1.4e-41 * loii
    dsfr = 0.4e-41 * loii
    
    return sfr, dsfr


def sfr_Gilbank_oii(floii, z, lmstar):
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
    '''
    
    tanh = -1.424 * np.tanh((lmstar - 9.827)/0.572)
    
    # Do not correct [OII] luminosity for extinction with Gilbank
    loii = flux_to_lum(floii, z, av=None, ebv=None, lbda=None)
    
    return loii / 3.8e40 / (tanh + 1.7)