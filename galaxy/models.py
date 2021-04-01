#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Useful functions for galaxy modelling and other related computation.
"""

import numpy                              as     np
import astropy.modeling.functional_models as     astmod
from   .misc                              import check_bns, compute_bn, PSFconvolution2D, checkAndComputeIe, intensity_at_re
from   astropy.constants                  import G

# If Planck18 not available we use Planck15
try:
    from astropy.cosmology                import Planck18 as cosmo
except ImportError:
    from astropy.cosmology                import Planck15 as cosmo



####################################################################################################################
#                                           1D profiles                                                            #
####################################################################################################################

def bulge(r, re, b4=None, Ie=None, mag=None, offset=None):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Computes the value of the intensity of a de Vaucouleur bulge at position r defined as
    
    .. math::
        
        \Sigma(r) = I_{\rm{e}} e^{\left [ \left (r/R_{\rm{e}} \right )^{1/4} - 1 \right ]},
    
    with :math:`R_{\rm{e}}` the effective radius and :math:`I_{\rm{e}}` the surface brightness at the effective radius.

    :param float r: position at which the profile is computed
    :param float re: half-light radius
                
    :param float b4: (**Optional**) b4 factor appearing in the Sersic profile. If None, its value will be computed.
    :param float Ie: (**Optional**) surface brightness at half-light radius
    :param float mag: (**Optional**) total integrated magnitude used to compute Ie if not given
    :param float offset: (**Optional**) magnitude offset in the magnitude system used
        
    :returns: surface brightness
    :rtype: float
    """
    
    b4, = check_bns([4], [b4])
    Ie  = checkAndComputeIe(Ie, 4, b4, re, mag, offset)
        
    return sersic_profile(r, 4, re, Ie=Ie, bn=b4)


def exponential_disk(r, re, b1=None, Ie=None, mag=None, offset=None):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Computes the value of the intensity of an exponential disk at position r defined as
    
    .. math::
        
        \Sigma(r) = I_{\rm{e}} e^{\left [ r/R_{\rm{e}} - 1 \right ]},
    
    with :math:`R_{\rm{e}}` the effective radius and :math:`I_{\rm{e}}` the surface brightness at the effective radius.
    
    :param float r: position at which the profile is computed
    :param float re: half-light radius
                
    :param float b1: (**Optional**) b1 factor appearing in the Sersic profile. If None, its value will be computed.
    :param float Ie: (**Optional**) surface brightness at half-light radius
    :param float mag: (**Optional**) total integrated magnitude used to compute Ie if not given
    :param float offset: (**Optional**) magnitude offset in the magnitude system used
        
    :returns: surface brightness
    :rtype: float
    """
    
    b1, = check_bns([1], [b1])
    Ie  = checkAndComputeIe(Ie, 1, b1, re, mag, offset)
        
    return sersic_profile(r, 1, re, Ie=Ie, bn=b1)


def hernquist(r, a, M):
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Hernquist profile defined as
    
    .. math::
        
        \rho(r) = \frac{M_{\rm{b}}}{2\pi} \frac{a}{r} (r + a)^{-3},
    
    with :math:`M_{\rm{b}}` the total mass and :math:`a` the scale radius.

    :param a: scale radius
    :type a: int or float
    :param M: total mass
    :type M: int or float
    :param r: radial distance(s) where to compute the Hernquist profile. Unit must be the same as **a**.
    :type r: int or float or ndarray[int] or ndarray[float]

    :returns: Hernquist profile evaluated at the given distance(s). Unit is that of **M**/**a^3**.
    :rtype: float or ndarray[float]
    
    :raises TypeError: if **r**, **M** and **a** are neither int, nor float
    :raises ValueError: if np.any(**r**) < 0, if **a** <= 0 or if **M** < 0
    '''
    
    # Checking dtypes and values
    if isinstance(r, (float, int)):
        if   r<0:
            raise ValueError('r must be positive only. Cheers !')
        elif r==0:
            return np.inf
        
    elif isinstance(r, np.ndarray):
        if np.any(r)<0:
            raise ValueError('r must be positive only. Cheers !')
        elif np.any(r==0):
            mask = r==0
        
    else:
        raise TypeError('r must either be int or float, or a numpy array of the same types. Cheers !')
        
    if not isinstance(M, (int, float)):
        raise TypeError('M must be int or float only. Cheers !')
    if not isinstance(a, (int, float)):
        raise TypeError('a must be int or float only. Cheers !')
    
    if a<= 0:
        raise ValueError('a must be positive only. Cheers !')
        
    if   M==0:
        return 0*r
    elif M<0:
        raise ValueError('M must be positive only. Cheers !')
    
    out        = r*0
    out[mask]  = np.inf
    out[~mask] = (M*a/2/np.pi) / (r*(r+a)**3)
    
    return out


def nfw(r, c, rs):
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    NFW profile defined as
    
    .. math::
        
        \rho(r) = \delta_{\rm{c}} \rho_{\rm{crit}} (r/r_{\rm{s}})^{-1} (1 + r/r_{\rm{s}})^{-2},
    
    with :math:`r_{\rm{s}} = r_{200} / c` the halo scale radius, with :math:`r_{200}` the virial radius where the mean overdensity is equal to 200, :math:`c` the halo concentration, :math:`\rho_{\rm{crit}} = 3 H_0^2 / (8\pi G)` the Universe closure density, and :math:`\delta_{\rm{c}}` the halo overdensity.

    :param c: halo concentration
    :type c: int or float
    :param r: radial distance(s) where to compute the profile. Unit must be the same as rs.
    :type r: int or float or ndarray[int] or ndarray[float]
    :param rs: scale radius
    :type rs: int or float
        
    :returns: NFW profile evaluated at the given distance. Unit is that of a 3D mass density in SI (i.e. kg/m^3).
    :rtype: float or ndarray[float]
    
    :raises TypeError: if **r**, **c** and **rs** are neither int, nor float
    :raises ValueError: if np.any(**r**)<0, if c<=0, or if rs<=0
    '''
    
    # Checking dtypes and values
    if isinstance(r, (float, int)):
        if   r<0:
            raise ValueError('r must be positive only. Cheers !')
        elif r==0:
            return np.inf
    elif isinstance(r, np.ndarray):
        if np.any(r)<0:
            raise ValueError('r must be positive only. Cheers !')
        elif np.any(r==0):
            mask = r==0
    else:
        raise TypeError('r must either be int or float, or a numpy array of the same types. Cheers !')
        
    if not isinstance(c, (int, float)):
        raise TypeError('c must be int or float only. Cheers !')
    if not isinstance(rs, (int, float)):
        raise TypeError('rs must be int or float only. Cheers !')
    
    if c <= 0:
        raise ValueError('c must be positive only. Cheers !')
        
    if rs<=0:
        raise ValueError('rs must be positive only. Cheers !')
    
    deltaC     = (200/3) * c**3 / (np.log(1+c) - c/(1+c))
    rhoCrit    = 3*cosmo.H(0)/(8*np.pi*G)
    out        = r*0
    out[mask]  = np.inf
    out[~mask] = deltaC*rhoCrit / ((r/rs) * (1 + r/rs)**2)
    
    return out


def sersic_profile(r, n, re, Ie=None, bn=None, mag=None, offset=None):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    General Sersic profile defined as 
    
    .. math::
        
        \Sigma(r) = I_{\rm{e}} e^{\left [ \left (r/R_{\rm{e}} \right )^{1/n} - 1 \right ]},
    
    with :math:`R_{\rm{e}}` the effective radius, :math:`I_{\rm{e}}` the surface brightness at the effective radius and :math:`n` the Sersic index.
    
    .. note::
        
        Compute it with:
            
            * n, re and Ie
            * n, re, mag and offset
            
    :param float r: position at which the profile is computed
    :param float re: half-light radius
                
    :param float bn: (**Optional**) bn factor appearing in the Sersic profile. If None, its value will be computed.
    :param float Ie: (**Optional**) surface brightness at half-light radius
    :param float mag: (**Optional**) total integrated magnitude used to compute Ie if not given
    :param float offset: (**Optional**) magnitude offset in the magnitude system used
        
    :returns: surface brightness
    :rtype: float
    """
    
    bn, = check_bns([n], [bn])
    Ie  = checkAndComputeIe(Ie, n, bn, re, mag, offset)
        
    return Ie*np.exp( -bn*((r/re)**(1.0/n) - 1) )


####################################################################################################
#                                      2D modelling                                                #
####################################################################################################

def _checkParams(nx, ny, samplingZone, fineSampling, verbose):
    ''''Check that given parameters are ok.'''
    
    if not isinstance(samplingZone, dict) or 'where' not in samplingZone:
        
        if verbose:
            print('sampling zone was not provided or syntax was incorrect. Thus, performing sampling (if relevant) on the full array.')
            
        samplingZone = {'where':'all'}
        
    if samplingZone['where'] not in ['all', 'centre']:
        raise ValueError("'where' keyword in samplingZone dictionnary should be either 'all' or 'centre'. Cheers !")
        
    if samplingZone['where']=='centre':
        if 'dx' not in samplingZone or 'dy' not in samplingZone:
            raise KeyError("'dx' and 'dy' keywords were missing in samplingZone dictionnary with 'where' keyword equal to 'centre'. Please provide values for the sampling box size around the centre. Cheers !")
        else:
            if not isinstance(samplingZone['dx'], (int, np.integer)) or not isinstance(samplingZone['dy'], (int, np.integer)):
                raise TypeError("At least one of the following keys in samplingZone dictionnary was not given as an integer: 'dx' or 'dy'. Please provide these as int. Cheers !")
    
    if not isinstance(fineSampling, (int, np.integer)) or not isinstance(nx, (int, np.integer)) or not isinstance(ny, (int, np.integer)):
        raise TypeError('One of the following parameter is not an integer, which is not valid: fineSampling (%s), nx (%s), ny (%s).' %(type(fineSampling), type(nx), type(ny)))

    if fineSampling < 1:
        raise ValueError('Fine sampling cannot be less than 1.')
        
    return samplingZone


def bulgeDiskOnSky(nx, ny, Rd, Rb, x0=None, y0=None, Id=None, Ib=None, magD=None, magB=None, offsetD=None, offsetB=None, inclination=0, PA=0, combine=True,
                   PSF={'name':'Gaussian2D', 'FWHMX':0.8, 'FWHMY':0.8, 'sigmaX':None, 'sigmaY':None, 'unit':'arcsec'}, noPSF=False, arcsecToGrid=0.03,
                   fineSampling=1, samplingZone={'where':'centre', 'dx':2, 'dy':2}, skipCheck=False, verbose=True):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Generate a bulge + (sky projected) disk 2D model (with PSF convolution).
    
    .. rubric:: **How to use**
        
    Apart from the mandatory inputs, it is necessary to provide

    * an intensity at Re for each profile
    * a total magnitude value for each profile and a corresponding magnitude offset per profile (to convert from magnitudes to intensities)
   
    .. rubric:: **Infos about sampling** 
   
    **fineSampling** parameter can be used to rebin the data. The shape of the final image will depend on the samplingZone used
    
    * if the sampling is performed everywhere ('where' keyword in **samplingZone** equal to 'all'), the final image will have dimensions (**nx*fineSampling**, **ny*fineSampling**)
    * if the sampling is performed around the centre ('where' equal to 'centre'), the central part is over-sampled, but needs to be binned in the end so that pixels have the same size in the central part and around. Thus, the final image will have the dimension (**nx**, **ny**).        
       
    .. warning::
        
        **Rd** and **Rb** should be given in pixel units. 
        
        If you provide them in arcsec, you must update the **arcsecToGrid** value to 1 (since 1 pixel will be equal to 1 arcsec). 
    
    :param int nx: size of the model for the x-axis
    :param int ny: size of the model for the y-axis
    :param float Rb: bulge half-light radius. Best practice is to provide it in pixels.
    :param float Rd: disk half-light radius. Best practice is to provide it in pixels.

    :param float arcsecToGrid: (**Optional**) pixel size conversion in arcsec/pixel, used to convert the FWHM/sigma from arcsec to pixel      
    :param float Ib: (**Optional**) bulge intensity at (bulge) half-light radius. If not provided, magnitude and magnitude offset must be given instead.
    :param float Id: (**Optional**) disk intensity at (disk) half-light radius. If not provided, magnitude and magnitude offset must be given instead.
    :param inclination: (**Optional**) disk inclination on sky. Generally given between -90° and +90°. Value must be given in degrees.
    :type inclination: (**Optional**) int or float
    :param float magB: (**Optional**) bulge total magnitude
    :param float magD: (**Optional**) disk total magnitude
    :param float offsetB: (**Optional**) bulge magnitude offset
    :param float offsetD: (**Optional**) disk magnitude offset
    :param bool noPSF: (**Optional**) whether to not perform PSF convolution or not
    :param PA: disk position angle (in degrees)
    :type PA: int or float
    :param int(>0) fineSampling: fine sampling for the pixel grid used to make high resolution models. For instance, a value of 2 means that a pixel will be split into two subpixels.
    :param dict PSF: (**Optional**) Dictionnary of the PSF (and its parameters) to use for the convolution. For now, only 2D Gaussians are accepted as PSF. 
    :param dict samplingZone: where to perform the over sampling. Dictionnaries should have the following keys:
    
        * 'where' (type str) -> either 'all' to perform everywhere or 'centre' to perform around the centre
        * 'dx'    (type int) -> x-axis maximum distance from the centre coordinate. A sub-array with x-axis values within [xpos-dx, xpos+dx] will be selected. If the sampling is performed everywhere, 'dx' does not need to be provided.
        * 'dy'    (type int) -> y-axis maximum distance from the centre coordinate. A sub-array with y-axis values within [ypos-dy, ypos+dy] will be selected. If the sampling is performed everywhere, 'dy' does not need to be provided.
               
    :param bool skipCheck: whether to skip the checking part or not
    :param x0: x-axis centre position. Default is None so that nx//2 will be used.
    :type x0: int or float
    :param y0: y-axis centre position. Default is None so that ny//2 will be used.
    :type y0: int or float
    :param bool verbose: whether to print text on stdout or not
    
    :returns: X, Y grids and the total (sky projected + PSF convolved) model of the bulge + disk decomposition
    :rtype: 2D ndarray, 2D ndarray, 2D ndarray
    '''
    
    ##############################################
    #          Checking input parameters         #
    ##############################################
    
    if not skipCheck:
        samplingZone = _checkParams(nx, ny, samplingZone, fineSampling, verbose)
        
        if any([i<0 for i in [nx, ny, Rb, Rd, arcsecToGrid]]):
            raise ValueError('At least one of the following parameters was provided as a negative number, which is not correct: nx, ny, Rb, Rd, arcsecToGrid.')
        
        if PA<-90 or PA>90:
            raise ValueError('PA should be given in the range -90° <= PA <= 90°, counting angles anti clock-wise (0° means major axis is vetically aligned). Cheers !')
       
    ##################################
    #         Compute models         #
    ##################################

    listn       = [1, 4]
    listbn      = [compute_bn(n) for n in listn]

    # Checking that we have the correct information to model correctly our data    
    if Id is None:
        if magD is not None and offsetD is not None:
            Id  = intensity_at_re(listn[0], magD, Rd, offsetD, bn=listbn[0])
        else:
            raise ValueError("Id is None, but magD or offsetD is also None. If no Id is given, please provide a value for the total magnitude and magnitude offset in order to compute the intensity. Cheers !")
    
    if Ib is None:
        if magB is not None and offsetB is not None:
            Ib  = intensity_at_re(listn[1], magB, Rb, offsetB, bn=listbn[1])
        else:
            raise ValueError("Ib is None, but magB or offsetB is also None. If no Ib is given, please provide a value for the total magnitude and magnitude offset in order to compute the intensity. Cheers !")

    X, Y, model = Sersic2D(nx, ny, listn, [Rd, Rb], x0=x0, y0=y0, listIe=[Id, Ib], listInclination=[inclination, 0], listPA=[PA, 0], fineSampling=fineSampling, samplingZone=samplingZone, combine=combine)
    
    if not noPSF:
        # If we perform fine sampling only in the central part, Sersic2D function rebins the data in the end, so the arcsec to pixel conversion factor does not need to be updated since we do not have a finer pixel scale
        if samplingZone['where'] == 'centre':
            fineSampling   = 1
        
        if combine:
            model          = PSFconvolution2D(model, model=PSF, arcsecToGrid=arcsecToGrid/fineSampling, verbose=verbose)
        else:
            for pos, mod in enumerate(model):
                model[pos] = PSFconvolution2D(mod, model=PSF, arcsecToGrid=arcsecToGrid/fineSampling, verbose=verbose)
                
    return X, Y, model


def bulge2D(nx, ny, Rb, x0=None, y0=None, Ib=None, mag=None, offset=None, inclination=0, PA=0, 
            PSF={'name':'Gaussian2D', 'FWHMX':0.8, 'FWHMY':0.8, 'sigmaX':None, 'sigmaY':None, 'unit':'arcsec'}, noPSF=False, arcsecToGrid=0.03,
            fineSampling=1, samplingZone={'where':'centre', 'dx':5, 'dy':5}, skipCheck=False, verbose=True):
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Generate a 2D model for a de Vaucouleur bulge. This model can be sky projected and PSF convolved.
    
    .. rubric:: **How to use**
    
    You must provide the size of the image with **nx** and **ny** as well as a bulge effective radius (usually in pixel unit) with **Rb**. Additionally, one must provide one of the following
    
    * surface brightness at **Rb** with **Ib**
    * total magnitude and magnitude offset to go from magnitude to **Ie** using the parameters **mag** and **offset**
    
    The PSF convolution only accepts 2D Gaussians for now. You can either provide
    
    * a FWHM in the X and Y directions
    * a dispersion in the X and Y directions
    
    You must also provide a unit for the FWHM or sigma values. This unit must be recognised by astropy. Since images have dimensions in pixel unit, if the FWHM or sigma values are not given in pixel unit, you must also provide :samp:`arcsecToGrid` to convert from physical unit to pixel unit.
    
    If you do not want to convolve with the PSF, provide :samp:`noPSF=True`.
    
    .. rubric: **Infos about sampling** 
  
    **fineSampling** parameter can be used to rebin the data. The shape of the final image will depend on the samplingZone used
        
    * if the sampling is performed everywhere ('where' keyword in **samplingZone** equal to 'all'), the final image will have dimensions (**nx*fineSampling**, **ny*fineSampling**)
    * if the sampling is performed around the centre ('where' equal to 'centre'), the central part is over-sampled, but needs to be binned in the end so that pixels have the same size in the central part and around. Thus, the final image will have the dimension (**nx**, **ny**).        

    :param int nx: size of the model for the x-axis
    :param int ny: size of the model for the y-axis
    :param float Rb: bulge half-light radius. Best practice is to provide it in pixels.
    :param x0: (**Optional**) x-axis centre position. Default is None so that **nx**//2 will be used.
    :type x0: int or float
    :param y0: (**Optional**) y-axis centre position. Default is None so that **ny**//2 will be used.
    :type y0: int or float
    :param float Ib: (**Optional**) bulge intensity at (bulge) half-light radius. If not provided, magnitude and magnitude offset must be given instead.
    :param float magB: (**Optional**) bulge total magnitude
    :param float offsetB: (**Optional**) bulge magnitude offset
    :param inclination: (**Optional**) inclination on sky. Generally given between -90° and +90°. Value must be given in degrees.
    :type inclination: (**Optional**) int or float
    :param PA: position angle (in degrees)
    :type PA: int or float
    :param dict PSF: (**Optional**) Dictionnary of the PSF (and its parameters) to use for the convolution. For now, only 2D Gaussians are accepted as PSF. 
    :param bool noPSF: (**Optional**) whether to not perform PSF convolution or not
    :param float arcsecToGrid: (**Optional**) pixel size conversion in arcsec/pixel, used to convert the FWHM/sigma from arcsec to pixel      
    :param int(>0) fineSampling: fine sampling for the pixel grid used to make high resolution models. For instance, a value of 2 means that a pixel will be split into two subpixels.
    :param dict samplingZone: where to perform the over sampling. Dictionnaries should have the following keys:
    
        * 'where' (type str) -> either 'all' to perform everywhere or 'centre' to perform around the centre
        * 'dx'    (type int) -> x-axis maximum distance from the centre coordinate. A sub-array with x-axis values within [xpos-dx, xpos+dx] will be selected. If the sampling is performed everywhere, 'dx' does not need to be provided.
        * 'dy'    (type int) -> y-axis maximum distance from the centre coordinate. A sub-array with y-axis values within [ypos-dy, ypos+dy] will be selected. If the sampling is performed everywhere, 'dy' does not need to be provided.
               
    :param bool skipCheck: whether to skip the checking part or not
    :param bool verbose: (**Optional**) whether to print info on stdout or not
    
    :returns: X coordinate array, Y coordinate array and the 2D bulge model
    :rtype: 2D ndarray[float], 2D ndarray[float], 2D ndarray[float]
    
    .. rubric:: **Example**
    
    .. plot::
        :include-source:
                    
        from   matplotlib.colors   import LogNorm
        from   matplotlib          import rc
        from   matplotlib.gridspec import GridSpec
        from   wilfried.galaxy     import models as mod
        import matplotlib.pyplot   as     plt
        import matplotlib          as     mpl
        
        # Bulge model without using fine sampling and without PSF
        X, Y, bulge1 = mod.bulge2D(100, 100, 35, mag=20, offset=30, noPSF=True)
        
        # Bulge mode with fine sampling but without PSF
        X, Y, bulge2 = mod.bulge2D(100, 100, 35, mag=20, offset=30, noPSF=True, fineSampling=81)
        
        # Bulge model with fine sampling and with PSF convolution (FWHM=0.8 arcsec)
        X, Y, bulge3 = mod.bulge2D(100, 100, 35, mag=20, offset=30, noPSF=False, fineSampling=81,
                                  PSF={'name':'Gaussian2D', 'FWHMX':0.8, 'FWHMY':0.8, 'unit':'arcsec'}, arcsecToGrid=0.03)
        
        ###############################
        #          Plot part          #
        ###############################
        
        # Setup figure and axes
        rc('font', **{'family': 'serif', 'serif': ['Times']})
        rc('text', usetex=True)
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'
        
        f            = plt.figure(figsize=(18, 7))
        gs           = GridSpec(1, 3, figure=f, wspace=0, hspace=0, left=0.01, right=0.99, top=0.99, bottom=0.1)
        
        ax1          = f.add_subplot(gs[0])
        ax2          = f.add_subplot(gs[1])
        ax3          = f.add_subplot(gs[2])
        
        ax1.set_title(r'No fine sampling',      size=20)
        ax2.set_title(r'Fine sampling = $9^2$', size=20)
        ax3.set_title(r'Fine sampling \& PSF',  size=20)
        
        for a in [ax1, ax2, ax3]:
           a.set_xticklabels([])
           a.set_yticklabels([])
           a.tick_params(axis='x', which='both', direction='in')
           a.tick_params(axis='y', which='both', direction='in')
           a.yaxis.set_ticks_position('both')
           a.xaxis.set_ticks_position('both')
        
        # Show bulges
        ret1  = ax1.imshow(bulge1, origin='lower', norm=LogNorm(), cmap='plasma')
        ret2  = ax2.imshow(bulge2, origin='lower', norm=LogNorm(), cmap='plasma')
        ret3  = ax3.imshow(bulge3, origin='lower', norm=LogNorm(), cmap='plasma')
        
        # Add colorbar
        cb_ax = f.add_axes([0.01, 0.08, 0.98, 0.025])
        cbar  = f.colorbar(ret1, cax=cb_ax, orientation='horizontal')
        cbar.set_label(r'Surface brightness [arbitrary unit]', size=20)
        cbar.ax.tick_params(labelsize=20)
        
        plt.show()
    '''
    
    if not skipCheck:
        samplingZone = _checkParams(nx, ny, samplingZone, fineSampling, verbose)
        
        if any([i<0 for i in [nx, ny, Rb, arcsecToGrid]]):
            raise ValueError('At least one of the following parameters was provided as a negative number, which is not correct: nx, ny, Rb, arcsecToGrid.')
        
        if PA<-90 or PA>90:
            raise ValueError('PA should be given in the range -90° <= PA <= 90°, counting angles anti clock-wise (0° means major axis is vetically aligned). Cheers !')
    
    # Generating bulge model
    X, Y, model = Sersic2D(nx, ny, [4], [Rb],
                           x0=x0, y0=y0, listIe=[Ib], listInclination=[inclination], listPA=[PA],
                           fineSampling=fineSampling, samplingZone=samplingZone,
                           verbose=verbose, skipCheck=True)
    
    # PSF convolution
    if not noPSF:
        
        # If we perform fine sampling only in the central part, Sersic2D function rebins the data at the end of the function,
        # So the arcsec to pixel conversion factor does not need to be updated since we do not have a finer pixel scale in our model
        if samplingZone['where'] == 'centre':
            fineSampling = 1
    
        model = PSFconvolution2D(model, model=PSF, arcsecToGrid=arcsecToGrid/fineSampling, verbose=verbose)

    return X, Y, model


def Sersic2D(nx, ny, listn, listRe, x0=None, y0=None, listIe=None, listMag=None, listOffset=None, listInclination=None, listPA=None, combine=True, 
            fineSampling=1, samplingZone={'where':'centre', 'dx':5, 'dy':5}, 
            skipCheck=False, verbose=True):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Generate a (sky projected) 2D model (image) of a sum of Sersic profiles. Neither PSF smoothing, nor projections onto the sky whatsoever are applied here.
    
    .. rubric:: **How to use**
    
    Apart from the mandatory inputs, it is necessary to provide

    * an intensity at Re for each profile
    * a total magnitude value for each profile and a corresponding magnitude offset per profile (to convert from magnitudes to intensities)
   
    .. rubric: **Infos about sampling** 
  
    **fineSampling** parameter can be used to rebin the data. The shape of the final image will depend on the samplingZone used
        
    * if the sampling is performed everywhere ('where' keyword in **samplingZone** equal to 'all'), the final image will have dimensions (**nx*fineSampling**, **ny*fineSampling**)
    * if the sampling is performed around the centre ('where' equal to 'centre'), the central part is over-sampled, but needs to be binned in the end so that pixels have the same size in the central part and around. Thus, the final image will have the dimension (**nx**, **ny**).        
       
    :param listn: list of Sersic index for each profile
    :type listn: list[int] or list[float]
    :param list[float] listRe: list of half-light radii for each profile
    :param int nx: size of the model for the x-axis
    :param int ny: size of the model for the y-axis
    
    :param bool combine: (**Optional**) whether to combine (sum) all the components and return a single intensity map, or to return each component separately in lists
    :param list[float] listIe: (**Optional**) list of intensities at re for each profile
    :param listInclination: (**Optional**) list of inclination of each Sersic component on the sky in degrees
    :type listInclination: list[int] or list[float]
    :type list[float] listMag: (**Optional**) list of total integrated magnitudes for each profile
    :type list[float] listOffset: (**Optional**) list of magnitude offsets used in the magnitude system for each profile
    :param listPA: (**Optional**) list of position angle of each Sersic component on the sky in degrees. Generally, these values are given between -90° and +90°.
    :type listPA: list[int] or list[float]
    :param int(>0) fineSampling: (**Optional**) fine sampling for the pixel grid used to make high resolution models. For instance, a value of 2 means that a pixel will be split into two subpixels.
    :param dict samplingZone: (**Optional**) where to perform the sampling. Default is everywhere. Dictionnaries should have the following keys:
       
        * 'where' (type str) -> either 'all' to perform everywhere or 'centre' to perform around the centre
        * 'dx'    (type int) -> x-axis maximum distance from the centre coordinate. An sub-array with x-axis values within [xpos-dx, xpos+dx] will be selected. If the sampling is performed everywhere, 'dx' does not need to be provided.
        * 'dy'    (type int) -> y-axis maximum distance from the centre coordinate. An sub-array with y-axis values within [ypos-dy, ypos+dy] will be selected. If the sampling is performed everywhere, 'dy' does not need to be provided.
          
    :param bool skipCheck: (**Optional**) whether to skip the checking part or not
    :param bool verbose: (**Optional**) whether to print info on stdout or not
    :param x0: (**Optional**) x-axis centre position. Default is None so that **nx**//2 will be used.
    :type x0: int or float
    :param y0: (**Optional**) y-axis centre position. Default is None so that **ny**//2 will be used.
    :type y0: int or float
        
    :returns: 
        
        * X, Y grids and the intensity map if **combine** is True
        * X, Y grids and a listof intensity maps for each component if **combine** is False
        
    :raises TypeError: 
        
        * if 'dx' and 'dy' keys are not in **samplingZone**
        * if **fineSampling**, **nx** and **ny** are neither int, nor np.integer
        
    :raises ValueError:
        
        * if 'where' key value in **samplingZone** is neither 'all', nor 'centre'
        * if nx, ny or arcsecToGrid are < 0
        * if at least one n or one Re is < 0
        * if **fineSampling** < 1
        * if at least one PA is not in the range [-90, 90] deg
        * if Ie and mag and offset are None
    """
    
    def computeSersic(X, Y, nbModels, listn, listRe, listIe, listInclination, listPA):
        
        # We need not specify a centre coordinate offset, because the X and Y grids are automatically centred on the real centre.
        # If we combine models, we add them, if we do not combine them, we place them into a list
        for pos, n, re, ie, inc, pa in zip(range(nbModels), listn, listRe, listIe, listInclination, listPA):
            
            # We add 90 to PA because we want a PA=0° galaxy to be aligned with the vertical axis
            ell                = 1-np.cos(inc*np.pi/180)
            pa                *= np.pi/180
            theModel           = astmod.Sersic2D(amplitude=ie, r_eff=re, n=n, x_0=0, y_0=0, ellip=ell, theta=np.pi/2+pa)
            
            if pos == 0:
                if combine:
                    intensity  = theModel(X, Y)/(1-ell)
                else:
                    intensity  = [theModel(X, Y)/(1-ell)]
            else:
                if combine:
                    intensity += theModel(X, Y)/(1-ell)
                else:
                    intensity += [theModel(X, Y)/(1-ell)]
        return intensity
    
    ##############################################
    #          Checking input parameters         #
    ############################################## 
    
    if not skipCheck:
        samplingZone = _checkParams(nx, ny, samplingZone, fineSampling, verbose)
        
        if any([i<0 for i in [nx, ny, arcsecToGrid]]):
            raise ValueError('At least one of the following parameters was provided as a negative number, which is not correct: nx, ny, arcsecToGrid.')
            
        for ll in [listn, listRe]:
            if any([i<0 for i in ll]):
                raise ValueError('At least one element in listn or listRe is a negative number, which is not correct.')
            
        if any([pa<-90 for pa in listPA]) or any([pa>90 for pa in listPA]):
            raise ValueError('PA should be given in the range -90° <= PA <= +90°, counting angles anti clock-wise. Cheers !')
        

    if listIe is None:
        if listMag is not None and listOffset is not None:
            listIe         = intensity_at_re(np.array(listn), np.array(listMag), np.array(listRe), np.array(listOffset))
        else:
            raise ValueError("listIe is None, but listMag or listOffset is also None. If no listIe is given, please provide a value for the total magnitude and magnitude offset in order to compute the intensities. Cheers !")

    nbModels               = len(listn)
    if listInclination is None:
        listInclination    = [0]*nbModels
    if listPA          is None:
        listPA             = [0]*nbModels
    
    ##################################
    #         Compute models         #
    ##################################

    # Define image centre
    midX                   = nx//2
    midY                   = ny//2
    
    if x0 is None:
        x0                 = midX
    if y0 is None:
        y0                 = midY
        
    # Pixel width is not 1 if we use fineSampling
    pixWidth               = 1.0/fineSampling
    pixHeight              = 1.0/fineSampling
    
    # We centre the coordinate X and Y grids to the given centre coordinates
    # The centre is recentred inside an 'original' pixel because of fine sampling (to not break any symmetry when rebinning)
    if samplingZone['where'] == 'all':
        newX0              = x0 + (1-pixWidth)/2
        newY0              = y0 + (1-pixHeight)/2
        listX              = np.arange(0, nx, pixWidth)  - newX0
        listY              = np.arange(0, ny, pixHeight) - newY0
        X, Y               = np.meshgrid(listX, listY)
        intensity          = computeSersic(X, Y, nbModels, listn, listRe, listIe, listInclination, listPA)/(fineSampling**2)
        
        # Rebinning intensity map in the central part
        '''
        intensity   = intensity.reshape(int(intensity.shape[0] / fineSampling), fineSampling, int(intensity.shape[1] / fineSampling), fineSampling)
        intensity   = intensity.mean(1).mean(2)
        
        listX              = np.arange(0, nx, 1) - x0
        listY              = np.arange(0, ny, 1) - y0
        X, Y               = np.meshgrid(listX, listY)
        '''
        
    else:
        # We generate grids with pixel size of 1x1 (and we centre it on the galaxy centre)
        listX              = np.arange(0, nx, 1) - x0
        listY              = np.arange(0, ny, 1) - y0
        X, Y               = np.meshgrid(listX, listY)
        intensity          = computeSersic(X, Y, nbModels, listn, listRe, listIe, listInclination, listPA)
        
        # We generate a subarray around the centre in the given box, using the given over-sampling factor
        maxX               = samplingZone['dx'] + 0.5 -1.0/(2*fineSampling) # Weird but that's what comes out of a few diagrams
        maxY               = samplingZone['dy'] + 0.5 -1.0/(2*fineSampling)
        listXcenPart       = np.arange(-maxX, maxX+pixWidth, pixWidth)
        listYcenPart       = np.arange(-maxY, maxY+pixHeight, pixHeight)
        
        XcenPart, YcenPart = np.meshgrid(listXcenPart, listYcenPart)
        intensityCenPart   = computeSersic(XcenPart, YcenPart, nbModels, listn, listRe, listIe, listInclination, listPA)

        # Rebinning intensity map in the central part
        intensityCenPart   = intensityCenPart.reshape(int(intensityCenPart.shape[0] / fineSampling), fineSampling, int(intensityCenPart.shape[1] / fineSampling), fineSampling)
        intensityCenPart   = intensityCenPart.sum(1).sum(2)/(fineSampling**2)
        
        # Combining back the central part into the original array
        intensity[y0-samplingZone['dy']:y0+samplingZone['dy']+1, x0-samplingZone['dx']:x0+samplingZone['dx']+1] = intensityCenPart
        
    return X, Y, intensity





