#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Hugo Plombat - LUPM & Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Utilties related to generating 2D mass and SFR maps using LePhare SED fitting codes.
"""

import subprocess
import os
import os.path       as     opath
from   textwrap      import dedent
from   astropy.table import Table
from   .misc         import Property, PathlikeProperty
from   .catalogues   import LePhareCat

from   ..symlinks.coloredMessages import errorMessage

ERROR   = errorMessage('Error: ')

class SED:
    r'''General SED object used for inheritance.'''
    
    def __init__(self, *args, **kwargs):
        r'''Init SED oject.'''
        
        # Code associated to the SED
        self.code       = None
        
        # Allowed keys and corresponding types for the SED parameters
        self.keys       = {}
        
        # SED parameter properties
        self.prop = {}
        
    
    #########################################
    #     Methods need be reimplemented     #
    #########################################
    
    def genParams(self, *args, **kwargs):
        r'''Generate a parameter file used by the SED fitting code.'''
        
        raise NotImplementedError('genParams method not implemented.')
    
    def __run__(self, *args, **kwargs):
        r'''Run the SED fitting code.'''
        
        raise NotImplementedError('run method not implemented.')
        
    ###########################
    #     Private methods     #
    ###########################
        
    def __getitem__(self, key):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Allow to get a SED parameter value using the syntax
        
        >>> sed = SED()
        >>> print(sed['STAR_LIB'])
        
        :param str key: SED parameter name
        :returns: SED parameter value if it is a valid key, or None otherwise
        '''
        
        if key in self.keys:
            return self.prop[key]
        else:
            print(WARNING + f'{key} is not a valid key.')
        
        return
        
    def __setitem__(self, key, value):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Allow to set a SED parameter using the syntax
        
        >>> sed = SED()
        >>> sed['STAR_LIB'] = 'LIB_STAR_bc03'
        
        :param str key: SED parameter name
        :param value: value corresponding to this parameter
        '''
            
        if key not in self.keys:
            print(ERROR + f'key {key} is not a valid key. Accepted keys are {list(self.keys.values())}.')
        elif not isinstance(value, self.keys[key]):
            print(ERROR + f'trying to set {key} parameter with value {value} of type {type(value)} but only allowed type(s) is/are {self.keys[key]}.')
        else:
            self.prop[key].set(value)
        
        return

class LePhareSED(SED):
    r'''Implements LePhare SED object.'''
    
    def __init__(self, ID, properties={}, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Init LePhare object.
        
        :param ID: an identifier used to name the output files created during the SED fitting process
        
        :param dict properties: (**Optional**) properties to be passed to LePhare to compute the models grid and perform the SED fitting
        
        :raises TypeError: if one of the properties items is not of type str
        :raises ValueError: if one of the properties does not have a valid name (see list below)
        
        Accepted properties are:
        
            * **STAR_SED** [str]: stellar library list file (full path)
            * **STAR_FSCALE** [int/float]: stellar flux scale
            * **STAR_LIB** [str]: stellar library to use (default libraries found at $LEPHAREWORK/lib_bin). To not use a stellar library, provide 'NONE'.
            * **QSO_SED** [str]: QSO list file (full path)
            * **QSO_FSCALE** [int, float]: QSO flux scale
            * **QSO_LIB** [str]: QSO library to use (default libraries found at $LEPHAREWORK/lib_bin). To not use a QSO library, provide 'NONE'.
            * **GAL_SED** [str]: galaxy library list file (full path)
            * **GAL_FSCALE** [int, float]: galaxy flux scale
            * **GAL_LIB** [str]: galaxy library to use (default libraries found at $LEPHAREWORK/lib_bin). To not use a galaxy library, provide 'NONE'.
            * **SEL_AGE** [str]: stellar ages list (full path)
            * **AGE_RANGE** [list[int/float]]: minimum and maximum ages in years
            * **FILTER_LIST** [list[str]]: list of filter names used for the fit (must all be located in $LEPHAREDIR/filt directory)
            * **TRANS_TYPE** [int]: transmission type (0 for Energy, 1 for photons)
            * **FILTER_CALIB** [int]: filter calibration (0 for fnu=ctt, 1 for nu.fnu=ctt, 2 for fnu=nu, 3=fnu=Black Body @ T=10000K, 4 for MIPS (leff with nu fnu=ctt and flux with BB @ 10000K) 
            * **FILTER_FILE** [str]: filter file (must be located in $LEPHAREWORK/filt directory)
            * **STAR_LIB_IN** [str]: input stellar library (dupplicate with **STAR_LIB** ?)
            * **STAR_LIB_OUT** [str]: output stellar magnitudes
            * **QSO_LIB_IN** [str]: input QSO library (dupplicate with **QSO_LIB** ?)
            * **QSO_LIB_OUT** [str]: output QSO magnitudes
            * **GAL_LIB_IN** [str]: input galaxy library (dupplicate with **GAL_LIB** ?)
            * **GAL_LIB_OUT** [str]: output galaxy magnitudes
            * **MAGTYPE** [str]: magnitude system used (AB or VEGA)
            * **Z_STEP** [list[int/float]]: redshift step properties. Values are: redshift step, max redshift, redshift step for redshifts above 6 (coarser sampling).
            * **COSMOLOGY** [list[int/float]]: cosmology parameters. Values are: Hubble constant H0, baryon fraction Omegam0, cosmological constant fraction Omegalambda0.
            * **MOD_EXTINC** [list[int/float]]: minimum and maximum model extinctions
            * **EXTINC_LAW** [str]: extinction law file (in $LEPHAREDIR/ext)
            * **EB_V** [list[int/float]]: color excess E(B-V). It must contain less than 50 values.
            * **EM_LINES** [str]: whether to consider emission lines or not. Accepted values are 'YES' or 'NO'.
            * **BD_SCALE** [int]: number of bands used for scaling (0 means all bands). See LePhare documentation for more details.
            * **GLB_CONTEXT** [int]: context number (0 means all bands). See LePhare documentation for more details.
            * **ERR_SCALE** [list[int/float]]: magnitude errors per band to add in quadrature
            * **ERR_FACTOR** [int/float]: scaling factor to apply to the errors
            * **ZPHOTLIB** [list[str]]: librairies used to compute the Chi2. Maximum number is 3.
            * **ADD_EMLINES** [str]: whether to add emission lines or not (dupplicate with **EM_LINES** ?). Accepted values are 'YES' or 'NO'.
            * **FIR_LIB** [str]: far IR library
            * **FIR_LMIN** [int/float]: minimum wavelength (in microns) for the far IR analysis
            * **FIR_CONT** [int/float]: far IR continuum. Use -1 for no continuum.
            * **FIR_SCALE** [int/float]: far IR flux scale. Use -1 to skip flux scale.
            * **FIR_FREESCALE** [str]: whether to let the far IR spectrum freely scale
            * **FIR_SUBSTELLAR** [str]: ???
            * **PHYS_LIB** [str]: physical stochastic library
            * **PHYS_CONT** [int/float]: physical continuum. Use -1 for no continuum.
            * **PHYS_SCALE** [int/float]: physical flux scale. Use -1 to skip flux scale.
            * **PHYS_NMAX** [int]: ???
            * **MAG_ABS** [list[int/float]]: minimum and maximum values for the magnitudes.
            * **MAG_REF** [int]: reference band used by **MAG_ABS**
            * **Z_RANGE** [list[int/float]]: minimum and maximum redshifts used by the galaxy library
            * **EBV_RANGE** [list[int/float]]: minimum and maximum colour excess E(B-V)
            * **ZFIX** [str]: whether to fix the redshift or let it free. Accepted values are 'YES' or 'NO'.
            * **Z_INTERP** [str]: whether to perform an interpolation to find the redshift. Accepted values are 'YES' or 'NO'.
            * **DZ_WIN** [int/float]: window search for second peak. Must be between 0 and 5.
            * **MIN_THRES** [int/float]: minimum threshold for second peak. Must be between 0 and 1.
            * **MABS_METHOD** [int]: method used to compute magnitudes (0 : obs->Ref, 1 : best obs->Ref, 2 : fixed obs->Ref, 3 : mag from best SED, 4 : Zbin). See LePhare documentation for more details.
            * **MABS_CONTEXT** [int]: context for absolute magnitudes. See LePhare documentation for more details.
            * **MABS_REF** [int]: reference band used to compute the absolute magnitudes. This is only used if **MABS_METHOD** = 2.
            * **MABS_FILT** [list[int]]: filters used in each redshift bin (see **MABS_ZBIN**). This is only used if **MABS_METHOD** = 4.
            * **MABS_ZBIN** [list[int/float]]: redshift bins (must be an even number). This is only used if **MABS_METHOD** = 4.
            * **SPEC_OUT** [str]: whether to output the spectrum of each object or not. Accepted values are 'YES' or 'NO'.
            * **CHI2_OUT** [str]: whether to generate an output file with all the values or not. Accepted values are 'YES' or 'NO'.
            * **PDZ_OUT** [str]: output file name for the PDZ analysis. To not do the pdz analysis, provide 'NONE'.
            * **PDZ_MABS_FILT** [list[int]]: absolute magnitude for reference filters to be extracted. See LePhare documentation for more details.
            * **FAST_MODE** [str]: whether to perform a fast computation or not. Accepted values are 'YES' or 'NO'.
            * **COL_NUM** [int]: number of colors used
            * **COL_SIGMA** [int/float]: quantity by which to enlarge the errors on the colors
            * **COL_SEL** [str]: operation used to combine colors. Accepted values are 'AND' or 'OR'.
            * **AUTO_ADAPT** [str]: whether to use an adaptive method with a z-spec sample. Accepted values are 'YES' or 'NO'.
            * **ADAPT_BAND** [list[int]]: reference band, band1 and band2 for colors
            * **ADAPT_LIM** [list[int/float]]: magnitude limit for spectro in reference band
            * **ADAPT_POLY** [int]: number of coefficients in polynomial. Maximum is 4.
            * **ADAPT_METH** [int]: fit method, 1 for color model, 2 for redshift, 3 for models. See LePhare documentation for more details.
            * **ADAPT_CONTEXT** [int]: context for the bands used for training. See LePhare documentation for more details.
            * **ADAPT_ZBIN** [list[int/float]]: minimum and maximum redshift interval used for training.
            
        .. warning::
            
            It is mandatory to define on your OS two environment variables:
                
                * $LEPHAREWORK which points to LePhare working directory
                * $LEPHAREDIR which points to LePhare main directory
                
            These paths may be expanded to check whether the given files exist and can be used by the user to shorten some path names when providing the SED properties.
        '''
        
        super().__init__(**kwargs)
        
        # Will be used to generate a custom directory
        self.id         = ID
        
        # Allowed keys and corresponding allowed types
        self.prop = {'STAR_SED'       : PathlikeProperty(opath.join('$LEPHAREDIR', 'sed', 'STAR', 'STAR_MOD.list')),
                     
                     'STAR_FSCALE'    : Property(3.432e-09, (int, float), minBound=0),
                     
                     'STAR_LIB'       : PathlikeProperty('LIB_STAR_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin'), ext='.bin'),
                     
                     'QSO_SED'        : PathlikeProperty(opath.join('$LEPHAREDIR', 'sed', 'QSO', 'QSO_MOD.list')),
                     
                     'QSO_FSCALE'     : Property(1, (int, float), minBound=0),
                     
                     'QSO_LIB'        : PathlikeProperty('LIB_QSO_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin'),  ext='.bin'),
                     
                     'GAL_SED'        : PathlikeProperty(opath.join('$LEPHAREDIR', 'sed', 'GAL', 'BC03_CHAB', 'BC03_MOD.list')),
                     
                     'GAL_FSCALE'     : Property(1, (int, float), minBound=0),
                     
                     'GAL_LIB'        : PathlikeProperty('LIB_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin'),      ext='.bin'),
                     
                     'SEL_AGE'        : PathlikeProperty(opath.join('$LEPHAREDIR', 'sed', 'GAL', 'BC03_CHAB', 'BC03_AGE.list')),
                     
                     'AGE_RANGE'      : Property([0, 14e9], list, subtypes=(int, float), minBound=0, 
                                                 testFunc=lambda value: len(value)!=2 or value[1] < value[0], 
                                                 testMsg='AGE_RANGE property must be an increasing length 2 list.'),
                     
                     'FILTER_LIST'    : PathlikeProperty(['hst/acs_f435w.pb', 'hst/acs_f606w.pb', 'hst/acs_f775w.pb', 'hst/acs_f850lp.pb'], 
                                                         path=opath.join('$LEPHAREDIR', 'filt')),
                     
                     'TRANS_TYPE'     : Property(0, int, minBound=0, maxBound=1),
                     
                     'FILTER_CALIB'   : Property(0, int, minBound=0, maxBound=4),
                     
                     'FILTER_FILE'    : PathlikeProperty('HDF_bc03.filt', 
                                                         path=opath.join('$LEPHAREWORK', 'filt')),
                     
                     'STAR_LIB_IN'    : PathlikeProperty('LIB_STAR_bc03', 
                                                         path=opath.join('$LEPHAREWORK', 'lib_bin'), ext='.bin'),
                     
                     'STAR_LIB_OUT'   : Property('STAR_HDF_bc03', str),                                             # We do not use PathlikeProperty because this is an output file
                     
                     'QSO_LIB_IN'     : PathlikeProperty('LIB_QSO_bc03', 
                                                         path=opath.join('$LEPHAREWORK', 'lib_bin'), ext='.bin'),
                     
                     'QSO_LIB_OUT'    : Property('QSO_HDF_bc03', str),                                              # We do not use PathlikeProperty because this is an output file
                     
                     'GAL_LIB_IN'     : PathlikeProperty('LIB_bc03', 
                                                         path=opath.join('$LEPHAREWORK', 'lib_bin'), ext='.bin'),
                     
                     'GAL_LIB_OUT'    : Property('HDF_bc03', str),                                                  # We do not use PathlikeProperty because this is an output file
                     
                     'MAGTYPE'        : Property('AB', str,
                                                 testFunc=lambda value: value not in ['AB', 'VEGA'],
                                                 testMsg='MAGTYPE must either be AB or VEGA.'),
                     
                     'Z_STEP'         : Property([0.01, 2, 0.1], list, subtypes=(int, float), minBound=0,
                                                 testFunc=lambda value: len(value)!=3 or value[2] < value[0], 
                                                 testMsg='Z_STEP property must be a length 3 list where the last step must be larger than the first one.'),
                     
                     'COSMOLOGY'      : Property([70, 0.3, 0.7], list, subtypes=(int, float), minBound=0),
                     
                     'MOD_EXTINC'     : Property([0, 27], list, subtypes=(int, float),
                                                 testFunc=lambda value: len(value)!=2 or value[1] < value[0], 
                                                 testMsg='MOD_EXTINC property must be an increasing length 2 list.'),
                     
                     'EXTINC_LAW'     : PathlikeProperty('calzetti.dat', path=opath.join('$LEPHAREDIR', 'ext')),
                     
                     'EB_V'           : Property([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], list, subtypes=(int, float),
                                                 testFunc=lambda value: len(value)>49 or any((j<=i for i, j in zip(value[:-1], value[1:]))), 
                                                 testMsg='EB_V property must be an increasing list with a maximum length of 49.'),
                     
                     'EM_LINES'       : Property('NO', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='EM_LINES must either be YES or NO.'),
                     
                     'LIB_ASCII'      : Property('NO', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='LIB_ASCII must either be YES or NO.'),
                     
                     'BD_SCALE'       : Property(0, int),
                     
                     'GLB_CONTEXT'    : Property(0, int),
                     
                     'ERR_SCALE'      : Property([0.03, 0.03, 0.03, 0.03], list, subtypes=(int, float), minBound=0),
                     
                     'ERR_FACTOR'     : Property(1, (int, float)),
                     
                     'ZPHOTLIB'       : Property(['HDF_bc03', 'STAR_HDF_bc03', 'QSO_HDF_bc03'], list, subtypes=str),
                     
                     'ADD_EMLINES'    : Property('NO', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='ADD_EMLINES must either be YES or NO.'),
                     
                     'FIR_LIB'        : PathlikeProperty('NONE'),
                     
                     'FIR_LMIN'       : Property(7, (int, float), minBound=0),
                     
                     'FIR_CONT'       : Property(-1, (int, float)),
                     
                     'FIR_SCALE'      : Property(-1, (int, float)),
                     
                     'FIR_FREESCALE'  : Property('NO', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='FIR_FREESCALE must either be YES or NO.'),
                     
                     'FIR_SUBSTELLAR' : Property('NO', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='FIR_SUBSTELLAR must either be YES or NO.'),
                     
                     'PHYS_LIB'       : PathlikeProperty('NONE'),
                     
                     'PHYS_CONT'      : Property(-1, (int, float)),
                     
                     'PHYS_SCALE'     : Property(-1, (int, float)),
                     
                     'PHYS_NMAX'      : Property(100000, int),
                     
                     'MAG_ABS'        : Property([-20, -30], list, subtypes=(int, float),
                                                 testFunc=lambda value: len(value)!=2,
                                                 testMsg='MAG_ABS property must be a length 2 list.'),
                     
                     'MAG_REF'        : Property(1, int, minBound=0),
                     
                     'Z_RANGE'        : Property([0, 2], list, subtypes=(int, float), minBound=0,
                                                 testFunc=lambda value: len(value)!=2 or value[1] < value[0],
                                                 testMsg='Z_RANGE property must be an increasing length 2 list.'),
                     
                     'EBV_RANGE'      : Property([0, 9], list, subtypes=(int, float),
                                                 testFunc=lambda value: len(value)!=2 or value[1] < value[0],
                                                 testMsg='Z_RANGE property must be an increasing length 2 list.'),
                     
                     'ZFIX'           : Property('YES', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='ZFIX must either be YES or NO.'),
                     
                     'Z_INTERP'       : Property('NO', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='Z_INTERP must either be YES or NO.'),
                     
                     'DZ_WIN'         : Property(0.5, (int, float), minBound=0, maxBound=5),
                     
                     'MIN_THRES'      : Property(0.1, (int, float), minBound=0, maxBound=1),
                     
                     'MABS_METHOD'    : Property( 1, int, minBound=0, maxBound=4),
                     
                     'MABS_CONTEXT'   : Property(-1, int),
                     
                     'MABS_REF'       : Property( 0, int),
                     
                     'MABS_FILT'      : Property([1, 2, 3, 4], list, subtypes=int, minBound=0),
                     
                     'MABS_ZBIN'      : Property([0, 0.5, 1, 1.5, 2, 3, 3.5, 4], list, subtypes=(int, float), minBound=0,
                                                 testFunc=lambda value: len(value)%2!=0 or any([j<=i for i, j in zip(value[:-1], value[1:])]), 
                                                 testMsg='MABS_ZBIN property must be an increasing list with an even length.'),
                     
                     'SPEC_OUT'       : Property('NO', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='SPEC_OUT must either be YES or NO.'),
                     
                     'CHI2_OUT'       : Property('NO', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='CHI2_OUT must either be YES or NO.'),
                     
                     'PDZ_OUT'        : PathlikeProperty('NONE'),
                     
                     'PDZ_MABS_FILT'  : Property([2, 10, 14], list, subtypes=int, minBound=0),
                     
                     'FAST_MODE'      : Property('NO', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='FAST_MODE must either be YES or NO.'),
                     
                     'COL_NUM'        : Property(3, int, minBound=0),
                     
                     'COL_SIGMA'      : Property(3, (int, float), minBound=0),
                     
                     'COL_SEL'        : Property('AND', str,
                                                 testFunc=lambda value: value not in ['AND', 'OR'],
                                                 testMsg='COL_SEL must either be YES or NO.'),
                     
                     'AUTO_ADAPT'     : Property('NO', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='AUTO_ADAPT must either be YES or NO.'),
                     
                     'ADAPT_BAND'     : Property([4, 2, 4], list, subtypes=int, minBound=0),
                     
                     'ADAPT_LIM'      : Property([20, 40], list, subtypes=(int, float),
                                                 testFunc=lambda value: len(value)!=2 or value[1] < value[0],
                                                 testMsg='ADAPT_LIM property must be an increasing length 2 list.'),
                     
                     'ADAPT_POLY'     : Property( 1, int, minBound=1, maxBound=4),
                     
                     'ADAPT_METH'     : Property( 1, int, minBound=1, maxBound=3),
                     
                     'ADAPT_CONTEXT'  : Property(-1, int),
                     
                     'ADAPT_ZBIN'     : Property([0.01, 6], list, subtypes=(int, float), minBound=0,
                                                 testFunc=lambda value: len(value)!=2 or value[1] < value[0],
                                                 testMsg='ADAPT_ZBIN property must be an increasing length 2 list.'),
                     
                     'ADAPT_MODBIN'   : Property([1, 1000], list, subtypes=int, minBound=1, maxBound=1000,
                                                 testFunc=lambda value: len(value)!=2 or value[1] < value[0],
                                                 testMsg='ADAPT_MODBIN property must be an increasing length 2 list.'),
                     
                     'ERROR_ADAPT'    : Property('NO', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='ERROR_ADAPT must either be YES or NO.')
                     }
        
        # Set properties given by the user
        for item, value in properties.items():
            
            if not isinstance(item, str):
                raise TypeError(f'item in properties has type {type(item)} but it must have type str')
            
            item = item.upper()
            if item not in self.prop:
                raise ValueError(f'item {item} in properties does not correspond to a valid item name.')
                
            # Set Property value
            self.prop[item].set(value)
        
    
    @property
    def parameters(self, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Generate a parameter file used by the SED fitting code.
        '''
        
        ######################################
        #         Libraries handling         #
        ######################################
        
        # %INPUTCATALOGUEINFORMATION% is replaced when the run method is launched
        
        
        text = f'''\
        ##############################################################################
        #                CREATION OF LIBRARIES FROM SEDs List                        #
        # $LEPHAREDIR/source/sedtolib -t (S/Q/G) -c $LEPHAREDIR/config/zphot.para    #
        # help : $LEPHAREDIR/source/sedtolib -h (or -help)                           #
        ##############################################################################
        #
        #------      STELLAR LIBRARY (ASCII SEDs)
        STAR_SED \t{self.prop['STAR_SED']} \t\t# STAR list (full path)
        STAR_FSCALE \t{self.prop['STAR_FSCALE']} \t\t\t\t\t# Arbitrary Flux Scale
        STAR_LIB \t{self.prop['STAR_LIB']} \t\t\t\t\t# Bin. STAR LIBRARY -> $LEPHAREWORK/lib_bin
        #
        #------      QSO LIBRARY (ASCII SEDs)
        QSO_SED \t{self.prop['QSO_SED']} \t\t# QSO list (full path)
        QSO_FSCALE \t{self.prop['QSO_FSCALE']} \t\t\t\t\t\t# Arbitrary Flux Scale 
        QSO_LIB	\t{self.prop['QSO_LIB']} \t\t\t\t\t# Bin. QSO LIBRARY -> $LEPHAREWORK/lib_bin
        #
        #------      GALAXY LIBRARY (ASCII or BINARY SEDs)
        GAL_SED	\t{self.prop['GAL_SED']} \t# GAL list (full path)
        GAL_FSCALE \t{self.prop['GAL_FSCALE']} \t\t\t\t\t\t# Arbitrary Flux Scale
        GAL_LIB	\t{self.prop['GAL_LIB']} \t\t\t\t\t# Bin. GAL LIBRARY -> $LEPHAREWORK/lib_bin
        SEL_AGE \t{self.prop['SEL_AGE']} \t# Age list(full path, def=NONE)	
        AGE_RANGE \t{self.prop['AGE_RANGE']} \t\t\t\t\t# Age Min-Max in yr
        #
        #############################################################################
        #                           FILTERS                                         #
        #  $LEPHAREDIR/source/filter  -c $LEPHAREDIR/config/zphot.para              #
        #  help: $LEPHAREDIR/source/filter  -h (or -help)                           #
        #############################################################################
        #  Filter number and context 
        #   f300 f450 f606 f814 J  H  K 
        #   1    2    3    4    5  6  7
        #   1    2    4    8   16  32 64 = 127 
        #
        FILTER_LIST \t{self.prop['FILTER_LIST']} \t# (in $LEPHAREDIR/filt/*)
        TRANS_TYPE \t{self.prop['TRANS_TYPE']} \t\t# TRANSMISSION TYPE
        FILTER_CALIB \t{self.prop['FILTER_CALIB']} \t\t# 0[-def]: fnu=ctt, 1: nu.fnu=ctt, 2: fnu=nu, 3: fnu=Black Body @ T=10000K, 4: for MIPS (leff with nu fnu=ctt and flux with BB @ 10000K
        FILTER_FILE \t{self.prop['FILTER_FILE']} \t# output name of filter's file -> $LEPHAREWORK/filt/
        #
        ############################################################################
        #                 THEORETICAL  MAGNITUDES                                  #
        # $LEPHAREDIR/source/mag_star -c  $LEPHAREDIR/config/zphot.para (star only)#
        # help: $LEPHAREDIR/source/mag_star -h (or -help)                          #
        # $LEPHAREDIR/source/mag_gal  -t (Q or G) -c $LEPHAREDIR/config/zphot.para #
        #                                                         (for gal. & QSO) #
        # help: $LEPHAREDIR/source/mag_gal  -h (or -help)                          #
        ############################################################################
        #
        #-------     From STELLAR LIBRARY
        STAR_LIB_IN \t{self.prop['STAR_LIB_IN']} \t# Input STELLAR LIBRARY in $LEPHAREWORK/lib_bin/
        STAR_LIB_OUT \t{self.prop['STAR_LIB_OUT']} \t# Output STELLAR MAGN -> $LEPHAREWORK/lib_mag/
        #
        #-------     From QSO     LIBRARY   
        QSO_LIB_IN \t{self.prop['QSO_LIB_IN']} \t# Input QSO LIBRARY in $LEPHAREWORK/lib_bin/
        QSO_LIB_OUT\t{self.prop['QSO_LIB_OUT']} \t# Output QSO MAGN -> $LEPHAREWORK/lib_mag/
        #
        #-------     From GALAXY  LIBRARY  
        GAL_LIB_IN \t{self.prop['GAL_LIB_IN']} \t# Input GAL LIBRARY in $LEPHAREWORK/lib_bin/
        GAL_LIB_OUT\t{self.prop['GAL_LIB_OUT']} \t# Output GAL LIBRARY -> $LEPHAREWORK/lib_mag/ 
        #
        #-------   MAG + Z_STEP + EXTINCTION + COSMOLOGY
        MAGTYPE \t{self.prop['MAGTYPE']} \t\t# Magnitude type (AB or VEGA)
        Z_STEP \t\t{self.prop['Z_STEP']} \t# dz, zmax, dzsup(if zmax>6)
        COSMOLOGY \t{self.prop['COSMOLOGY']} \t# H0,om0,lbd0 (if lb0>0->om0+lbd0=1)
        MOD_EXTINC \t{self.prop['MOD_EXTINC']} \t\t# model range for extinction 
        EXTINC_LAW \t{self.prop['EXTINC_LAW']} \t# ext. law (in $LEPHAREDIR/ext/*)
        EB_V \t\t{self.prop['EB_V']} \t# E(B-V) (<50 values)
        EM_LINES \t{self.prop['EM_LINES']}
        # Z_FORM 	8,7,6,5,4,3 	# Zformation for each SED in GAL_LIB_IN
        #
        #-------   ASCII OUTPUT FILES OPTION
        LIB_ASCII \t{self.prop['LIB_ASCII']} \t\t# Writes output in ASCII in working directory
        #
        ############################################################################
        #              PHOTOMETRIC REDSHIFTS                                       #
        # $LEPHAREDIR/source/zphot -c $LEPHAREDIR/config/zphot.para                #
        # help: $LEPHAREDIR/source/zphot -h (or -help)                             #
        ############################################################################ 
        #  
        #
        #
        %INPUTCATALOGUEINFORMATION%
        CAT_OUT \t{self.id}.out \t
        PARA_OUT \t{self.id}.para \t\t# Ouput parameter (full path)
        BD_SCALE \t{self.prop['BD_SCALE']} \t\t# Bands used for scaling (Sum 2^n; n=0->nbd-1, 0[-def]:all bands)
        GLB_CONTEXT \t{self.prop['GLB_CONTEXT']} \t\t# Overwrite Context (Sum 2^n; n=0->nbd-1, 0 : all bands used, -1[-def]: used context per object)
        # FORB_CONTEXT -1               # context for forbitten bands
        ERR_SCALE \t{self.prop['ERR_SCALE']} \t# errors per band added in quadrature
        ERR_FACTOR \t{self.prop['ERR_FACTOR']} \t\t# error scaling factor 1.0 [-def]       
        #
        #-------    Theoretical libraries
        ZPHOTLIB \t{self.prop['ZPHOTLIB']} \t# Library used for Chi2 (max:3)
        ADD_EMLINES \t{self.prop['ADD_EMLINES']}
        #
        ########    PHOTOMETRIC REDSHIFTS OPTIONS      ###########
        # FIR LIBRARY
        FIR_LIB \t{self.prop['FIR_LIB']}
        FIR_LMIN \t{self.prop['FIR_LMIN']} \t# Lambda Min (micron) for FIR analysis 
        FIR_CONT \t{self.prop['FIR_CONT']}
        FIR_SCALE \t{self.prop['FIR_SCALE']}
        FIR_FREESCALE \t{self.prop['FIR_FREESCALE']} \t# ALLOW FOR FREE SCALING 
        FIR_SUBSTELLAR \t{self.prop['FIR_SUBSTELLAR']}
        # PHYSICAL LIBRARY with Stochastic models from  BC07        
        PHYS_LIB \t{self.prop['PHYS_LIB']}  
        PHYS_CONT \t{self.prop['PHYS_CONT']}
        PHYS_SCALE \t{self.prop['PHYS_SCALE']}
        PHYS_NMAX \t{self.prop['PHYS_NMAX']}
        #
        #-------     Priors  
        # MASS_SCALE	0.,0.		# Lg(Scaling) min,max [0,0-def]
        MAG_ABS \t{self.prop['MAG_ABS']} \t# Mabs_min, Mabs_max [0,0-def]
        MAG_REF \t{self.prop['MAG_REF']} \t\t# Reference number for band used by Mag_abs
        # ZFORM_MIN	5,5,5,5,5,5,3,1	# Min. Zformation per SED -> Age constraint
        Z_RANGE \t{self.prop['Z_RANGE']} \t\t# Z min-max used for the Galaxy library 
        EBV_RANGE \t{self.prop['EBV_RANGE']} \t\t# E(B-V) MIN-MAX RANGE of E(B-V) used  
        # NZ_PRIOR      4,2,4           # I Band for prior on N(z)
        #                          
        #-------     Fixed Z   (need format LONG for input Cat)
        ZFIX \t{self.prop['ZFIX']} \t# fixed z and search best model [YES,NO-def]
        #
        #-------     Parabolic interpolation for Zbest  
        Z_INTERP \t{self.prop['Z_INTERP']} \t# redshift interpolation [YES,NO-def]
        #-------  Analysis of normalized ML(exp-(0.5*Chi^2)) curve 
        #-------  Secondary peak analysis       
        DZ_WIN \t\t{self.prop['DZ_WIN']} \t# Window search for 2nd peaks [0->5;0.25-def]
        MIN_THRES \t{self.prop['MIN_THRES']} \t# Lower threshold for 2nd peaks[0->1; 0.1-def]
        #
        #-------  Probability (in %) per redshift intervals
        # PROB_INTZ     0,0.5,0.5,1.,1.,1.5     # even number
        #
        #########    ABSOLUTE MAGNITUDES COMPUTATION   ###########
        #
        MABS_METHOD\t{self.prop['MABS_METHOD']} \t\t\t\t# 0[-def] : obs->Ref, 1 : best  obs->Ref, 2 : fixed obs->Ref, 3 : mag from best SED, 4 : Zbin
        MABS_CONTEXT \t{self.prop['MABS_CONTEXT']} \t\t\t\t# CONTEXT for Band used for MABS 

        MABS_REF \t{self.prop['MABS_REF']} \t\t\t\t# 0[-def]: filter obs chosen for Mabs (ONLY USED IF MABS_METHOD=2)
        MABS_FILT \t{self.prop['MABS_FILT']} \t\t\t# Chosen filters per redshift bin (MABS_ZBIN - ONLY USED IF MABS_METHOD=4)
        MABS_ZBIN \t{self.prop['MABS_ZBIN']} \t# Redshift bins (even number - ONLY USED IF MABS_METHOD=4)
        #########   OUTPUT SPECTRA                     ###########
        #
        SPEC_OUT \t{self.prop['SPEC_OUT']} \t\t# spectrum for each object? [YES,NO-def]
        CHI2_OUT \t{self.prop['CHI2_OUT']} \t\t# output file with all values : z,mod,chi2,E(B-V),... BE CAREFUL can take a lot of space !!
        #########  OUTPUT PDZ ANALYSIS  
        PDZ_OUT \t{self.prop['PDZ_OUT']} \t\t# pdz output file name [def-NONE] - add automatically PDZ_OUT[.pdz/.mabsx/.mod/.zph] 
        PDZ_MABS_FILT \t{self.prop['PDZ_MABS_FILT']} \t# MABS for REF FILTERS to be extracted
        # 
        #########   FAST MODE : color-space reduction        #####
        #
        FAST_MODE \t{self.prop['FAST_MODE']} \t# Fast computation [NO-def] 
        COL_NUM	\t{self.prop['COL_NUM']} \t# Number of colors used [3-def]
        COL_SIGMA \t{self.prop['COL_SIGMA']} \t# Enlarge of the obs. color-errors[3-def]
        COL_SEL \t{self.prop['COL_SEL']} \t# Combination between used colors [AND/OR-def]
        #
        #########   MAGNITUDE SHIFTS applied to libraries   ######
        #
        # APPLY_SYSSHIFT  0.             # Apply systematic shifts in each band
                                         # used only if number of shifts matches
                                         # with number of filters in the library    
        #
        #########   ADAPTIVE METHOD using Z spectro sample     ###
        #
        AUTO_ADAPT \t{self.prop['AUTO_ADAPT']} \t\t# Adapting method with spectro [NO-def]
        ADAPT_BAND \t{self.prop['ADAPT_BAND']} \t\t# Reference band, band1, band2 for color 
        ADAPT_LIM \t{self.prop['ADAPT_LIM']} \t\t# Mag limits for spectro in Ref band [18,21.5-def]
        ADAPT_POLY\t{self.prop['ADAPT_POLY']} \t\t# Number of coef in  polynom (max=4) [1-def]
        ADAPT_METH \t{self.prop['ADAPT_METH']} \t\t# Fit as a function of 1 : Color Model  [1-def], 2 : Redshift, 3 : Models
        ADAPT_CONTEXT \t{self.prop['ADAPT_CONTEXT']} \t\t# Context for bands used for training, -1[-def] used context per object
        ADAPT_ZBIN \t{self.prop['ADAPT_ZBIN']} \t# Redshift's interval used for training [0.001,6-Def]
        ADAPT_MODBIN \t{self.prop['ADAPT_MODBIN']}'\t\t# Model's interval used for training [1,1000-Def]
        ERROR_ADAPT \t{self.prop['ERROR_ADAPT']} \t\t# [YES,NO-def] Add error in quadrature according to the difference between observed and predicted apparent magnitudes 
        #
        '''      
        
        return text
    
    def __call__(self, catalogue, skipSEDgen=False, skipFilterGen=False, skipMagGen=False, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Start the SED fitting on the given data using the built-in SED fitting parameters.
        
        :param LePhareCat catalogue: catalogue to use for the SED-fitting
        :param str inputFile: name of the input file containing 
        
        :param bool skipSEDgen: (**Optional**) whether to skip the SED models generation. Useful to gain time if the same SED are used for multiple sources.
        :param bool skipFilterGen: (**Optional**) whether to skip the filters generation. Useful to gain time if the same filters are used for multiple sources.
        :param bool skipMagGen: (**Optional**) whether to skip the predicted magnitude computations. Useful to gain time if the same libraries/parameters are used for multiple sources.

        '''
        
        if not isinstance(catalogue, LePhareCat):
            raise TypeError(f'catalogue has type {type(catalogue)} but it must be a LePhareCat instance.')
        
        # Make output directory
        directory = self.id
        if not opath.isdir(directory):
            os.mkdir(directory)
            
        # Generate and write parameters file
        params  = dedent(self.parameters.replace('%INPUTCATALOGUEINFORMATION%', catalogue.text))
        pfile   = opath.join(directory, catalogue.name.replace('.in', '.para'))
        with open(pfile, 'w') as f:
            f.write(params)
            
        # Write catalogue
        catalogue.save(path=directory)
        
        #########################################
        #          Generate SED models          #
        #########################################
        
        if not skipSEDgen:
            
            command = opath.expandvars('$LEPHAREDIR/source/sedtolib')
            if not opath.isfile(command):
                raise OSError(f'LePhare script $LEPHAREDIR/source/sedtolib (expanded as {command}) not found.')
            
            # Generate QSO, Star and Galaxy models
            for name in ['QSO', 'Stellar', 'Galaxy']:
                with subprocess.Popen([command, '-t', name[0], '-c', pfile], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as p:
                    for line in p.stdout:
                        print(line, end='')
                        
                if p.returncode != 0:
                    raise subprocess.CalledProcessError(f'{name} models generation failed.')
            
        ####################################
        #         Generate filters         #
        ####################################
        
        command = opath.expandvars('$LEPHAREDIR/source/filter')
        if not opath.isfile(command):
            raise OSError(f'LePhare script $LEPHAREDIR/source/filter (expanded as {command}) not found.')
            
        with subprocess.Popen([command, '-c', pfile], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                print(line, end='')
                
        if p.returncode != 0:
            raise subprocess.CalledProcessError('filters generation failed.')
        
        ##############################################
        #        Compute predicted magnitudes        #
        ##############################################
        
        if not skipMagGen:
            
            # QSO magnitudes
            command = opath.expandvars('$LEPHAREDIR/source/mag_gal')
            if not opath.isfile(command):
                raise OSError(f'LePhare script $LEPHAREDIR/source/mag_gal (expanded as {command}) not found.')
        
            with subprocess.Popen([command, '-t', 'Q', '-c', pfile], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as p:
                for line in p.stdout:
                    print(line, end='')
                
            if p.returncode != 0:
                raise subprocess.CalledProcessError('QSO magnitudes failed.')
                
            # Star magnitudes
            command = opath.expandvars('$LEPHAREDIR/source/mag_star')
            if not opath.isfile(command):
                raise OSError(f'LePhare script $LEPHAREDIR/source/mag_star (expanded as {command}) not found.')
        
            with subprocess.Popen([command, '-c', pfile], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as p:
                for line in p.stdout:
                    print(line, end='')
                
            if p.returncode != 0:
                raise subprocess.CalledProcessError('Stellar magnitudes failed.')
                
            # Galaxy magnitudes
            command = opath.expandvars('$LEPHAREDIR/source/mag_gal')
            if not opath.isfile(command):
                raise OSError(f'LePhare script $LEPHAREDIR/source/mag_gal (expanded as {command}) not found.')
        
            with subprocess.Popen([command, '-t', 'G', '-c', pfile], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as p:
                for line in p.stdout:
                    print(line, end='')
                
            if p.returncode != 0:
                raise subprocess.CalledProcessError('Galaxy magnitudes failed.')
        
        ###################################
        #         Run SED fitting         #
        ###################################
        
        command = '$LEPHAREDIR/source/zphota'
        if not opath.isfile(command):
                raise OSError(f'LePhare script $LEPHAREDIR/source/zphota (expanded as {command}) not found.')
        
        with subprocess.Popen([command, '-c', pfile], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                print(line, end='')
                
            if p.returncode != 0:
                raise subprocess.CalledProcessError('SED fitting failed.')
        
        return
 