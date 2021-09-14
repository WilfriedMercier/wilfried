    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Hugo Plombat - LUPM & Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Utilties related to generating 2D mass and SFR maps using either LePhare or CIGALE SED fitting codes.
"""

import yaml
import os.path                   as     opath
import numpy                     as     np
import astropy.io.fits           as     fits
from   astropy.table             import Table
from   copy                      import deepcopy
from   functools                 import reduce, partialmethod

from   .photometry               import countToMag, countToFlux
from   .symlinks.coloredMessages import *

# Custom colored messages
WARNING = warningMessage('Warning: ')
ERROR   = errorMessage('Error: ')


################################
#        Filter objects        #
################################

class Filter:
    r'''Base class implementing data related to a single filter.'''
    
    def __init__(self, filt, file, errFile, zeropoint, ext=0, extErr=0, texpFactor=1):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
            
        Initialise filter object.

        :param str filt: filter name
        :param str file: data file name. File must exist and be a loadable FITS file.
        :param str errFile: error file name. File must exist and be a loadable FITS file. Error file is assumed to be the variance map.
        :param float zeropoint: filter AB magnitude zeropoint
        
        :param int ext: (**Optional**) extension in the data FITS file
        :param int extErr: (**Optional**) extension in the error FITS file
        
        :raises TypeError:
            
            * if **filt** is not of type str
            * if **zeropoint** is neither an int nor a float
        '''
        
        if not isinstance(filt, str):
            raise TypeError(f'filt parameter has type {type(filt)} but it must be of type list.')
            
        if not isinstance(zeropoint, (int, float)):
            raise TypeError(f'zeropoint parameter has type {type(zeropoint)} but it must be of type int or float.')
            
        self.filter          = filt
        self.zpt             = zeropoint
        self.fname           = file
        self.ename           = errFile
        
        self.hdr,  self.data = self._loadFits(self.fname, ext=ext)
        self.ehdr, self.var  = self._loadFits(self.ename, ext=extErr)
        
        if self.data.shape != self.var.shape:
            raise ShapeError(self.data, self.var, msg=f' in filter {self.filter}')
            
            
    ###############################
    #       Private methods       #
    ###############################
    
    @staticmethod
    def _mask(arr, mask, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Apply the given mask by placing NaN values onto the array.
        
        :param ndarary arr: array to mask
        :param ndarary[bool] mask: mask with boolean values
        
        :returns: array with masked values
        :rtype: ndarray
        '''
        
        # We make a deep copy to be sure we do not modify the input arrays
        arr       = deepcopy(arr)
        arr[mask] = np.nan
        
        return arr
            
    def _checkFile(self, file, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
            
        Check whether a file exists.
        
        :param str file: file name
        
        :returns: 
            * True if the file exists
            * False otherwise
            
        :raises TypeError: if **file** is not of type str
        '''
        
        if not isinstance(file, str):
            raise TypeError(f'file has type {type(file)} but it must have type str.')
        
        if not opath.isfile(file):
            print(ERROR + 'file ' + brightMessage(file) + ' not found.')
            return False
        return True
    
    def _loadFits(self, file, ext=0, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Load data and header from a FITS file at the given extension.

        :param str file: file name
        :param int ext: (**Optional**) extension to load data from

        :returns: 
            * None, None if the file cannot be loaded as a FITS file or if the hdu extension is too large
            * header, data
        :rtype: astropy Header, ndarray
        
        :raises TypeError: if **ext** is not an int
        :raises ValueError: if **ext** is negative
        '''
        
        if not isinstance(ext, int):
            raise TypeError(f'ext has type {type(ext)} but it must have type int.')
        elif ext < 0:
            raise ValueError(f'ext has value {ext} but it must be larger than or equal to 0.')
        
        if self._checkFile(file):
            try:
                with fits.open(file) as hdul:
                    hdu = hdul[ext]
                    return hdu.header, hdu.data
                    
            # If an error is triggered, we always return None, None
            except OSError:
                print(ERROR + ' file ' + brightMessage(file) + ' could not be loaded as a FITS file.')
            except IndexError:
                print(ERROR + f' extension number {ext} too large.')
                
        return None, None


class FilterList:
    r'''Base class implementing the object used to stored SED fitting into.'''
    
    def __init__(self, filters, mask, code='cigale', redshift=0, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
            
        Initialise filter list object.

        :param list[Filter] filters: filters used to perform the SED fitting
        :param ndarray[bool] mask: mask for bad pixels (True for bad pixels, False for good ones)
        
        :param str code: (**Optional**) code used to perform the SED fitting. Either 'lephare' or 'cigale' are accepted.
        :param redshift: (**Optional**) redshift of the galaxy
        
        :raises TypeError: 
            * if **filters** is not a list
            * if **redshift** is neither an int nor a float
            * if one of the filters is not of type Filter
        '''
        
        ##############################
        #      Check parameters      #
        ##############################
            
        if not isinstance(redshift, (int, float)):
            raise TypeError(f'redshift parameter has type {type(redshift)} but it must be of type int or float.')
            
        if not isinstance(filters, list):
            raise TypeError(f'filters parameter has type {type(filters)} but they must be of type list.')
        
        ###################################
        #         Init attributes         #
        ###################################
        
        # :Redshift of the galaxy
        self.redshift = redshift
        
        # :Define a mask which hides pixels
        self.mask     = mask
        
        # :Table used by SED fitting code (default is None)
        self.table    = None
        
        #########################################
        #           Build filter list           #
        #########################################
        
        self.filters = []
        for filt in filters:
            
            if filt in [i.filter for i in self.filters]:
                print(WARNING + 'filter ' +  brightMessage(filt.filter) +  ' already present in filter list.')
                print(errorMessage(f'Skipping filter {filt.filter}...'))
            else:
                
                if not isinstance(filt, Filter):
                    raise TypeError(f'One of the filters has type {type(filt)} but it must have type Filter.')
                
                elif np.any([i is None for i in [filt.data, filt.var, filt.hdr, filt.ehdr]]):
                    print(errorMessage(f'Skipping filter {filt.filter}...'))
                else:
                    self.filters.append(filt)
                    
        # Check that data in all filters have the same shape
        if len(self.filters) > 0:
            for f in self.filters[1:]:
                if f.data.shape != self.filters[0].data.shape:
                    raise ShapeError(f.data, self.filters[0], msg=' in filter list')
                    
        # Set SED fitting code. This rebuilds the table since SED fitting codes do not expect tables with the same columns
        self.setCode(code)
        
        
    ##############################
    #       Table creation       #
    ##############################
    
    def toTable(self, cleanMethod='zero', scaleFactor=100, texpFac=0, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Generate an input table for the SED fitting codes.
        
        :param str cleanMethod: (**Optional**) method used to clean pixel with negative values. Accepted values are 'zero' and 'min'.
        :param scaleFactor: (**Optional**) factor used to multiply data and std map. Only used if SED fitting code is LePhare.
        :type scaleFactor: int or float
        :param int texpFactor: (**Optional**) exposure factor used to divide the exposure time when computing Poisson noise. A value of 0 means no Poisson noise is added to the variance map.
        
        :returns: output table
        :rtype: Astropy Table
        
        :raises ValueError: if there are no filters in the filter list
        '''
        
        if len(self.filters) < 1:
            raise ValueError('At least one filter must be in the filter list to build a table.')
        
        # Mean map (NaN values are set to 0 once the mean map is computed)
        if self.code.lower() == 'lephare':
            meanMap, _      = self.meanMap(maskVal=0)

        dataList            = []
        stdList             = []
        for filt in self.filters:
            
            # Clean data and error maps of bad pixels and pixels with negative values
            data, var       = self.clean(filt.data, filt.var, self.mask, method=cleanMethod)
            shp             = data.shape
        
            # Add Poisson noise to the variance map
            try:
                texp        = filt.hdr['TEXPTIME']
            except KeyError:
                print(ERROR + f'data header in {filt.filter} does not have TEXPTIME key. Cannot compute poisson variance.')
            else:
                var        += self.poissonVar(data, texp=texp, texpFac=texpFac)
            
            # Scaling data for LePhare
            if self.code.lower() == 'lephare':
                data, var   = self.scale(data, var, meanMap, factor=scaleFactor)
            
            # Transform data and error maps into 1D vectors
            data            = data.reshape(shp[0]*shp[1])
            var             = var.reshape( shp[0]*shp[1])
            
            # Get rid of NaN values
            nanMask         = ~(np.isnan(data) | np.isnan(var))
            data            = data[nanMask]
            var             = var[ nanMask]
            
            # Convert flux and variance to AB mag for LePhare
            if self.code.lower() == 'lephare':
                
                # 0 values are cast to NaN otherwise corresponding magnitude would be infinite
                mask0       = (np.asarray(data == 0) | np.asarray(var == 0))
                data[mask0] = np.nan
                var[ mask0] = np.nan
                    
                # Compute std instead of variance
                data, std   = countToMag(data, np.sqrt(var), filt.zpt)
                
                # Cast back pixels with NaN values to -99 mag to specify they are not to be used in the SED fitting
                data[mask0] = -99
                std[ mask0] = -99
                
            # Convert to mJy for Cigale
            elif self.code.lower() == 'cigale':
                
                # Compute std and convert std and data to mJy unit
                data, std   = [i.to('mJy').value for i in countToFlux(data, np.sqrt(var), filt.zpt)]
            
            # Append data and std to list
            dataList.append(data)
            stdList.append( std)
            
        # Keep track of indices with correct values (identical between filters)
        indices             = np.where(nanMask)[0]
        
        # Consistency check
        ll                  = len(indices)
        if ll != len(dataList[0]) or ll != len(stdList[0]):
            raise ValueError(f'indices have length {ll} but dataList and stdList have shapes {np.shape(dataList)} and {np.shape(stdList)}.')
        
        ####################################
        #         Generate columns         #
        ####################################
        
        # Shared between LePhare and Cigale
        zs                  = [self.redshift]*ll
        
        if self.code.lower() == 'lephare':
            
            # Compute context (number of filters used - see LePhare documentation) and redshift columns
            context         = [2**len(self.filters) - 1]*ll
            dtypes          = [int]     + [float]*2*len(self.filters)                                        + [int, float]
            colnames        = ['ID']    + [val for f in self.filters for val in [f.filter, f'e_{f.filter}']] + ['Context', 'zs']
            columns         = [indices] + [val for d, s in zip(dataList, stdList) for val in [d, s]]         + [ context, zs]
        
        elif self.code.lower() == 'cigale':
            
            dtypes          = [int, float]       + [float]*len(self.filters)
            colnames        = ['id', 'redshift'] + [val for f in self.filters for val in [f.filter, f'{f.filter}_err']]
            columns         = [indices, zs]      + [val for d, s in zip(dataList, stdList) for val in [d, s]]
        
        # Generate the output Table
        self.table          = Table(columns, names=colnames, dtype=dtypes)
        
        return self.table
     
        
    #############################
    #       Data handling       #
    #############################
    
    @staticmethod
    def clean(data, var, mask, method='zero', **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Clean given data and error maps by masking pixels and dealing with negative values.
        
        .. note::
                
            * If **method** is 'zero', negative values in the data and error maps are set to 0
            * If **method** is 'min', negative values in the data and error maps are set to the minimum value in the array
            * If **method** is neither 'zero' nor 'negative', 'zero' is used as default
            
        :param ndarray data: data map
        :param ndarray var: variance map
        :param ndarray[bool] mask: mask used to apply NaN values
        
        :param str method: (**Optional**) method to deal with negative values
        
        :returns: cleaned data and variance maps
        :rtype: ndarray, ndarray
        
        :raises TypeError: if **method** is not of type str
        '''
        
        if not isinstance(method, str):
            raise TypeError(f'method parameter has type {type(method)} but it must have type str.')
        
        method            = method.lower()
        if method not in ['zero', 'min']:
            print(WARNING + f'method {method} not recognised. Using ' +  brightMessage('zero') +  ' as default.')
            method        = 'zero'
        
        # Deep copies to avoid to overwrite input arrays
        data              = deepcopy(data)
        var               = deepcopy(var)
        
        # Apply mask
        data[mask]        = np.nan
        var[ mask]        = np.nan
        
        # Mask pixels having negative values
        negMask           = (data < 0) | (var < 0)
        if method == 'zero':
            data[negMask] = 0
            var[ negMask] = 0
        elif method == 'min':
            mini          = np.nanmin(data[~negMask])
            data[negMask] = mini
            var[ negMask] = mini
            
        return data, var
    
    def meanMap(self, maskVal=0, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Compute the averaged data and error maps over the spectral dimension for non masked pixels.
        
        :param maskVal: (**Optional**) value to put into masked pixels
        :type maskVal: int or float
        
        :returns: averaged data and averaged error map
        :rtype: ndarray, ndarray
        
        :raises TypeError: if **maskVal** is neither int nor float
        '''
        
        if not isinstance(maskVal, (int, float)):
            raise TypeError(f'maskVal parameter has type {type(maskVal)} but it must have type int or float.')
            
        # Compute masked arrays
        data  = [f._mask(f.data, self.mask) for f in self.filters]
        err   = [f._mask(f.var,  self.mask) for f in self.filters]
        
        # Compute mean value along spectral dimension
        data  = np.nanmean(data, axis=0)
        err   = np.nanmean(err,  axis=0)
        
        # Replace NaN values
        np.nan_to_num(data, copy=False, nan=maskVal)
        np.nan_to_num(err,  copy=False, nan=maskVal)
        
        return data, err
    
    @staticmethod
    def poissonVar(data, texp=1, texpFac=1, **kwargs):
        '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Compute a scaled Poisson variance term from a given flux map. The variance :math:`(\Delta F)^2` is computed as
        
        .. math::
            
            (\Delta F)^2 = \alpha F
            
        where :math:`F` is the flux map and :math:`\alpha` is a scale factor defined as
        
        .. math::
            
            \alpha = {\rm{TEXP / TEXPFAC}}
            
        where :math:`\rm{TEXP}` is the exposure time and :math:`\rm{TEXPFAC}` is a coefficient used to scale it down.
        
        :param ndarray data: flux map
        
        :param texp: (**Optional**) exposure time in seconds
        :type text: int or float
        :param texpFac: (**Optional**) exposure factor
        :type texpFac: int or float
        
        :raises TypeError: if **texp** or **texpFac** are not both int or float
        :raises ValueError:
            
            * if **texp** is less than or equal to 0
            * if **texpFac** is less than 0
        '''
        
        if not all([isinstance(i, (int, float)) for i in [texp, texpFac]]):
            raise TypeError(f'texp and texpFac parameters have types {type(texp)} and {type(texpFac)} but they must have type int or float.')
            
        if texp <= 0:
            raise ValueError(f'texp has value {texp} but it must be positive.')
            
        if texpFac < 0:
            raise ValueError(f'texpFac has value {texpFac} but it must be positive or null.')
        
        return data * texpFac / texp
    
    @staticmethod
    def scale(data, var, norm, factor=100):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Normalise given data and error maps using a norm map and scale by a certain amount. Necessary for LePhare SED fitting code.
        
        :param ndarray data: data map
        :param ndarray var: variance map
        :param ndarray norm: normalisation map which divides data and error maps
        
        :param factor: (**Optional**) scale factor which multiplies the output array
        :type factor: int or float
        
        :returns: scaled data and variance maps
        :rtype: ndarray, ndarray
        
        :raises ValueError: if **data** and **norm** do not have the same shapes
        '''
        
        if norm.shape != data.shape:
            raise ValueError(f'Incompatible norm and data shapes. norm map has shape {norm.shape} but data map has shape {self.data.shape}.')
        
        # Deep copies to avoid to overwrite input arrays
        d        = deepcopy(data)
        v        = deepcopy(var)
        
        mask     = norm != 0
        d[mask] *= factor/norm[mask]
        v[mask] *= (factor*factor/(norm[mask]*norm[mask])) # Variance normalisation is squared
        
        return d, v
        
        
    ####################
    #       Misc       #
    ####################    
    
    def setCode(self, code, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Set the SED fitting code.
        
        .. warning::
            
            This function also recomputes and rewrites the output table used for the SED fitting.
            If you want a table with different parameters you must run :py:meth:`FilterList.toTable` again, for e.g.
            
            >>> flist = FilterList(filters, mask)                # setCode and toTable methods are called with default SED fitting code name
            >>> flist.setCode('lephare')                         # setCode and toTable methods are called with 'lephare' SED fitting code name
            >>> flist.toTable(cleanMethod='min', scaleFactor=50) # toTable is run again with different parameters but still 'lephare' SED fitting code name
        
        :param str code: code used for SED fitting acceptable values are cigale and lephare. If code name is not recognised, cigale is set as default value.
        
        :raises TypeError: if **code** is not of type str
        '''
        
        if not isinstance(code, str):
            raise TypeError(f'code parameter has type {type(code)} but it must be of type str.')
        
        self.code     = code.lower()
        if self.code not in ['cigale', 'lephare']:
            print(WARNING + f'code name {self.code} could not be resolved. Acceptable values are cigale or lephare. Using ' + brightMessage('cigale') + ' as default.')
            self.code = 'cigale'
            
        # Update output table with default parameters
        self.toTable()
            
        return
    
    
    #############################
    #      Partial methods      #
    #############################
    
    #: Set Cigale as fitting code
    setCigale  = partialmethod(setCode, 'cigale')
    
    #: Set LePhare as fitting code
    setLePhare = partialmethod(setCode, 'lephare')
        
    
#########################################
#              SED objects              #
#########################################
        
class SED:
    '''General SED object used for inheritance.'''
    
    def __init__(self, *args, **kwargs):
        '''Init SED oject.'''
        
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
        '''Generate a parameter file used by the SED fitting code.'''
        
        raise NotImplementedError('genParams method not implemented.')
    
    def run(self, *args, **kwargs):
        '''Run the SED fitting code.'''
        
        raise NotImplementedError('run method not implemented.')
        
    ###########################
    #     Private methods     #
    ###########################
        
    def __getitem__(self, key):
        '''
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
        '''
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
            self.prop[key] = value
        
        return
            
        
class LePhare(SED):
    '''Implements LePhare SED object.'''
    
    def __init__(self, properties, *args, **kwargs):
        '''
        Init LePhare object.
        
        :param dict properties: properties to be passed to LePhare to compute the models grid and perform the SED fitting
        
        Accepted properties are:
        
            * **STAR_SED**: stellar library list file (full path)
            * **STAR_FSCALE**: stellar flux scale
            * **STAR_LIB**: stellar library to use (default libraries found at $LEPHAREWORK/lib_bin)
            * **QSO_SED**: QSO list file (full path)
            * **QSO_FSCALE**: QSO flux scale
            * **QSO_LIB**: QSO library to use (default libraries found at $LEPHAREWORK/lib_bin)
            * **GAL_SED**: galaxy library list file (full path)
            * **GAL_FSCALE** : galaxy flux scale
            * **GAL_LIB**: galaxy library to use (default libraries found at $LEPHAREWORK/lib_bin)
            * **SEL_AGE**: stellar ages list (full path)
            * **AGE_RANGE**: minimum and maximum ages in years
            * **FILTER_LIST**: list of filter names used for the fit (must all be located in $LEPHAREDIR/filt directory)
            * **TRANS_TYPE**: transmission type (0 for Energy, 1 for photons)
            * **FILTER_CALIB**: filter calibration (0 for fnu=ctt, 1 for nu.fnu=ctt, 2 for fnu=nu, 3=fnu=Black Body @ T=10000K, 4 for MIPS (leff with nu fnu=ctt and flux with BB @ 10000K) 
            * **FILTER_FILE**: filter file (must be located in $LEPHAREWORK/filt directory)
            * **STAR_LIB_IN**: input stellar library (dupplicate with **STAR_LIB** ?)
            * **STAR_LIB_OUT**: output stellar magnitudes
            * **QSO_LIB_IN**: input QSO library (dupplicate with **QSO_LIB** ?)
            * **QSO_LIB_OUT**: output QSO magnitudes
            * **GAL_LIB_IN**: input galaxy library (dupplicate with **GAL_LIB** ?)
            * **GAL_LIB_OUT**: output galaxy magnitudes
            
        .. warning::
            
            It is mandatory to define on your OS two environment variables:
                
                * $LEPHAREWORK which points to LePhare working directory
                * $LEPHAREDIR which points to LePhare main directory
                
            These paths may be expanded to check whether the given files exist and can be used by the user to shorten some path names when providing the SED properties.
        '''
        
        super().__init__(*args, **kwargs)
        
        self.code       = 'lephare'
        #self.prop       = properties
        
        # Allowed keys and corresponding allowed types
        self.prop = {'STAR_SED'     : PathlikeProperty(opath.join('$LEPHAREDIR', 'sed', 'STAR', 'STAR_MOD.list')),
                     'STAR_FSCALE'  : Property(3.432e-09, (int, float), minBound=0),
                     'STAR_LIB'     : PathlikeProperty('LIB_STAR_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin')),
                     'QSO_SED'      : PathlikeProperty(opath.join('$LEPHAREDIR', 'sed', 'QSO', 'QSO_MOD.list')),
                     'QSO_FSCALE'   : Property(1, (int, float), minBound=0),
                     'QSO_LIB'      : PathlikeProperty('LIB_QSO_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin')),
                     'GAL_SED'      : PathlikeProperty(opath.join('$LEPHAREDIR', 'sed', 'GAL', 'BC03_CHAB', 'BC03_MOD.list')),
                     'GAL_FSCALE'   : Property(1, (int, float), minBound=0),
                     'GAL_LIB'      : PathlikeProperty('LIB_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin')),
                     'SEL_AGE'      : PathlikeProperty(opath.join('$LEPHAREDIR', 'sed', 'GAL', 'BC03_CHAB', 'BC03_AGE.list')),
                     'AGE_RANGE'    : Property([0, 14e9], list, subtypes=(int, float), minBound=0),
                     'FILTER_LIST'  : PathlikeProperty(['hst/acs_f435w.pb', 'hst/acs_f606w.pb', 'hst/acs_f775w.pb', 'hst/acs_f850lp.pb'], 
                                                       path=opath.join('$LEPHAREDIR', 'filt')),
                     'TRANS_TYPE'   : Property(0, int, minBound=0, maxBound=1),
                     'FILTER_CALIB' : Property(0, int, minBound=0, maxBound=4),
                     'FILTER_FILE'  : PathlikeProperty('HDF_bc03.filt', path=opath.join('$LEPHAREWORK', 'filt')),
                     'STAR_LIB_IN'  : PathlikeProperty('LIB_STAR_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin')),
                     'STAR_LIB_OUT' : Property('STAR_HDF_bc03', str), # We do not use PathlikeProperty because this is an output file
                     'QSO_LIB_IN'   : PathlikeProperty('LIB_QSO_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin')),
                     'QSO_LIB_OUT'  : Property('QSO_HDF_bc03', str), # We do not use PathlikeProperty because this is an output file
                     'GAL_LIB_IN'   : PathlikeProperty('LIB_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin')),
                     'GAL_LIB_OUT'  : Property('HDF_bc03', str), # We do not use PathlikeProperty because this is an output file
                     }
        
        
        # Param file text
        self.paramText  = None
    
    def genParams(self, *args, **kwargs):
        '''Generate a parameter file used by the SED fitting code.'''
        
        ######################################
        #         Libraries handling         #
        ######################################
        
        text = f'''\
        ##############################################################################
        #                CREATION OF LIBRARIES FROM SEDs List                        #
        # $LEPHAREDIR/source/sedtolib -t (S/Q/G) -c $LEPHAREDIR/config/zphot.para    #
        # help : $LEPHAREDIR/source/sedtolib -h (or -help)                           #
        ##############################################################################
        #
        #------      STELLAR LIBRARY (ASCII SEDs)
        STAR_SED \t{self.prop['STAR_SED']} \t# STAR list (full path)
        STAR_FSCALE \t{self.prop['STAR_FSCALE']} \t# Arbitrary Flux Scale
        STAR_LIB \t{self.prop['STAR_LIB']} \t# Bin. STAR LIBRARY -> $LEPHAREWORK/lib_bin
        #
        #------      QSO LIBRARY (ASCII SEDs)
        QSO_SED \t{self.prop['QSO_SED']} \t# QSO list (full path)
        QSO_FSCALE \t{self.prop['QSO_FSCALE']} \t# Arbitrary Flux Scale 
        QSO_LIB	\t{self.prop['QSO_LIB']} \t# Bin. QSO LIBRARY -> $LEPHAREWORK/lib_bin
        #
        #------      GALAXY LIBRARY (ASCII or BINARY SEDs)
        GAL_SED	\t{self.prop['GAL_SED']} \t# GAL list (full path)
        GAL_FSCALE \t{self.prop['GAL_FSCALE']} \t# Arbitrary Flux Scale
        GAL_LIB	\t{self.prop['GAL_LIB']} \t# Bin. GAL LIBRARY -> $LEPHAREWORK/lib_bin
        SEL_AGE \t{self.prop['SEL_AGE']} \t# Age list(full path, def=NONE)	
        AGE_RANGE \t{self.prop['AGE_RANGE']} \t# Age Min-Max in yr
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
        TRANS_TYPE \t{self.prop['TRANS_TYPE']} \t# TRANSMISSION TYPE
        FILTER_CALIB \t{self.prop['FILTER_CALIB']} \t# 0[-def]: fnu=ctt, 1: nu.fnu=ctt, 2: fnu=nu, 3: fnu=Black Body @ T=10000K, 4: for MIPS (leff with nu fnu=ctt and flux with BB @ 10000K
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
        QSO_LIB_IN \t{self.prop['QSO_LIB_IN']} \t# Input  QSO LIBRARY  in $LEPHAREWORK/lib_bin/
        QSO_LIB_OUT	\t{self.prop['QSO_LIB_OUT']} \t# Output QSO MAGN     -> $LEPHAREWORK/lib_mag/
        #
        #-------     From GALAXY  LIBRARY  
        GAL_LIB_IN \t{self.prop['GAL_LIB_IN']} \t# Input  GAL LIBRARY  in $LEPHAREWORK/lib_bin/
        GAL_LIB_OUT	\t{self.prop['GAL_LIB_OUT']} \t# Output GAL LIBRARY  -> $LEPHAREWORK/lib_mag/ 
        #
        #-------   MAG + Z_STEP + EXTINCTION + COSMOLOGY
        '''        
 

###################################
#          Miscellaneous          #
###################################
        
class ShapeError(Exception):
    '''Error which is caught when two arrays do not share the same shape.'''
    
    def __init__(self, arr1, arr2, msg='', **kwargs):
        '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Init method for this exception.
        
        :param ndarray arr1: first array
        :param ndarray arr2: second array
        
        :param str msg: (**Optional**) message to append at the end
        '''
        
        if not isinstance(msg, str):
            msg = ''
        
        super.__init__(f'Array 1 has shape {arr1.shape} but array 2 has shape {arr2.shape}{msg}.')
        
class PathlikeProperty(Property):
    '''Define a property object where the data stored must be a valid path or file.'''
    
    def __init__(self, default, path='', *args, **kwargs):
        '''
        Init the path-like object. Path-like type must either be str or list of str.
        
        :param default: default value used at init
        
        :raises TypeError: if **path** is not of type str
        
        .. seealso:: :py:class:`Property`
        '''
        
        if not isinstance(path, str):
            raise TypeError(f'path has type {type(path)} but it must have type str.')
        
        # Set path to append at the beginning of a check
        self.path    = path
        
        super().__init__(name, default, (list, str), subtypes=str, minBound=None, maxBound=None, **kwargs)
        
        # Need to call set again since it was called in super
        self.set(default)
        self.default = default
        
        
    def set(self, value, *args, **kwargs):
        '''
        Set the current value.

        :param value: new value. Must be of correct type, and within bounds.

        :raises TypeError: 
            
            * if **value** does not have correct type (str or list)
            * if **value** is a list and at least one value in the list is not of correct subtype (str)
        
        :raises OSError: if expanded path (**value**) is neither a valid path, nor a valid file name
        '''
        
        if not isinstance(value, self.types):
            raise TypeError(f'cannot set property with value {value} of type {type(value)}. Acceptable types are {self.types}.')
            
        if self.types == list:
            
            if not all([isinstance(i, self.subtypes) for i in value]):
                raise TypeError(f'at least one value does not have the type {self.subtypes}.')

            for p in value:    
                path  = opath.join(self.path, p)
                epath = opath.expandvars(path)
                
                if not opath.exists(path) and not opath.isfile(path):
                    raise OSError(f'path {path} (expanded as {epath}) does not exist.')
            
        else:
                
            path     = opath.join(self.path, value)
            epath    = opath.expandvars(path)
            if not opath.exists(path) and not opath.isfile(path):
                raise OSError(f'path {path} (expanded as {epath}) does not exist.')
        
        self.value   = value
        return
        
class Property:
    '''Define a property object used by SED objects to store SED parameters.'''
    
    def __init__(self, default, types, subtypes=None, minBound=None, maxBound=None, **kwargs):
        '''
        Init the property object.

        :param type: type of the property. If the property is a list, provide the elements type in **subtypes**.
        :param default: default value used at init

        :param subtypes: (**Optional**) type of the elements if **type** is a list. If None, it is ignored.
        :param minBound: (**Optional**) minimum value for the property. If None, it is ignored.
        :param maxBound: (**Optional**) maximum value for the property. If None, it is ignored.
        '''
        
        self.types     = types
        
        self.substypes = subtypes
        self.min       = minBound
        self.max       = maxBound
        
        self.set(default)
        self.default   = default
        
    ##################################
    #        Built-in methods        #
    ##################################
    
    def __str__(self, *args, **kwargs):
        '''Implement a string representation of the class.'''
        
        typ = type(self.value)
        
        if typ == int:
            return f'{self.value}'
        elif typ == float:
            return f'{self.value:.3e}'
        elif self.types == str:
            return self.value
        elif typ == list:
            
            subtyp = type(self.value[0])
            
            if subtyp == int:
                return ','.join([f'{i}' for i in self.value])
            elif subtyp == float:
                return ','.join([f'{i:.3e}' for i in self.value])
            elif subtyp == str:
                return ','.join(self.value)
            else:
                raise NotImplementedError('no string representation available for Property object with value of type {self.subtypes}.')
        else:
             raise NotImplementedError('no string representation available for Property object with value of type {self.types}.')
        
    ###############################
    #        Miscellaneous        #
    ###############################
    
    def set(self, value, *args, **kwargs):
        '''
        Set the current value.

        :param value: new value. Must be of correct type, and within bounds.

        :raises TypeError: 
            
            * if **value** does not have correct type
            * if **value** is a list and at least one value in the list is not of correct subtype
        
        :raises ValueError: 
            
            * if **value** is below the minimum bound
            * if **value** is above the maximum bound
        '''
        
        if not isinstance(value, self.types):
            raise TypeError(f'cannot set property with value {value} of type {type(value)}. Acceptable types are {self.types}.')
            
        if self.types == list and not all([isinstance(i, self.subtypes) for i in value]):
            raise TypeError(f'at least one value does not have the type {self.subtypes}.')
            
        if self.min is not None and value < self.min:
            raise ValueError('value is {value} but minimum acceptable bound is {self.min}.')
        
        if self.max is not None and value > self.max:
            raise ValueError('value is {value} but maximum acceptable bound is {self.max}.')
        
        self.value = value
        return
    
    
    