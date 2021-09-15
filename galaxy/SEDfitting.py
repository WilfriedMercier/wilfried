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


###################################
#        Catalogue objects        #
###################################

class Catalogue:
    r'''Class implementing a catalogue consisting of as Astropy Table and additional information used by the SED fitting codes.'''
    
    def __init__(self, table, tunit='M', magtype='AB', tformat='MEME', ttype='LONG'):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Init Catalogue.

        :param table: input table
        :type table: Astropy Table
        
        :param str tunit: (**Optional**) unit of the table data. Must either be 'M' for magnitude or 'F' for flux.
        :param str magtype: (**Optional**) magnitude type if data are in magnitude unit. Must either be 'AB' or 'VEGA'.
        :param str tformat: (**Optional**) format of the table. Must either be 'MEME' if data and error columns are intertwined or 'MMEE' if columns are first data and then errors.
        :param str ttype: (**Optional**) data type. Must either be SHORT or LONG.

        :raises TypeError:
             
            * if **table** is not an astropy Table
            * if one of **tunit**, **magtype**, **tformat** or **ttype**  is not of type str
            
        :raises ValueError:
        
            * if **tunit** is neither F, nor M
            * if **magtype** is neither AB nor VEGA
            * if **tformat** is neither MEME nor MMEE
            * if **ttype** is neither SHORT nor LONG
        ''' 
        
        if not isinstance(table, Table):
            raise TypeError(f'table has type{type(table)} but it must be an Astropy Table.')
            
        if not all([isinstance(i, str) for i in [tunit, magtype, tformat, ttype]]):
            raise TypeError(f'one of the parameters in tunit, magtype, tformat or ttype is not of type str.')
            
        tunit   = tunit.upper()
        magtype = magtype.upper()
        tformat = tformat.upper()
        ttype   = ttype.upper()
            
        if tunit not in ['F', 'M']:
            raise ValueError(f'tunit has value {tunit} but it must either be F (for flux) or M (for magnitude).')
            
        if magtype not in ['AB', 'VEGA']:
            raise ValueError(f'magtype has value {magtype} but it must either be AB or VEGA.')
            
        if tformat not in ['MEME', 'MMEE']:
            raise ValueError(f'tformat has value {tformat} but it must either be MEME (if each error column follows its associated magnitude column) or MMEE (if magnitudes are listed first and then errors).')
            
        if ttype not in ['SHORT', 'LONG']:
            raise ValueError(f'ttype has value {ttype} but it must either be SHORT or LONG.')
            
        self.data   = table
        
        # Define default properties
        self.unit   = Property('M',    str)
        self.mtype  = Property('AB',   str)
        self.format = Property('MEME', str)
        self.ttype  = Property('LONG', str)
        
        # Set to given values
        self.unit.set(  tunit)
        self.mtype.set( magtype)
        self.format.set(tformat)
        self.ttype.set( ttype)
        
    def save(self, fname, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Save the catalogue into the given file.
        
        :param str fname: output file name
        :param *args: other parameters passed to Astropy.table.Table.writeto method
        :param **kwargs: other optional parameters passed to Astropy.table.Table.writeto method
        
        :raises TypeError: if **fname** is not of type str
        '''
        
        if not isinstance(fname, str):
            raise TypeError(f'output file name has type {type(fname)} but it must have type str.')
            
        self.data.writeto(fname, *args, **kwargs)
        return
    
    def textCigale(self, *args, **kwargs):
        raise NotImplementedError('this method has not been implemented yet.')
    
    def textLePhare(self, fname, nlines=[0, 100000000], *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Generate an Input Catalog Informations section for LePhare parameter file.
        
        :param str fname: input catalogue file name used by LePhare to load data
        
        :param list[int] nlines: minimum and maximum line numbers for which the SED fitting is performed onto
        
        :raises TypeError: 
            
            * if **fname** is not of type str
            * if **nlines** is not of type list
            
        :raises OSError: if **fname** is not a regular file
        :raises ValueError: 
            
            * if **nlines** is not a list of length 2
            * if one of the values in **nlines** is not an int
            * if maximum line number is less than minimum one
            * if minimum line number is less than 0
        '''
        
        if not isinstance(fname, str):
            raise TypeError(f'file name has type {type(fname)} but it must have type str.')
            
        if not isinstance(nlines, list):
            raise TypeError(f'nlines parameter has type {type(nlines)} but it must have type list.')
            
        if len(nlines) != 2 or not all([isinstance(i, int) for i in nlines]):
            raise ValueError(f'nlines must be a length 2 list of integers.')  
        elif nlines[1] < nlines[0]:
            raise ValueError('maximum line number must be greater than minimum one.')  
        elif nlines[0] < 0:
            raise ValueError('minimum line number must be greater than or equal to 0.')
            
        efname = opath.expandvars(fname)
        if not opath.isfile(efname):
            raise OSError(f'file {fname} (expanded as {efname}) not found.')
        
        text =  f'''
        #-------    Input Catalog Informations   
        CAT_IN \t{fname}

        INP_TYPE \t{self.unit} \t# Input type (F:Flux or M:MAG)
        CAT_MAG \t{self.mtype} \t# Input Magnitude (AB or VEGA)
        CAT_FMT \t{self.format} \t# MEME: (Mag,Err)i or MMEE: (Mag)i,(Err)i  
        CAT_LINES \t{nlines[0]},{nlines[1]} \t# MIN and MAX RANGE of ROWS used in input cat [def:-99,-99]
        CAT_TYPE \t{self.ttype} # Input Format (LONG,SHORT-def)
        '''
        
        return text
        
        
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
        
        
    #############################################
    #       Table  and catalogue creation       #
    #############################################
    
    def toCatalogue(self, tunit='M', magtype='AB', tformat='MEME', ttype='LONG'):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Construct a Catalogue instance given the table associated to the filter list.
        
        .. seealso:: :py:class:`Catalogue`
        '''
        
        return Catalogue(self.table, tunit=tunit, magtype=magtype, tformat=tformat, ttype=ttype)
        
    
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
            
        
class LePhare(SED):
    r'''Implements LePhare SED object.'''
    
    def __init__(self, ID, properties, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Init LePhare object.
        
        :param ID: an identifier used to name the output files created during the SED fitting process
        :param dict properties: properties to be passed to LePhare to compute the models grid and perform the SED fitting
        
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
            * **MAG_TYPE** [str]: magnitude system used (AB or VEGA)
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
        
        super().__init__(*args, **kwargs)
        
        self.id         = ID
        self.code       = 'lephare'
        
        # Allowed keys and corresponding allowed types
        self.prop = {'STAR_SED'       : PathlikeProperty(opath.join('$LEPHAREDIR', 'sed', 'STAR', 'STAR_MOD.list')),
                     'STAR_FSCALE'    : Property(3.432e-09, (int, float), minBound=0),
                     'STAR_LIB'       : PathlikeProperty('LIB_STAR_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin')),
                     'QSO_SED'        : PathlikeProperty(opath.join('$LEPHAREDIR', 'sed', 'QSO', 'QSO_MOD.list')),
                     'QSO_FSCALE'     : Property(1, (int, float), minBound=0),
                     'QSO_LIB'        : PathlikeProperty('LIB_QSO_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin')),
                     'GAL_SED'        : PathlikeProperty(opath.join('$LEPHAREDIR', 'sed', 'GAL', 'BC03_CHAB', 'BC03_MOD.list')),
                     'GAL_FSCALE'     : Property(1, (int, float), minBound=0),
                     'GAL_LIB'        : PathlikeProperty('LIB_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin')),
                     'SEL_AGE'        : PathlikeProperty(opath.join('$LEPHAREDIR', 'sed', 'GAL', 'BC03_CHAB', 'BC03_AGE.list')),
                     
                     'AGE_RANGE'      : Property([0, 14e9], list, subtypes=(int, float), minBound=0, 
                                                 testFunc=lambda value: len(value)!=2 or value[1] < value[0], 
                                                 testMsg='AGE_RANGE property must be an increasing length 2 list.'),
                     
                     'FILTER_LIST'    : PathlikeProperty(['hst/acs_f435w.pb', 'hst/acs_f606w.pb', 'hst/acs_f775w.pb', 'hst/acs_f850lp.pb'], path=opath.join('$LEPHAREDIR', 'filt')),
                     'TRANS_TYPE'     : Property(0, int, minBound=0, maxBound=1),
                     'FILTER_CALIB'   : Property(0, int, minBound=0, maxBound=4),
                     'FILTER_FILE'    : PathlikeProperty('HDF_bc03.filt', path=opath.join('$LEPHAREWORK', 'filt')),
                     'STAR_LIB_IN'    : PathlikeProperty('LIB_STAR_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin')),
                     'STAR_LIB_OUT'   : Property('STAR_HDF_bc03', str), # We do not use PathlikeProperty because this is an output file
                     'QSO_LIB_IN'     : PathlikeProperty('LIB_QSO_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin')),
                     'QSO_LIB_OUT'    : Property('QSO_HDF_bc03', str),  # We do not use PathlikeProperty because this is an output file
                     'GAL_LIB_IN'     : PathlikeProperty('LIB_bc03', path=opath.join('$LEPHAREWORK', 'lib_bin')),
                     'GAL_LIB_OUT'    : Property('HDF_bc03', str),      # We do not use PathlikeProperty because this is an output file
                     
                     'MAG_TYPE'       : Property('AB', str,
                                                 testFunc=lambda value: value not in ['AB', 'VEGA'],
                                                 testMsg='MAG_TYPE must either be AB or VEGA.'),
                     
                     'Z_STEP'         : Property([0.01, 2, 0.1], list, subtypes=(int, float), minBound=0,
                                                 testFunc=lambda value: len(value)!=3 or value[2] < value[0], 
                                                 testMsg='Z_STEP property must be a length 3 list where the last step must be larger than the first one.'),
                     
                     'COSMOLOGY'      : Property([70, 0.3, 0.7], list, subtypes=(int, float), minBound=0),
                     
                     'MOD_EXTINC'     : Property([0, 27], list, subtypes=(int, float),
                                                 testFunc=lambda value: len(value)!=2 or value[1] < value[0]), 
                                                 testMsg='MOD_EXTINC property must be an increasing length 2 list.'),
                     
                     'EXTINC_LAW'     : PathlikeProperty('calzetti.dat', path=opath.join('$LEPHAREDIR', 'ext')),
                     
                     'EB_V'           : Property([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], list, subtypes=(int, float),
                                                 testFunc=lambda value: len(value)>49 or any([j<=i for i, j in zip(value[:-1], value[1:])]), 
                                                 testMsg='EB_V property must be an increasing list with a maximum length of 49.'),
                     
                     'EM_LINES'       : Property('NO', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='EM_LINES must either be YES or NO.'),
                     
                     'LIB_ASCII'      : Property('YES', str,
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
                     
                     'EBV_RANGE'      : Property([0, 9], list, subtypes=(int, float)
                                                 testFunc=lambda value: len(value)!=2 or value[1] < value[0],
                                                 testMsg='Z_RANGE property must be an increasing length 2 list.'),
                     
                     'ZFIX'           : Property('YES', str,,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='ZFIX must either be YES or NO.'),
                     
                     'Z_INTERP'       : Property('NO', str,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='Z_INTERP must either be YES or NO.'),
                     
                     'DZ_WIN'         : Property(0.5, (int, float), minBound=0, maxBound=5),
                     'MIN_THRES'      : Property(0.1, (int, float), minBound=0, maxBound=1),
                     'MABS_METHOD'    : Property(1, int, minBound=0, maxBound=4),
                     'MABS_CONTEXT'   : Property(-1, int),
                     'MABS_REF'       : Property(0, int),
                     'MABS_FILT'      : Property([1, 2, 3, 4], list, subtypes=int, minBound=0),
                     
                     'MABS_ZBIN'      : Property([0, 0.5, 1, 1.5, 2, 3, 3.5, 4], list, subtypes=(int, float), minBound=0,
                                                 testFunc=lambda value: len(value)%2!=0 or any([j<=i for i, j in zip(value[:-1], value[1:])]), 
                                                 testMsg='MABS_ZBIN property must be an increasing list with an even length.',
                     
                     'SPEC_OUT'       : Property('NO', str,,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='SPEC_OUT must either be YES or NO.'),
                     
                     'CHI2_OUT'       : Property('NO', str,,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='CHI2_OUT must either be YES or NO.'),
                     
                     'PDZ_OUT'        : PathlikeProperty('NONE'),
                     'PDZ_MABS_FILT'  : Property([2, 10, 14], list, subtypes=int, minBound=0),
                     
                     'FAST_MODE'      : Property('NO', str,,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='FAST_MODE must either be YES or NO.'),
                     
                     'COL_NUM'        : Property(3, int, minBound=0),
                     'COL_SIGMA'      : Property(3, (int, float), minBound=0),
                     
                     'COL_SEL'        : Property('AND', str,
                                                 testFunc=lambda value: value not in ['AND', 'OR'],
                                                 testMsg='COL_SEL must either be YES or NO.'),
                     
                     'AUTO_ADAPT'     : Property('NO', str,,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='AUTO_ADAPT must either be YES or NO.'),
                     
                     'ADAPT_BAND'     : Property([4, 2, 4], list, subtypes=int, minBound=0),
                     
                     'ADAPT_LIM'      : Property([20, 40], list, subtypes=(int, float),
                                                 testFunc=lambda value: len(value)!=2 or value[1] < value[0],
                                                 testMsg='ADAPT_LIM property must be an increasing length 2 list.'),
                     
                     'ADAPT_POLY'     : Property(1, int, minBound=1, maxBound=4),
                     'ADAPT_METH'     : Property(1, int, minBound=1, maxBound=3),
                     'ADAPT_CONTEXT'  : Property(-1, int),
                     
                     'ADAPT_ZBIN'     : Property([0.01, 6], list, subtypes=(int, float), minBound=0,
                                                 testFunc=lambda value: len(value)!=2 or value[1] < value[0],
                                                 testMsg='ADAPT_ZBIN property must be an increasing length 2 list.'),
                     
                     'ADAPT_MODBIN'   : Property([1, 1000], list, subtypes=int, minBound=1, maxBound=1000,
                                                 testFunc=lambda value: len(value)!=2 or value[1] < value[0],
                                                 testMsg='ADAPT_MODBIN property must be an increasing length 2 list.'),
                     
                     'ERROR_ADAPT'    : Property('NO', str,,
                                                 testFunc=lambda value: value not in ['NO', 'YES'],
                                                 testMsg='ERROR_ADAPT must either be YES or NO.')
                     }
    
    def genParams(self, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Generate a parameter file used by the SED fitting code.
        '''
        
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
        QSO_LIB_IN \t{self.prop['QSO_LIB_IN']} \t# Input QSO LIBRARY in $LEPHAREWORK/lib_bin/
        QSO_LIB_OUT	\t{self.prop['QSO_LIB_OUT']} \t# Output QSO MAGN -> $LEPHAREWORK/lib_mag/
        #
        #-------     From GALAXY  LIBRARY  
        GAL_LIB_IN \t{self.prop['GAL_LIB_IN']} \t# Input GAL LIBRARY in $LEPHAREWORK/lib_bin/
        GAL_LIB_OUT	\t{self.prop['GAL_LIB_OUT']} \t# Output GAL LIBRARY -> $LEPHAREWORK/lib_mag/ 
        #
        #-------   MAG + Z_STEP + EXTINCTION + COSMOLOGY
        MAGTYPE \t{self.prop['MAGTYPE']} \t# Magnitude type (AB or VEGA)
        Z_STEP \t{self.prop['Z_STEP']} \t# dz, zmax, dzsup(if zmax>6)
        COSMOLOGY \t{self.prop['COSMOLOGY']} \t# H0,om0,lbd0 (if lb0>0->om0+lbd0=1)
        MOD_EXTINC \t{self.prop['MOD_EXTINC']} \t# model range for extinction 
        EXTINC_LAW \t{self.prop['EXTINC_LAW']} \t# ext. law (in $LEPHAREDIR/ext/*)
        EB_V \t{self.prop['EB_V']} \t# E(B-V) (<50 values)
        EM_LINES \t{self.prop['EM_LINES']}
        # Z_FORM 	8,7,6,5,4,3 	     # Zformation for each SED in GAL_LIB_IN
        #
        #-------   ASCII OUTPUT FILES OPTION
        LIB_ASCII \t{self.prop['LIB_ASCII']} \t# Writes output in ASCII in working directory
        #
        ############################################################################
        #              PHOTOMETRIC REDSHIFTS                                       #
        # $LEPHAREDIR/source/zphot -c $LEPHAREDIR/config/zphot.para                #
        # help: $LEPHAREDIR/source/zphot -h (or -help)                             #
        ############################################################################ 
        #  
        #
        #
        %%INPUTCATALOGUEINFORMATION%%
        CAT_OUT \t{self.id}.out \t
        PARA_OUT \t{self.id}.para \t# Ouput parameter (full path)
        BD_SCALE \t{self.prop['BD_SCALE']} \t# Bands used for scaling (Sum 2^n; n=0->nbd-1, 0[-def]:all bands)
        GLB_CONTEXT \t{self.prop['GLB_CONTEXT']} \t# Overwrite Context (Sum 2^n; n=0->nbd-1, 0 : all bands used, -1[-def]: used context per object)
        # FORB_CONTEXT -1                   # context for forbitten bands
        ERR_SCALE \t{self.para['ERR_SCALE']} \t# errors per band added in quadrature
        ERR_FACTOR \t{self.para['ERR_FACTOR']} \t# error scaling factor 1.0 [-def]       
        #
        #-------    Theoretical libraries
        ZPHOTLIB \t{self.para['ZPHOTLIB']} \t# Library used for Chi2 (max:3)
        ADD_EMLINES \t{self.para['ADD_EMLINES']}
        #
        ########    PHOTOMETRIC REDSHIFTS OPTIONS      ###########
        # FIR LIBRARY
        FIR_LIB \t{self.para['FIR_LIB']}
        FIR_LMIN \t{self.para['FIR_LMIN']} \t# Lambda Min (micron) for FIR analysis 
        FIR_CONT \t{self.para['FIR_CONT']}
        FIR_SCALE \t{self.para['FIR_SCALE']}
        FIR_FREESCALE \t{self.para['FIR_FREESCALE']} \t# ALLOW FOR FREE SCALING 
        FIR_SUBSTELLAR \t{self.para['FIR_SUBSTELLAR']}
        # PHYSICAL LIBRARY with Stochastic models from  BC07        
        PHYS_LIB \t{self.para['PHYS_LIB']}  
        PHYS_CONT \t{self.para['PHYS_CONT']}
        PHYS_SCALE \t{self.para['PHYS_SCALE']}
        PHYS_NMAX \t{self.para['PHYS_NMAX']}
        #
        #-------     Priors  
        # MASS_SCALE	0.,0.		 # Lg(Scaling) min,max [0,0-def]
        MAG_ABS \t{self.para['MAG_ABS']} \t# Mabs_min, Mabs_max [0,0-def]
        MAG_REF \t{self.para['MAG_REF']} \t# Reference number for band used by Mag_abs
        # ZFORM_MIN	5,5,5,5,5,5,3,1	 # Min. Zformation per SED -> Age constraint
        Z_RANGE \t{self.para['Z_RANGE']} \t# Z min-max used for the Galaxy library 
        EBV_RANGE \t{self.para['EBV_RANGE']} \t# E(B-V) MIN-MAX RANGE of E(B-V) used  
        # NZ_PRIOR      4,2,4                # I Band for prior on N(z)
        #                          
        #-------     Fixed Z   (need format LONG for input Cat)
        ZFIX \t{self.para['ZFIX']} \t# fixed z and search best model [YES,NO-def]
        #
        #-------     Parabolic interpolation for Zbest  
        Z_INTERP \t{self.para['Z_INTERP']} \t# redshift interpolation [YES,NO-def]
        #-------  Analysis of normalized ML(exp-(0.5*Chi^2)) curve 
        #-------  Secondary peak analysis       
        DZ_WIN \t{self.para['DZ_WIN']} \t# Window search for 2nd peaks [0->5;0.25-def]
        MIN_THRES \t{self.para['MIN_THRES']} \t# Lower threshold for 2nd peaks[0->1; 0.1-def]
        #
        #-------  Probability (in %) per redshift intervals
        # PROB_INTZ \t{self.para['PROB_INTZ']} \t# even number 
        #
        #########    ABSOLUTE MAGNITUDES COMPUTATION   ###########
        #
        MABS_METHOD	\t{self.para['MABS_METHOD']} \t# 0[-def] : obs->Ref, 1 : best  obs->Ref, 2 : fixed obs->Ref, 3 : mag from best SED, 4 : Zbin
        MABS_CONTEXT \t{self.para['MABS_CONTEXT']} \t# CONTEXT for Band used for MABS 

        MABS_REF \t{self.para['MABS_REF']} \t# 0[-def]: filter obs chosen for Mabs (ONLY USED IF MABS_METHOD=2)
        MABS_FILT \t{self.para['MABS_FILT']} \t# Chosen filters per redshift bin (MABS_ZBIN - ONLY USED IF MABS_METHOD=4)
        MABS_ZBIN \t{self.para['MABS_ZBIN']} \t# Redshift bins (even number - ONLY USED IF MABS_METHOD=4)
        #########   OUTPUT SPECTRA                     ###########
        #
        SPEC_OUT \t{self.para['SPEC_OUT']} \t# spectrum for each object? [YES,NO-def]
        CHI2_OUT \t{self.para['CHI2_OUT']} \t# output file with all values : z,mod,chi2,E(B-V),... BE CAREFUL can take a lot of space !!
        #########  OUTPUT PDZ ANALYSIS  
        PDZ_OUT \t{self.para['PDZ_OUT']} \t# pdz output file name [def-NONE] - add automatically PDZ_OUT[.pdz/.mabsx/.mod/.zph] 
        PDZ_MABS_FILT \t{self.para['PDZ_MABS_FILT']} \t# MABS for REF FILTERS to be extracted
        # 
        #########   FAST MODE : color-space reduction        #####
        #
        FAST_MODE \t{self.para['FAST_MODE']} \t# Fast computation [NO-def] 
        COL_NUM	\t{self.para['COL_NUM']} \t# Number of colors used [3-def]
        COL_SIGMA \t{self.para['COL_SIGMA']} # Enlarge of the obs. color-errors[3-def]
        COL_SEL \t{self.para['COL_SEL']} # Combination between used colors [AND/OR-def]
        #
        #########   MAGNITUDE SHIFTS applied to libraries   ######
        #
        # APPLY_SYSSHIFT  0.             # Apply systematic shifts in each band
                                         # used only if number of shifts matches
                                         # with number of filters in the library    
        #
        #########   ADAPTIVE METHOD using Z spectro sample     ###
        #
        AUTO_ADAPT \t{self.para['AUTO_ADAPT']} \t# Adapting method with spectro [NO-def]
        ADAPT_BAND \t{self.para['ADAPT_BAND']} \t# Reference band, band1, band2 for color 
        ADAPT_LIM \t{self.para['ADAPT_LIM']} \t# Mag limits for spectro in Ref band [18,21.5-def]
        ADAPT_POLY	\t{self.para['ADAPT_POLY']} \t# Number of coef in  polynom (max=4) [1-def]
        ADAPT_METH \t{self.para['ADAPT_METH']} \t# Fit as a function of 1 : Color Model  [1-def], 2 : Redshift, 3 : Models
        ADAPT_CONTEXT \t{self.para['ADAPT_CONTEXT']} \t# Context for bands used for training, -1[-def] used context per object
        ADAPT_ZBIN \t{self.para['ADAPT_ZBIN']} \t# Redshift's interval used for training [0.001,6-Def]
        ADAPT_MODBIN \t{self.para['ADAPT_MODBIN']}'\t# Model's interval used for training [1,1000-Def]
        ERROR_ADAPT \t{self.para['ERROR_ADAPT']} # [YES,NO-def] Add error in quadrature according to the difference between observed and predicted apparent magnitudes 
        #
        '''      
        
        return
    
    def __run__(self, catalogue, tunit='M', magtype='AB', tformat='MEME', ttype='LONG', **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Start the SED fitting on the given data using the built-in SED fitting parameters.
        
        .. note::
            
            The optional parameters are mandatory only if **catalogue** is an astropy Table.
            For a Catalogue instance, the values from the instance are used.
        
        :param catalogue: catalogue to use for the SED-fitting
        :type catalogue: astropy Table or Catalogue instance
        
        :param str tunit: (**Optional**) unit of the table data. Must either be 'M' for magnitude or 'F' for flux.
        :param str magtype: (**Optional**) magnitude type if data are in magnitude unit. Must either be 'AB' or 'VEGA'.
        :param str tformat: (**Optional**) format of the table. Must either be 'MEME' if data and error columns are intertwined or 'MMEE' if columns are first data and then errors.
        :param str ttype: (**Optional**) data type. Must either be SHORT or LONG.

        '''
        
        if isinstance(catalogue, Table):
            cat = Catalogue(catalogue, tunit=tunit, magtype=magtype, tformat=tformat, ttype=ttype)
        elif isinstance(catalogue, Catalogue):
            cat = catalogue
        else:
            raise TypeError(f'catalogue has type {type(catalogue)} but it must either be an astropy Table or a Catalogue instance.')
        
        
        
        return
 

###################################
#          Miscellaneous          #
###################################
        
class ShapeError(Exception):
    r'''Error which is caught when two arrays do not share the same shape.'''
    
    def __init__(self, arr1, arr2, msg='', **kwargs):
        r'''
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
    r'''Define a property object where the data stored must be a valid path or file.'''
    
    def __init__(self, default, path='', **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Init the path-like object. Path-like type must either be str or list of str.
        
        :param default: default value used at init
        
        :param str path: (**Optional**) path to append each time to the new value
        :param **kwargs: (**Optional**) additional parameters passed to the Property constructor
        
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
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Set the current value.
        
        .. note::
            
            If **value** is 'NONE', the path check is not performed.

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
                
                if p.upper() != 'NONE' and not opath.exists(path) and not opath.isfile(path):
                    raise OSError(f'path {path} (expanded as {epath}) does not exist.')
            
        else:
                
            path     = opath.join(self.path, value)
            epath    = opath.expandvars(path)
            if value.upper() != 'NONE' and not opath.exists(path) and not opath.isfile(path):
                raise OSError(f'path {path} (expanded as {epath}) does not exist.')
        
        if not self.testFunc(values):
            raise ValueError(self.testMsg)
        
        self.value   = value
        return
        
class Property:
    r'''Define a property object used by SED objects to store SED parameters.'''
    
    def __init__(self, default, types, subtypes=None, minBound=None, maxBound=None, testFunc=lambda value: True, testMsg='', **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Init the property object.

        :param type: type of the property. If the property is a list, provide the elements type in **subtypes**.
        :param default: default value used at init

        :param subtypes: (**Optional**) type of the elements if **type** is a list. If None, it is ignored.
        :param minBound: (**Optional**) minimum value for the property. If None, it is ignored.
        :param maxBound: (**Optional**) maximum value for the property. If None, it is ignored.
        :param testFunc: (**Optional**) a test function with the value to test as argument which must be passed in order to set a value. This can be used to add additional checks which are not taken into account by default.
        :param testMsg: (**Optional**) a test message used to throw an error if testFunc returns False
        
        :raises TypeError: if **testFunc** is not callable or **testMsg** is not of type str
        '''
        
        if not callable(testFunc) or not isinstance(testMsg, str):
            raise TypeError('test function and test message must be a callable object and of type str respectively.')
        
        self.types     = types
        
        self.substypes = subtypes
        self.min       = minBound
        self.max       = maxBound
        
        self._testFunc = testFunc
        self._testMsg  = testMsg
        
        self.set(default)
        self.default   = default
        
    ##################################
    #        Built-in methods        #
    ##################################
    
    def __str__(self, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Implement a string representation of the class.
        '''
        
        typ = type(self.value)
        
        if typ == int:
            return f'{self.value}'
        elif typ == float:
            return f'{self.value:.3e}'
        elif self.types == str:
            return self.value
        elif typ == list:
            
            joinL  = [f'{i}' if type(i)==int else f'{i:.3e}' if type(i)==float else i if type(i)==str else None for i in self.value]
            
            if None not in joinL:
                return ','.join(joinL)
            else:
                raise NotImplementedError('no string representation available for Property object with value of type {self.subtypes}.')
        else:
             raise NotImplementedError('no string representation available for Property object with value of type {self.types}.')
        
    ###############################
    #        Miscellaneous        #
    ###############################
    
    def set(self, value, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
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
            
        if not self.testFunc(value):
            raise ValueError(self.testMsg)
        
        self.value = value
        return
    
    
    