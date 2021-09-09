    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Hugo Plombat - LUPM & Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Utilties related to generating 2D mass and SFR maps using either LePhare or CIGALE SED fitting codes.
"""

import yaml
import numpy                     as     np
import os.path                   as     opath
from   .symlinks.coloredMessages import *
from   copy                      import deepcopy

def _parseConfig(file):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
    Read a config file and return config parameters as dictionary.

    :param str file: YAML configuration file
    
    :returns: configuration dictionary
    :rtype: dict
    
    :raises IOError: if **file** does not exist
    '''
    
    if opath.isfile(file):
        out = yaml.load(file, Loader=yaml.Loader)
        
        if not isinstance(out, dict):
            raise TypeError(f'File {file} could not be cast into dict type.')
        
    else:
        raise IOError(f'File {file} not found.')
            
    return out


class SEDobject:
    '''Base class implementing the object used to stored SED fitting into.'''
    
    def __init__(self, filters=[], files=[], errfiles=[], path='./', texpFactor=1, normalise=False, extData=None, extErr=None, **kwargs):
        '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
            
        Initialise SED objects.

        :param list[str] filters: (**Optional**) filters used to perform the SED fitting
        :param list[str] files: (**Optional**) files corresponding to each filter (order is assumed similar)
        :param list[str] errfiles: (**Optional**) error files corresponding to each filter (order is assumed similar)
        :param str path: (**Optional**) common path appended before each file name
        :param texpFactor: (**Optional**) factor used to correct the exposition time when correcting the error maps
        :type texpFactor: int or float
        :param bool normalise: (**Optional**) whether to use the normalisation scheme or not
        
        :param list[int] extData: (**Optional**) list of FITS file extensions for each data file. If None, 0th extension is used by default for each file.
        :param list[int] extErr: (**Optional**) list of FITS file extensions for each error file. If None, 0th extension is used by default for each file.
        
        :raises TypeError: 
            * if **normalise** is not a bool
            * if **filters**, **files** and **errfiles** are not all lists
            * if **texpFactor** is neither an int nor a float
        '''
        
        if not isinstance(i, normalise):
            raise TypeError(f'normalise parameter has type {type(normalise)} but it must be of type bool.')
            
        if not isinstance(texpFactor, (int, float)):
            raise TypeError(f'texpFactor parameter has type {type(texpFactor)} but it must be of type int or float.')
            
        if not all([isinstance(i, list) for i in [filters, files, errfiles]]):
            raise TypeError(f'filters, files and errfiles parameters have types {type(filters)}, {type(files)} and {type(errfiles)} but they must all be of type list.')
            
        self.texpFac  = texpFactor
        self.norm     = normalise
        
        # Init files lists with empty values
        self.filters  = []
        self.files    = []
        self.errFiles = []
        
        # Init data lists with empty values
        self.hdrList  = [] # data header for each filter
        self.dataList = [] # data for each filter
        
        self.ehdrList = [] # error map header for each filter
        self.errList  = [] # error map for each filter
        
        ll            = len(filters)
        if extData is None:
            extData   = [0] * ll
        elif not isinstance(extData, list):
            extData   = [extData] * ll
            
        if extErr is None:
            extErr    = [0] * ll
        elif not isinstance(extErr, list):
            extErr    = [extErr] * ll
        
        # Insert filter if data can be correctly loaded
        for filt, file, err, extD, extE in zip(filters, files, errFiles, extData, extErr):
            file      = opath.join(path, file)
            err       = opath.join(path, err)
            self.insertFilter(filt, file, err, extension=exetD, errExtension=extE)
            
        # Define a mask which hides pixels (default no pixel is hidden)
        self.mask     = False
        
        
    ##############################
    #       Public methods       #
    ##############################
        
    def insertFilter(self, filt, file, errFile, extension=0, errExtension=0, **kwargs):
        '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Insert a filter into the database after loading data and error files.
        
        :param str filt: filter name (must not already be in the filter list)
        :param str file: data file name corresponding to the filter
        :param str errFile: error file correspond to the data file
        
        :param int extension: (**Optional**) data fits file extension to load data from
        :param int errExtension: (**Optional**) error map fits file extension to load error map from
        
        :raises TypeError: if **filt**, **file** and **errFile** are not str, or if **extension** is not an int
        '''
        
        if not isinstance(filt, str):
            raise TypeError(f'filt parameter has type {type(filt)} but it must have type str.')
        
        # Check that filter does not already exist
        if filt in self.filters:
            print(warningMessage('Warning: ') + 'filter ' +  brightMessage(filt) +  ' already present in filter list.')
            return
        
        # Load data and header from FITS file (file and extension checked here)
        hdr, data = self._loadFits(file, ext=extension)
        
        # Load error and header from FITS file (file and extension checked here)
        ehdr, err = self._loadFits(errFile, ext=errExtension)
        
        if None in [data, err, hdr, ehdr]:
            print(errorMessage('Skipping filter...'))
        else:                    
            self.filters.append( filt)
            self.files.append(   file)
            self.errFiles.append(errFile)
            
            self.dataList.append(data)
            self.hdrList.append( hdr)
            
            self.errList.append( err)
            self.ehdrList.append(ehdr)
        
        return
    
    def meanMaps(self, maskVal=0, **kwargs):
        '''
        Compute the averaged data and error maps over the spectral dimension for non masked pixels.
        
        :param maskVal: (**Optional**) value to put into masked pixels
        :type maskVal: int or float
        
        :returns: averaged data and averaged error map
        :rtype: ndarray, ndarray
        
        :raises TypeError: if **maskVal** is neither in nor float
        '''
        
        if not isinstance(maskVal, (int, float)):
            raise TypeError(f'maskVal parameter has type {type(maskVal)} but it must have type int or float.')
            
        # Compute masked arrays
        data  = self._applyMask(self.dataList)
        err   = self._applyMask(self.errList)
        
        # Compute mean value along spectral dimension
        data  = np.nanmean(data, axis=0)
        err   = np.nanmean(err,  axis=0)
        
        # Replace NaN values
        np.nan_to_num(data, copy=False, nan=maskVal)
        np.nan_to_num(err,  copy=False, nan=maskVal)
        
        return data, err
        
        
    ###############################
    #       Private methods       #
    ###############################
    
    def _applyMask(self, larray, *args, **kwargs):
        '''
        Apply the mask by placing NaN values onto each array in the list.
        
        :param list[ndarary] larray: list of arrays to mask
        
        :returns: list with masked arrays
        :rtype: list[ndarray]
        '''
        
        data                    = []
        
        # We make a deep copy to be sure we do not modify the input arrays
        for arr  in larray:
            data.append(deepcopy(arr))
            data[-1][self.mask] = np.nan
        
        return data
        
    def _checkFile(self, file):
        '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
            
        Check whether a file exists.
        
        :param str file: file name
        
        :returns: 
            * True if the file exists
            * False otherwise
        '''
        
        if not isinstance(file, str):
            raise TypeError(f'file has type {type(file)} but it must have type str.')
        
        if not opath.isfile(fname):
            print(errorMessage('Error: ') + 'file ' + brightMessage(fname) + ' not found.')
            return False
        return True
        
    @staticmethod
    def _loadFits(file, ext=0):
        '''
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
                    
            # If an error is triggered, we always return None, None
            except OSError:
                print(errorMessage('Error: ') + ' file ' + brightMessage(file) + ' could not be loaded as a FITS file.')
            except IndexError:
                print(errorMessage('Error: ') + f' extension number {ext} too large.')
            else:
                return hdu.header, hdu.data
                
        return None, None
            
        
        
        
        
        
        
        