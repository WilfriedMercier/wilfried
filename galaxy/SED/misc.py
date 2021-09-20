#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 17:12:07 2021

@author: wilfried
"""

import os.path as opath
      
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
        
class Property:
    r'''Define a property object used by SED objects to store SED parameters.'''
    
    def __init__(self, default, types, subtypes=None, minBound=None, maxBound=None, testFunc=lambda value: False, testMsg='', **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Init the property object.

        :param type: type of the property. If the property is a list, provide the elements type in **subtypes**.
        :param default: default value used at init

        :param subtypes: (**Optional**) type of the elements if **type** is a list. If None, it is ignored.
        :param minBound: (**Optional**) minimum value for the property. If None, it is ignored.
        :param maxBound: (**Optional**) maximum value for the property. If None, it is ignored.
        :param testFunc: (**Optional**) a test function with the value to test as argument which must not be passed in order to set a value. This can be used to add additional checks which are not taken into account by default.
        :param testMsg: (**Optional**) a test message used to throw an error if testFunc returns False
        
        :raises TypeError: if **testFunc** is not callable or **testMsg** is not of type str
        '''
        
        if not callable(testFunc) or not isinstance(testMsg, str):
            raise TypeError('test function and test message must be a callable object and of type str respectively.')
        
        self.types     = types
        
        self.subtypes  = subtypes
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
        
        if isinstance(self.value, int):
            return f'{self.value}'
        elif isinstance(self.value, float):
            if self.value < 1e-3 or self.value > 1e3:
                return f'{self.value:.3e}'
            else:
                return f'{self.value:.3f}'
        elif isinstance(self.value, str):
            return self.value
        elif isinstance(self.value, list):
            
            joinL  = []
            for i in self.value:
                if isinstance(i, int):
                    joinL.append(f'{i}')
                elif isinstance(i, float):
                    if i < 1e-3 or i > 1e3:
                        joinL.append(f'{i:.3e}')
                    else:
                        joinL.append(f'{i:.3f}')
                elif isinstance(i, str):
                    joinL.append(i)
                else:
                    raise NotImplementedError(f'no string representation available for Property object with value {i} of type {type(i)}.')
            
            return ','.join(joinL)
            
        else:
             raise NotImplementedError(f'no string representation available for Property object with value of type {type(self.value)}.')
        
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
            
            * if one of the values in **value** is below the minimum bound
            * if one of the values in **value** is above the maximum bound
        '''
        
        if not isinstance(value, self.types):
            raise TypeError(f'cannot set property with value {value} of type {type(value)}. Acceptable types are {self.types}.')
            
        
        if isinstance(value, list):
            
            if any ((not isinstance(i, self.subtypes) for i in value)):
                raise TypeError(f'at least one value does not have the type {self.subtypes}.')
        
            if self.min is not None and any((elem < self.min for elem in value)):
                raise ValueError(f'one of the values is less than the minimum acceptable bound {self.min}')
                
            if self.max is not None and any((elem > self.max for elem in value)):
                raise ValueError(f'one of the values is more than the maximum acceptable bound {self.max}')
        else:
            
            if self.min is not None and value < self.min:
                raise ValueError('value is {value} but minimum acceptable bound is {self.min}.')
        
            if self.max is not None and value > self.max:
                raise ValueError('value is {value} but maximum acceptable bound is {self.max}.')
            
        if self._testFunc(value):
            raise ValueError(self._testMsg)
        
        self.value = value
        return

class PathlikeProperty(Property):
    r'''Define a property object where the data stored must be a valid path or file.'''
    
    def __init__(self, default, path='', ext='', **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Init the path-like object. Path-like type must either be str or list of str.
        
        :param default: default value used at init
        
        :param str path: (**Optional**) path to append each time to the new value
        :param str ext: (**Optionall**) extension to append at the end of the file name when checking the path
        :param **kwargs: (**Optional**) additional parameters passed to the Property constructor
        
        :raises TypeError: if **path** and **ext** are not of type str
        
        .. seealso:: :py:class:`Property`
        '''
        
        if not isinstance(path, str):
            raise TypeError(f'path has type {type(path)} but it must have type str.')
            
        if not isinstance(ext, str):
            raise TypeError(f'extension has type {type(ext)} but it must have type str.')
        
        # Extension to append when checking path
        self.ext     = ext
        
        # Set path to append at the beginning of a check
        self.path    = path
        
        super().__init__(default, (list, str), subtypes=str, minBound=None, maxBound=None, **kwargs)
        
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
            
        if isinstance(value, list):
            
            if any((not isinstance(i, self.subtypes) for i in value)):
                raise TypeError(f'at least one value does not have the type {self.subtypes}.')

            for p in value:    
                path  = opath.join(self.path, p) + self.ext
                epath = opath.expandvars(path)
                
                if p.upper() != 'NONE' and not opath.exists(epath) and not opath.isfile(epath):
                    raise OSError(f'path {path} (expanded as {epath}) does not exist.')
            
        else:
                
            path     = opath.join(self.path, value) + self.ext
            epath    = opath.expandvars(path)
            if value.upper() != 'NONE' and not opath.exists(epath) and not opath.isfile(epath):
                raise OSError(f'path {path} (expanded as {epath}) does not exist.')
        
        if self._testFunc(value):
            raise ValueError(self._testMsg)
        
        self.value   = value
        return
        