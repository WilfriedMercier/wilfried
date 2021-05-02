#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Functions acting on dictionaries.
"""

import numpy         as     np
from   astropy.table import Table, vstack

def checkDictKeys(dictionary, keys=[], dictName='NOT PROVIDED'):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Check that all the dictionary keys are among the list **keys**.

    :param dict dictionary: dictionary from which we want to test if given keys are in **keys** list

    :param str dictName: (**Optional**) dictionary variable name printed in the error message
    :param list[str] keys: (**Optional**) list of authorised key names 
            
    :returns: None if **dictionary** keys are all in **keys** list 
    
    :raises KeyError: if one of the keys is not in **keys** list
    """
    
    for k in dictionary.keys():
        if k not in keys:
            raise KeyError("key '%s' in dictionary %s is not a valid key among %s" %(k, dictName, keys))
            
    return None
    

def checkInDict(dictionary, keys=[], dictName='NOT PROVIDED'):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Check that the given keys exist in the dictionary.

    :param dict dictionary: dictionary from which we want to test if given keys already exist
    :param str dictName: (**Optional**) dictionary variable name printed in the error message
    :param list[str] keys: (**Optional**) key names we want to test if they exist in **dictionary**
            
    :returns: None if keys exist
    
    :raises KeyError: if one of the keys is missing in the dictionary
    """
    
    dictkeys = dictionary.keys()
    for name in keys:
        if name not in dictkeys:
            raise KeyError("at least one of the mandatory keys in dictionary '%s' was not provided. Please provide at least the mandatory keys. Cheers !" %dictName)
    return None


def concatenateDictValues(myDicts, astropyTable=True):
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Take a dictionary containing either numpy arrays or numpy structured arrays as values and concatenate them.
    
    .. warning::
        
        NOT WORKING PROPERLY !
    
    .. note::
        
        - If working with structured arrays/astropy tables, be careful to have the same column names/fields, otherwise the concatenation shall still produce a result which may be different from what would be expected.
        - When combining a standard array with a masked one, a new mask (with False values everywhere) will be concatenated to the previous one so that the masks will be preserved.

    :param myDicts: either a single dictionary containing numpy arrays or a list of dictionnaries (numpy arrays as values as well)
    :type myDicts: dict or list of dicts
    :param bool astropyTable: (**Optional**) whether to return an astropy table or not. If False, it will return a numpy masked array.
            
    :returns: table where each numpy array in the dictionary has been concatenated to the others, or a list of tables
    :rtype: masked ndarray or astropy Table of list[masked ndarray] or list[astropy Table]
    
    :raises ValueError: if **myDicts** is an empty list
    :raises TypeError:
        
        * if at least one element in **myDicts** is not a dict
        * if **myDicts** is not a list
    '''
    
    def doItForOneDict(myDict):
        '''Just do the concatenation for a single dictionary.'''
        
        for pos, value in enumerate(myDict.values()):
            # If we have an array of int/floats instead of an array of tuples (as for structured arrays), astropy will not be able to correctly transform it into an astropy table.
            # Thus we must embed it into a list
            try:
                iter(value[0])
            except TypeError:
                value = [value]
                
            if pos == 0:
                myArray = Table(value)
            else:
                myArray = vstack([myArray, Table(value)])
                    
        if not astropyTable:
            myArray         = myArray.as_array()
            
        return myArray
        
    def checkanArray(element, pos):
        ''''Check whether the given element is a a numpy array'''
        
        if not isinstance(element, np.ndarray):
            if str(pos+1)[-1] == '1':
                text = 'st'
            elif str(pos+1)[-1] == '2':
                text = 'nd'
            else:
                text = 'th'
            raise TypeError('One of the elements of the %d%s dictionary is not a numpy array (or a derived class). Please only provide numpy arrays/astropy tables within the given dictionnaries. Cheers !' %(pos+1, text))
        return
    

    if isinstance(myDicts, list):
        
        # Check whether the list is empty or not
        if len(myDicts) == 0:
            raise ValueError('The given list is empty. Please provide a non-empty list of dictionnaries. Cheers !')
        
        # Check whether each element in the list is a dictionary
        for aDict in myDicts:
            if not isinstance(aDict, dict):
                raise TypeError('One of the element in the input list is not a dictionary. Either provide a single dictionary or a list of dictionnaries as input. Cheers !')
                
        # If we deal with dictionnaries, check that each element value is either a list or a numpy array
        for aDict in myDicts:
            for pos, element in enumerate(aDict.values()):
                checkanArray(element, pos)
                
        # If it went fine, concatenate for every dictionary
        return [doItForOneDict(i) for i in myDicts]
    
    elif isinstance(myDicts, dict):
        for pos, element in enumerate(aDict.values()):
            checkanArray(element, pos)
            
        return doItForOneDict(myDicts)
    else:
        raise TypeError('Given input was neiter a dictionary, nor a list of dictionnaries. Please provide one of those types. Cheers !')


def removeKeys(dictionary, keys=[]):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Creates a new dictionary with given keys removed.

    :param dict dictionary: dictionary from which the keys are removed    
    :param list[str] keys: (**Optional**) keys to remove from the dictionary
            
    :returns: new dictionary with given keys removed
    :rtype: dict
    """
    
    dicCopy = dictionary.copy()
    for k in keys:
        try:
            dicCopy.pop(k)
        except KeyError:
            pass
        
    return dicCopy
    

def setDict(dictionary, keys=[], default=[]):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Create a new dictionary whose keys are either retrieved from a dictionary, or set to default values if not present in the dictionary.

    :param dict dictionary: dictionary from which the values in **keys** list are retrieved
        
    :param list default: (**Optional**) list of default values for keys listed in **keys** if no value is found in **dictionary**
    :param list[str] keys: (**Optional**) list of key names which should be retrieved either from **dictionary** or from **default**
    
    :returns: new dictionary with keys listed in **keys** having values either from **dictionary** or from **default**
    :rtype: dict
    """
    
    dicCopy = dictionary.copy()
    for k, df in zip(keys, default):
        dicCopy.setdefault(k, df)
        
    return dicCopy