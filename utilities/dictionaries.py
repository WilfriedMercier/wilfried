#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:26:30 2019

@author: wilfried

Functions acting on dictionnaries.
"""

import numpy         as     np
from   astropy.table import Table, vstack

def checkDictKeys(dictionnary, keys=[], dictName='NOT PROVIDED'):
    """
    Check that every dictionnary key is among the keys list 'keys'.
    
    Mandatory inputs
    ----------------
        dictionnary : dict
            dictionnary from which we want to test if given keys are in 'keys' list
        
    Optional inputs
    ---------------
        dictName : str
            dictionnary variable name printed in the error message
        keys : list of str
            list of authorised keys names 
            
    Return None if 'dictionnary' has only keys in 'keys', or KeyError if one of the keys is not in 'keys' list.
    """
    
    for k in dictionnary.keys():
        if k not in keys:
            raise KeyError("key '%s' in dictionnary %s is not a valid key among %s" %(k, dictName, keys))
    return None
    

def checkInDict(dictionnary, keys=[], dictName='NOT PROVIDED'):
    """
    Check that given keys exist in the dictionnary.
    
    Mandatory inputs
    ----------------
        dictionnary : dict
            dictionnary from which we want to test if given keys already exist
        
    Optional inputs
    ---------------
        dictName : str
            dictionnary variable name printed in the error message
        keys : list of str
            keys names we want to test if they exist in 'dictionnary'
            
    Return None if keys exist, or KeyError if one of the keys was missing.
    """
    
    dictkeys = dictionnary.keys()
    for name in keys:
        if name not in dictkeys:
            raise KeyError("at least one of the mandatory keys in dictionnary '%s' was not provided. Please provide at least the mandatory keys. Cheers !" %dictName)
    return None


def concatenateDictValues(myDicts, astropyTable=True):
    '''
    IMPORTANT : NOT WORKING PROPERLY !
    
    Take a dictionnary containing either numpy arrays or numpy structured arrays as values and concatenate them.
    
    Notes:
    ------
        - If working with structured arrays/astropy tables, be careful to have the same column names/fields, otherwise the concatenation shall still produce a result which may be different from what would be expected.
        - When combining a standard array with a masked one, a new mask (with False values everywhere) will be concatenated to the previous one so that the masks will be preserved.
    
    Mandatory inputs
    ----------------
        myDicts : dict or list of dicts
            either a single input dictionnary containing numpy arrays or a list of dictionnaries (numpy arrays as values as well)
            
    Optional inputs
    ---------------
        astropyTable : bool
            whether to return an astropy table or not. If False, it will return a numpy masked array.
            
    Return either a single (masked) numpy array/astropy Table where each numpy array in the dictionnary has been concatenated to the others, or a list of (masked) numpy arrays/astropy tables.
    '''
    
    def doItForOneDict(myDict):
        '''Just do the concatenation for a single dictionnary.'''
        
        for pos, value in enumerate(myDict.values()):
            # If we have an array of int/floats instead of an array of tuples (as for structured arrays), astropy will not be able to correctly transform it into an astropy table.
            # Thus we must embed it into a list
            try:
                iter(value[0])
            except TypeError:
                print('coucou')
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
            raise TypeError('One of the elements of the %d%s dictionnary is not a numpy array (or a derived class). Please only provide numpy arrays/astropy tables within the given dictionnaries. Cheers !' %(pos+1, text))
        return
    

    if isinstance(myDicts, list):
        
        # Check whether the list is empty or not
        if len(myDicts) == 0:
            raise ValueError('The given list is empty. Please provide a non-empty list of dictionnaries. Cheers !')
        
        # Check whether each element in the list is a dictionnary
        for aDict in myDicts:
            if not isinstance(aDict, dict):
                raise TypeError('One of the element in the input list is not a dictionnary. Either provide a single dictionnary or a list of dictionnaries as input. Cheers !')
                
        # If we deal with dictionnaries, check that each element value is either a list or a numpy array
        for aDict in myDicts:
            for pos, element in enumerate(aDict.values()):
                checkanArray(element, pos)
                
        # If it went fine, concatenate for every dictionnary
        return [doItForOneDict(i) for i in myDicts]
    
    elif isinstance(myDicts, dict):
        for pos, element in enumerate(aDict.values()):
            checkanArray(element, pos)
            
        return doItForOneDict(myDicts)
    else:
        raise TypeError('Given input was neiter a dictionnary, nor a list of dictionnaries. Please provide one of those types. Cheers !')


def removeKeys(dictionnary, keys=[]):
    """
    Creates a new dictionnary with given keys removed.
    
    Mandatory inputs
    ----------------
        dictionnary : dict
            dictionnary from which the keys are removed    
        
    Optional inputs
    ---------------
        keys : list of str
            list of keys to remove from the dictionnary
            
    Returns a new dictionnary with given keys removed.        
    """
    
    dicCopy = dictionnary.copy()
    for k in keys:
        try:
            dicCopy.pop(k)
        except KeyError:
            pass
    return dicCopy
    

def setDict(dictionnary, keys=[], default=[]):
    """
    Create a dictionnary whose desired keys are either retrieved from a dictionnary, or set to default values otherwise.
    
    Mandatory inputs
    ----------------
        dictionnary : dict
            dictionnary from which the values in 'keys' list are retrieved
            
    Optional inputs
    ---------------
        default : list
            list of default values for keys listed in 'keys' if no value is given in 'dictionnary'
        keys : list of str
            list of key names which should be retrieved either from 'dictionnary' or from 'default'
    
    Return a new dictionnary with keys listed in 'keys' having values either from 'dictionnary' or 'default'.
    """
    
    for k, df in zip(keys, default):
        dictionnary.setdefault(k, df)
    return dictionnary