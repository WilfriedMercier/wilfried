#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:26:30 2019

@author: wilfried

Functions acting on dictionnaries.
"""

import numpy as np

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


def concatenateDictValues(myDicts):
    '''
    Take a dictionnary containing lists/numpy array as values and concatenate them.
    
    Inputs
    ------
        myDicts : dict or list of dicts
            either a single input dictionnary containing lists or numpy arrays or a list of dictionnaries (with lists/numpy arrays as values as well)
            
    Return either a single numpy array where each list/numpy array in the dictionnary has been concatenated to the others, or a list of numpy arrays where each element contains the concatenation of the lists/numpy arrays in the corresponding dictionnary.
    '''
    
    def doItForOneDict(myDict):
        '''Just do the concatenation for a single dictionnary.'''
        
        for pos, key in enumerate(myDict.keys()):
            if len(myDict) == 1:
                myArray     = myDict[key]
            else:
                if pos == 0:
                    myArray = np.array(myDict[key])
                else:
                    myArray = np.concatenate([myArray, np.array(myDict[key])])
        return myArray
        
    def checkaList(element, pos):
        ''''Check whether the given element is a list or a numpy array'''
        
        if not isinstance(element, list) and not isinstance(element, np.ndarray):
            if str(pos+1)[-1] == '1':
                text = 'st'
            elif str(pos+1)[-1] == '2':
                text = 'nd'
            else:
                text = 'th'
            raise TypeError('One of the elements of the %d%s dictionnary is neither a list nor a numpy array. Please only provide lists or numpy arrays within the given dictionnaries. Cheers !' %(pos+1, text))

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
                checkaList(element, pos)
                
        # If it went fine, concatenate for every dictionnary
        return [doItForOneDict(i) for i in myDicts]
    
    elif isinstance(myDicts, dict):
        for pos, element in enumerate(aDict.values()):
            checkaList(element, pos)
            
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