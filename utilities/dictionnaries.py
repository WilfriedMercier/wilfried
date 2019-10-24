#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:26:30 2019

@author: wilfried

Functions acting on dictionnaries.
"""

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
        dicCopy.pop(k)
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