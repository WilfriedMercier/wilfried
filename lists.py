#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:05:54 2020

@author: wilfried
"""

def transpose(theList):
    '''
    Compute the transpose of a list with pure python.

    Parameters
    ----------
        theList : list
            list to transpose

    Return the transposed list.
    '''
    
    if type(theList) != list:
        raise ValueError('Data is not a list.')
        
    return list(map(list, zip(*theList)))