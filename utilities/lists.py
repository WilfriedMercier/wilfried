#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Fonctions related to manipulating lists.
"""

def transpose(theList):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Compute the transpose of a list without the use of numpy.

    :param list theList: list to transpose

    :returns: transposed list
    :rtype: list
    
    :raises TypeError: if **theList** is not a list
    '''
    
    if type(theList) != list:
        raise TypeError('Data is not a list.')
        
    return list(map(list, zip(*theList)))