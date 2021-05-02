#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

String related functions.
"""

################################################################################################
#                                   String manipulations                                       #
################################################################################################


def computeStringsLen(listStrings):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    For each string in the list, compute its length and return them all.
    
    :param list[str] listStrings: list of strings
        
    :returns: list with the length for each string
    :rtype: list
    """
    
    return [len(i) for i in listStrings]


def maxStringsLen(listStrings):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    For each string in the list, compute its length and return the maximum value.
    
    :param list[str] listStrings: list of strings
        
    :returns: the length of the longest string
    :rtype: int
    """
    
    return max(computeStringsLen(listStrings))


def putStringsTogether(listStrings):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Combine strings from a list with a newline character (\\n) between each string (except after the final one).
    
    :param list[str] listStrings: list of strings to combine
        
    :returns: the combined strings as formatted text
    :rtype: str
    """
    
    string = ""
    for i in listStrings[:-1]:
        string += i + "\n"
        
    return string+listStrings[-1]


def toStr(listStuff):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Transform a list of values into a list of strings.

    :param list listStuff: list whose elements will be transformed into strings
            
    :returns: the input list as a list of strings
    :rtype: list[str]
    """
    
    return [str(i) for i in listStuff]