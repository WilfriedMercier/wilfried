#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:40:55 2019

@author: Wilfried Mercier - IRAP

String related functions.
"""

################################################################################################
#                                   String manipulations                                       #
################################################################################################


def computeStringsLen(listStrings):
    """
    For each string in the list, compute its length and return them all.
    
    Mandatory inputs
    ------
    listStrings : list of str
        list of strings whose length is computed
        
    Return a list with the length for each string in the input list.
    """
    
    return [len(i) for i in listStrings]


def maxStringsLen(listStrings):
    """
    For each string in the list, compute its length and return the maximum value.
    
    Mandatory inputs
    ------
    listStrings : list of str
        list of strings
        
    Return the length of the longest strings.
    """
    
    return max(computeStringsLen(listStrings))


def putStringsTogether(listStrings):
    """
    Combine strings from a list with a newline character (\n) between each string (except after the final one).
    
    Mandatory inputs
    ------
    listStrings : list of str
        list of strings to combine
        
    Return the combined strings as formatted text.
    """
    string = ""
    for i in listStrings[:-1]:
        string += i + "\n"
    return string+listStrings[-1]


def toStr(listStuff):
    """
    Transform a list of values into a list of strings.
    
    Mandatory inputs
    ----------------
        listStuff : list
            input lits whose elements will be transformed into strings
            
    Return the input list as a list of strings.
    """
    
    return [str(i) for i in listStuff]