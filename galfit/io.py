#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 11:12:24 2020

@author: wilfried

General functions to format io output using ANSII sequences via colorama
"""

import colorama as col

def errorMessage(text):
    '''Error message as a red bright text'''
    
    return col.Fore.RED + col.Style.BRIGHT + text + col.Style.RESET_ALL

def okMessage(text):
    '''Validation message as a dim green text'''
    
    return col.Fore.GREEN + col.Style.DIM + text + col.Style.RESET_ALL

def brightMessage(text):
    '''Emphasize a text by brightening it'''
    
    return col.Style.BRIGHT + text + col.Style.RESET_ALL

def dimMessage(text):
    '''Dim a text'''
    
    return col.Style.DIM + text + col.Style.RESET_ALL
    