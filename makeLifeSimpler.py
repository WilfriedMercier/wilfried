#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:32:44 2019

@author: wilfried

A set of useful functions to make life simpler when analysing data.
"""

import numpy        as np
import numpy.random as rand


def inferError(function, parameters, err_params, minBounds=None, maxBounds=None, num=1000, centralValue=None, **kwargs):
    '''
    Infer the 1 sigma error on the output of a function by estimating its standard deviation when perturbating its parameters according to their errors.
    Errors are assumed to be Gaussian.

    Mandatory parameters
    --------------------
        function : func
            function to infer the output error
        parameters : list
            parameters to pass to the function. Order must be identical as in the function declaration.
        err_params : list
            1 sigma errors of the parameters. If the parameter is fixed or perfectly known, provide None instead to bypass the pertubation part.
            
    Optional parameters
    -------------------
        centralValue : int/float
            value used to compute the standard deviation (usually median or mean). If no value is provided, the median value of the perturbed iterations is used.
        num : int
            number of iterations to perform
        maxBounds : list
            maximum limit allowed for the parameters when perturbating them. All perturbed parameters beyond these thresholds will be set to these values.
        minBounds ; list
            minimum limit allowed for the parameters when perturbating them. All perturbed parameters beyond these thresholds will be set to these values.
        **kwargs : dict
            additional parameters to pass onto the function at the end

    Return the 1 sigma error of the function around the median value.
    '''
    
    if not isinstance(num, int):
        raise TypeError('Number of iterations must be an integer.')
        
    if not isinstance(parameters, list):
        raise TypeError('Parameters must be given as a list.')
    
    if not isinstance(err_params, list):
        raise TypeError('Parameters errors must be given as a list.')
        
    if minBounds is None:
        minBounds = [None]*len(parameters)
    if maxBounds is None:
        maxBounds = [None]*len(parameters)

    if not isinstance(minBounds, list):
        raise TypeError('Minimum bounds should be given as a list.')

    if not isinstance(maxBounds, list):
        raise TypeError('Maximum bounds should be given as a list.')
        
    # Generating the perturbations for each parameter
    perturbations = []
    for value, error, mini, maxi in zip(parameters, err_params, minBounds, maxBounds):
        
        # Without error, we had the parameter value without perturbations
        if error is None:
            perturbations.append([value]*num)
        else:
            param = rand.normal(loc=value, scale=error, size=num)
            
            # If there is a min or max bound, we clip outliers
            if mini is not None:
                param[param<mini] = mini
            if maxi is not None:
                param[param>maxi] = maxi
                
            perturbations.append(param)
    perturbations = transpose(perturbations)

    # Apply the function to each element
    result        = np.asarray([function(*i, **kwargs) for i in perturbations])
    
    if centralValue is None:
        centralValue = np.nanmedian(result)
        
    if not isinstance(centralValue, (int, float)):
        raise TypeError('Central value should either be an int or a float.')
        
    return np.nanstd(result-centralValue)
    
    

def transpose(lists):
    '''
    Take the transpose of a list of lists/tuples.
    All the inner lists are assumed to be of equal length, otherwise the transpose will not work correctly.

    Parameters
    ----------
        lists : list of lists/tuples
            list of all the lists or tuples to perform the transpose onto

    Return the transposed list.
    '''
    
    if not isinstance(lists, list):
        raise TypeError('lists parameter should be of type list only. Cheers !')
    
    return list(zip(*lists))


def convertCoords(coordinates, inSize=(200.0, 200.0), outSize=(31.0, 31.0), conversionFactor=1.0):
    '''
    Transforms the coordinates of a/many point(s) from one image to another
    
    Input
    -----
    coordinates : dictionnary or list of dictionnaries
        the coordinates of the points to convert form one image to another
    conversionFactor : float
        a numerical factor to convert the position from pixel to another relavant unit
    inSize : tuple/list
        the size of the image the points are from
    outSize : tuple/list
        the size of the image whereto we want to convert the positions of the points
        
    Returns a list of dictionnaries with transformed coordinates.
    '''
    
    try:
        np.shape(coordinates)[0]
    except:
        coordinates = [coordinates]
        
    for num, points in enumerate(coordinates):
        for pos, key in enumerate(points.keys()):
            coordinates[num][key] *= outSize[pos]/inSize[pos]*conversionFactor
    return coordinates



def printSimpleStat(catalog, unit=None):
    """
    Print basic stats such as median and mean values, as well as 1st and 3rd quantiles.
    
    Input
    -----
    catalog : array/astropy table/list or list of arrays/astropy tables/lists
        array from which the statistic is computed
    unit: astropy unit
        unit of the array if there is one
    """

    try:
        np.shape(catalog[1])
    except IndexError:
        catalog = [catalog]
    
    for cat, num in zip(catalog, range(len(catalog))):
        if unit is not None:
            cat = cat*unit
            
        print("Stat for catalog number", num, ":")
        print("Maximum separation is", str(np.max(cat)) + ".")
        print("Mean separation is", str(np.mean(cat)) + ".")
        print("Median separation is", str(np.median(cat)) + ".")
        print("1st quantile is", str(np.quantile(cat, 0.25)) + ".")
        print("3rd quantile is", str(np.quantile(cat, 0.75)) + ".")
        
    return   
   

def uniqueArr(tables, arraysToBeUnique):
    """
    Apply a mask from np.unique on arraysToBeUnique for many arrays.
    
    Input
    -----
    tables : table/array or list of tables/arrays
        tables to which the mask is applied
    arraysToBeUnique : table/array or list of tables/arrays
        tables or arrays from which the mask is computed (with np.unique)
        
    Returns tables with the mask applied.
    """
    
    #Transform into a list if it is an array
    try:
        np.shape(tables[1])
    except IndexError:
        tables = [tables]
    try:
        np.shape(arraysToBeUnique[1])
    except IndexError:
        arraysToBeUnique = [arraysToBeUnique]
        
    for num, uniq in zip(range(len(tables)), arraysToBeUnique):    
        arr, indices = np.unique(uniq, return_index=True)
        tables[num]  = tables[num][indices]
        
    return tables


def maskToRemoveVal(listOfArrays, val=None, keep=True, astroTableMask=False):
    """
    Computes a mask by finding occurences in a list of arrays.
    
    Input
    -----
    listOfArrays : list of numpy arrays
        the list of arrays from which the mask is built
    val : float or None
        the value to find. If val=None, it looks for nan values.
    keep : boolean
        if True, it builds a mask with True everywhere the value val is encountered. If False, it does the opposite
    astroTableMask : boolean
        if True returns a mask from the astropy table column instead of looking for some value/nans with False values everywhere the data is masked
    
    Returns a mask as a numpy array.
    """
    
    shp = listOfArrays[0].shape
    #Checking that arrays have the same shape
    for array in listOfArrays[1:]:
        if shp != array.shape:
            exit("Arrays do not have the same dimensions, thus making the masking operation unfit. Exiting.")
  
    #Constructing first mask
    if astroTableMask:
        tmp = np.logical_not(listOfArrays[0].mask)
    elif val is None:
        tmp = np.logical_not(np.isnan(listOfArrays[0]))
    else:
        tmp = listOfArrays[0] == val
        if not keep:
            tmp = np.logical_not(tmp)
        
    #Applying logical and on all the masks
    for (num, array) in enumerate(listOfArrays[1:]):
        #consider we are looking for nan in the arrays
        if astroTableMask:
            tmp = np.logical_and(tmp, np.logical_not(array.mask))  
        elif val is None:
            tmp = np.logical_and(tmp, np.logical_not(np.isnan(array)))
        else:
            if keep:
                tmp = np.logical_and(tmp, array==val)
            else:
                tmp = np.logical_and(tmp, array != val)
    return tmp


def logicalAndFromList(lst):
    """
    Compute the intersection of all the subarrays in the main array
    
    Input
    -----
    lst : list of numpy arrays
        a list of arrays containing True of False values
        
    Returns np.logical_and applied on all the subarrays
    """
    
    
    tmp = np.logical_and(lst[0], lst[1])
    for i in range(2, len(lst)):
        tmp = np.logical_and(tmp, lst[i])
    return tmp


def applyMask(listOfArrays, mask):
    """
    Apply the same mask to a list of arrays and return the new arrays.
    
    Input
    -----
    listOfArrays : list of numpy arrays
        the list of arrays the mask is applied to
    mask : numpy array
        the mask to apply
        
    Returns the list of arrays with the mask applied. If len(listOfArrays) is 1, it returns only an array instead of a list of arrays with one object.
    """

    for (num, array) in enumerate(listOfArrays):
        if len(listOfArrays) == 1:
            listOfArrays = array[mask]
        else:
            listOfArrays[num] = array[mask]
    return listOfArrays


def findWhereIsValue(listOfArrays, val=None):
    """
    Find and print the first position where a value is found within a list of arrays.
    
    Input
    -----
    listOfArrays : list of numpy arrays
        list from which the value val is searched
    val : float or None
        value to look for. If val=None, it looks for nan values.
        
    Returns a list of booleans with the same length as listOfArrays, with True when the value was found in the array and False otherwise.
    """
    
    returnArr = []
    
    for (num, array) in enumerate(listOfArrays):
        if val is None:
            if np.any(np.isnan(array)):
                returnArr.append(True)
                print("A nan was found at position", np.where(np.isnan(array))[0], "within array number", num)
            else:
                returnArr.append(False)
                print("No nan was found in array number", num)
        else:
            if np.asarray(np.where(array==val)).shape[1] == 0:
                returnArr.append(False)
                print("No value", val, "found within array number", num)
            else:
                returnArr.append(True)
                print("Value", val, "found at position", np.where((array==val))[0], "within array number", num)
    return returnArr
                

def checkDupplicates(master, names=None):
    """
    Check if galaxies are found multiple times in an array by looking for duplicates of (RA, DEC) pairs.
    
    Input
    -----
    master : list of structured numpy arrays (with 'RA' and 'DEC' fields)
        a list of structured arrays to check
    names : list of strings
        the names of the arrays
    """
    
    if (names is None) or (len(names) != len(master)):
        try:
            len(names) != len(master)
            print("Given names were not enough. Using position in the list as name instead.")
        except TypeError:
            pass
        names = np.char.array(['catalog nb ']*len(master)) + np.char.array(np.array(range(len(master)), dtype='str'))
    
    for catalog, nameCat in zip(master, names):
        cnt = True
        for ra, dec, nb in zip(catalog['RA'], catalog['DEC'], range(catalog['RA'].shape[0])):
            
            where1 = np.where(catalog['RA']==ra)[0]
            where2 = np.where(catalog['DEC']==dec)[0]
            
            if (len(where1)>1) and (len(where2)>1):
                
                flag = True
                for w in where2:
                    
                    if flag and (w in where1):
                        print("RA =", ra, "deg and DEC =", dec, "deg galaxy (line " + str(nb) + ") is present more than once in catalog", nameCat)
                        flag = False
                        cnt  = False
        if cnt:
            print("All the galaxies are only listed once in the catalog", nameCat)     
    return