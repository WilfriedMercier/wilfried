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
