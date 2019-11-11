#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:32:44 2019

@author: wilfried

A set of useful functions to make life simpler when analysing data.
"""

#astropy imports
from astropy.table import Table
from astropy.io.votable import is_votable, writeto

#numpy imports
import numpy as np
import numpy.lib.recfunctions as rec


################################################################################################
#                                   VOtable functions                                          #
################################################################################################

def is_VOtable(fullname):
    """
    Check whether a file is a VOtable.
    
    Mandatory inputs
    ----------------
    fullname : str
        path+name of the file to test
    
    Returns True if it is a VOtable. False otherwise.
    """
    tag = is_votable(fullname)
    print("The file", fullname, "is a VOtable, right ?", tag)
    return tag

def write_array_to_vot(array, outputFile, isTable=False):
    """
    Writes an array or an astropy table into a .vot file.
    
    Mandatory inputs
    ----------------
    array : numpy array, astropy table
        The array to write into the file
    outputFile : str
        The file to write the array into
        
    Optional inputs
    ---------------
    isTable : boolean
        Whether the array is an astropy table or not.
    """
    
    #If it is an array it creates an astropy table
    if not isTable:
        array = Table(data=array)
        
    writeto(array, outputFile)
    return

def move_bad_fields_to_bottom(oldArray, orderedFieldList, orderedTypeList):
    """
    Move the given fields in a structured array to the bottom and change their type
    
    Input
    -----                                                           
    oldArray : numpy structured array
        previous array to modify               
    orderFieldList : list
        list of fields to move and change type 
    orderedTypeList : list
        list of new types for the fields
                           
    Returns an array with some fields moved to the bottom and with a different type
    """
    
    outArray = oldArray.copy()
    for name, typ in zip(orderedFieldList, orderedTypeList):
        #Remove field of interest from the array
        tmpArray = rec.rec_drop_fields(outArray, name)
        
        #Append the same field at the end of the array with the right data type
        outArray = rec.rec_append_fields(tmpArray, name, oldArray[name].copy(), dtypes=typ)
    return outArray

def add_new_array_to_previous(oldArray, newArray, fullFileName, fields, firstArray=False, fieldsToDrop=None, typesToDrop=None):
    """
    Append a new structured array from a catalog to another one, only keep the given fields and apply their corresponding data types onto the new columns
    
    Mandatory input
    -----
    fields : list of strings
        list containing the fields names as they should appear in every catalogue if they all had the same column names (it is never the case)
    fieldsToDrop : list of string
        the name of the fields to move to the bottom and change their type. If not None, typesToDrop must be a list of the same size.
    firstArray : 
        True if first array to build
    fullfilename : string
        filename (relative to the current directory) of the new array to append to the previous one
    newArray : numpy structured array
        new array to append to the previous one
    oldArray : numpy structured array
        previous array whereto append new data
    typesToDrop : list of data types
        data types corresponding to the specified fields which must be dropped
          
    Returns a new structured array where all the content of the previous ones has been correctly appended          
    """
    
    print(fullFileName)
    
    #Try to keep all the required fields (common to every catalogue if they all had the same name)
    try:
        array = newArray[fields].copy()
    #Dealing with exceptions because of variations in fields names between catalogues
    except ValueError:
        if "CGR34-32_FD_zcatalog_withLaigle+16_withFAST_withnewPLATEFIT_totalflux_nov18_withFOF_withGALFIT_withGALKIN_jan19.vot" in fullFileName:            
            newArray = rec.rename_fields(newArray, {'groupe_secure_z':'group_secure_z', 
                                           'groupe_unsecure_z':'group_unsecure_z'})
        if ("CGR79-77_FD_zcatalog_withLaigle+16_withFAST_withnewPLATEFIT_totalflux_nov18_withFOF_withGALFIT_withGALKIN_jan19.vot" in fullFileName or 
            "CGR32-32-M123_FD_zcatalog_withLaigle+16_withFAST_withnewPLATEFIT_totalflux_withnewz_jan19_withFOF_withGALFIT_withGALKIN_jan19_COSMOSGroupNumberOldCorrected.vot" in fullFileName):
            newArray = rec.rename_fields(newArray, {'TYPE_2':'TYPE', 'secure_z_ss':'secure_z', 
                                      'unsecure_z_ss':'unsecure_z', 'no_z_ss':'no_z', 
                                      'group_secure_z_ss':'group_secure_z', 
                                      'group_unsecure_z_ss':'group_unsecure_z'})
        if "CGR32-32-M123_FD_zcatalog_withLaigle+16_withFAST_withnewPLATEFIT_totalflux_withnewz_jan19_withFOF_withGALFIT_withGALKIN_jan19_COSMOSGroupNumberOldCorrected.vot" in fullFileName:
            newArray = rec.rename_fields(newArray, {'TYPE_2':'TYPE'})
            #print(sorted(list(newArray.dtype.names)))
        if "CGR114_116_zcatalog_withLaigle+16_withFAST_withPLATEFIT_weightedflux_oct18_withFOF_withGALFIT_withGALKIN_jan19.vot" in fullFileName:
            newArray = rec.rename_fields(newArray, {'TYPE_2':'TYPE', 'COSMOS_Group_number':'COSMOS_Group_Number',
                                      'COSMOS_Group_number__old_':'COSMOS_Group_Number__old_',
                                      'FLAG_COSMOS_1':'FLAG_COSMOS'})
        if "CGR30-28_FD_zcatalog_withLaigle+16_withFAST_withnewPLATEFIT_totalflux_nov18_withFOF_withGALFIT_withGALKIN_jan19.vot" in fullFileName:
            newArray = rec.rename_fields(newArray, {'TYPE_2':'TYPE', 'ID_Laigle_16_or_ORIGIN':'ID_Laigle_16'})

        array = newArray[fields].copy()
        
    #Moving to the bottom the fields of interest and changing their type accordingly to those specified
    if fieldsToDrop is not None and typesToDrop is not None and len(fieldsToDrop)==len(typesToDrop):
        array = move_bad_fields_to_bottom(array, fieldsToDrop, typesToDrop)

    #Checking that field management went fine
    if not firstArray:
        typeOld = oldArray.dtype
        typeNew = array.dtype
        sz      = len(typeOld)
        if sz != len(typeNew):
            print("ERROR: old and new arrays do not have the same number of fields. Exiting.")
            return None
        
        for i in range(sz):
            if typeOld[i] != typeNew[i]:
                print(typeOld.names[i], typeNew.names[i])
            
        outArray = np.append(oldArray, array)
    else:
        outArray = array
    
    return outArray

def linear_fit(x, A, offset):
    """
    Compute a linear relation A*x+offset.
    
    Input
    -----
    x : numpy array
        input data
    A : float
        Slope coefficient
    offset : float
        x=0 Y-coordinate
        
    Returns a numpy array A*x+offset.
    """
    return A*x+offset

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

def computeGroupFWHM(wavelength, groups, verbose=True, model='Moffat'):
    '''
    Computes the FWHM at a given observed wavelength assuming a linearly decreasing relation for the FWHM with wavelength (calibrated on OII and OIII measurements at different redshifts) stars measurements for each group in the COSMOS field.
    
    Input
    -----
    groups : string or list of strings
        the group for each desired wavelength
    model : string
        the model to use, either Moffat or Gaussian
    verbose : boolean
        whether to print a message on screen with the computed FWHM or not
    wavelength : integer
        the wavelength(s) at which we want to compute the FWHM (must be in Angstroms)
    
    Returns a list of tuples with the group and the computed FWHM.
    '''
    
    #structure is as folows : number of the group, o2 FWHM, o3hb FWHM, mean redshift of the group
    if model == 'Moffat':
        listGroups = {'23' : [3.97, 3.29, 0.850458], '26' : [3.16, 2.9, 0.439973], '28' : [3.18, 3.13, 0.950289],
                      '32-M1' : [2.46, 1.9, 0.753319], '32-M2' : [2.52, 2.31, 0.753319], '32-M3' : [2.625, 2.465, 0.753319],
                      '51' : [3.425, 2.95, 0.386245], '61' : [3.2, 3.02, 0.364009], '79' : [2.895, 2.285, 0.780482], 
                      '84-N' : [2.49, 2.21, 0.727755], '30_d' : [2.995, 2.68, 0.809828], '30_bs' : [2.745, 2.45, 0.809828],
                      '84' : [2.835, 2.715, 0.731648], '34_d' : [2.89, 2.695, 0.857549], '34_bs' : [np.nan, np.nan, 0.85754],
                      '114' : [3.115, 2.81, 0.598849]}
    elif model == "Gaussian":
        listGroups = {'23' : [4.28, 3.65, 0.850458], '26' : [3.68, 3.34, 0.439973], '28' : [3.62, 3.26, 0.950289],
                      '32-M1' : [2.975,	2.58,  0.753319], '32-M2' : [3.16,	2.54, 0.753319], '32-M3' : [3.61,	3.3, 0.753319],
                      '51' : [3.75, 3.28, 0.386245], '61' : [3.915,	3.34, 0.364009], '79' : [3.29,	2.695, 0.780482],
                      '84-N' : [2.89,	2.58, 0.727755], '30_d' : [3.485,	3.11, 0.809828], '30_bs' : [3.185,	2.815, 0.809828],
                      '84' : [3.24,	3.055, 0.731648], '34_d' : [3.31,	2.995, 0.857549], '34_bs' : [3.3,	3.003, 0.85754],
                      '114' : [3.705,	3.315, 0.598849]}
    else:
        raise Exception("Model %s not recognised. Available values are %s" %(model, ["Moffat", "Gaussian"]))
    
    #lines wavelengths in Anstrom
    OIIlambda   = 3729 
    OIIIlambda  = 5007
    deltaLambda = OIIIlambda - OIIlambda
    
    try:
        np.shape(wavelength)[0]
    except:
        wavelength = [wavelength]
    try:
        np.shape(groups)[0]
    except:
        groups = [groups]
        
    #check wavelength and groups have the same size
    if len(wavelength) != len(groups):
        exit("Wavelength and group lists do not have the same length. Please provide exactly one group for each wavelength you want to compute.")
    
    #checking given group names exist
    for pos, name in enumerate(groups):
        name        = str(name)
        groups[pos] = name
        
        try:
            listGroups[name]
        except KeyError:
            exit("Given group %s is not correct. Possible values are %s" %(name, listGroups.keys()))
            
    outputList = []
    for wv, gr in zip(wavelength, groups):
        #lines wavelength are rest-frame wavelengths, but FWHM measurements were made at a certain redshift
        #A factor of (1+z) must be applied to deltaLambda and OII lambda
        grVals = listGroups[gr]
        slope  = (grVals[1] - grVals[0])/(deltaLambda*(1+grVals[2]))
        offset = grVals[0] - slope*OIIlambda*(1+grVals[2])
        
        FWHM = slope*wv+offset
        outputList.append((gr, FWHM))
        
        if verbose:
            print("FWHM at wavelength", wv, "angstroms in group", gr, "is", FWHM)
            
    return outputList
        
    
    
    

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



        
        



