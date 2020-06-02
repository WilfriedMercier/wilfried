#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:34:54 2020

@author: wilfried

Functions related to input-output interaction.
"""

from   astropy.table          import Table
from   astropy.io.votable     import is_votable, writeto
import numpy.lib.recfunctions as     rec
import numpy                  as     np

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


def move_bad_fields_to_bottom(array, orderedFieldList, orderedTypeList):
    """
    Move a list of fields from a structured array to the bottom and change their type.
    
    Mandatory inputs
    ----------------                                                
        array : numpy structured array
            array to modify               
        orderedFieldList : list
            list of field names to move and to change type 
        orderedTypeList : list
            list of new types for the fields (same order as orderedFieldList)
                           
    Return an array with some fields moved to the bottom and with different types.
    """
    
    outArray = array.copy()
    for name, typ in zip(orderedFieldList, orderedTypeList):
        
        #Remove field of interest from the array
        tmpArray = rec.rec_drop_fields(outArray, name)
        
        #Append the same field at the end of the array with the right data type
        outArray = rec.rec_append_fields(tmpArray, name, array[name].copy(), dtypes=typ)
    return outArray


def write_array_to_vot(array, outputFile, isTable=False):
    """
    Write an array or an astropy table into a VOtable file.
    
    Mandatory inputs
    ----------------
        array : numpy array, astropy table
            The array to write into the file
        outputFile : str
            The file to write the array into
        
    Optional inputs
    ---------------
        isTable : boolean
            Whether the array is an astropy table or not. Default is False.
    """
    
    if not isTable:
        array = Table(data=array)
        
    writeto(array, outputFile)
    return



################################################################################################
#                          MUSE catalogues specific functions                                  #
################################################################################################


def add_new_array_to_previous(newArray, fullFileName, fields, oldArray=None, isFirstArray=False, fieldsToDrop=None, typesToDrop=None):
    """
    Append a new structured array to another one, only keeping some fields after applying their corresponding data types onto the new columns.
    This function is used to combine data from different MUSE catalogues where field names and data types may vary between different versions.
    
    Mandatory inputs
    ----------------
        fields : list of str
            list containing the field names
        fullfilename : str
            filename of the new array to append to the previous one
        newArray : numpy structured array
            new array to append to the previous one
            
    Optional inputs
    ---------------
        fieldsToDrop : list of string
            the name of the fields to move to the bottom and change their type. If not None, typesToDrop must be a list of the same size.
        isFirstArray : bool
            whether this is the first array one makes with this function or not. If True a check will be made at the end.
        oldArray : numpy structured array
            previous array whereto append new data
        typesToDrop : list of data types
            data types corresponding to the specified fields which must be dropped to the bottom
              
    Returns a new structured array where all the content of the previous ones has been correctly appended          
    """
    
    #Try to keep all the required fields (common to every catalogue if they all had the same name)
    try:
        array = newArray[fields].copy()
    #Dealing with exceptions because of variations in field names between catalogues
    except ValueError:
        if "CGR34-32_FD_zcatalog_withLaigle+16_withFAST_withnewPLATEFIT_totalflux_nov18_withFOF_withGALFIT_withGALKIN_jan19.vot" in fullFileName:            
            newArray = rec.rename_fields(newArray, {'groupe_secure_z':'group_secure_z', 'groupe_unsecure_z':'group_unsecure_z'})
            
        elif ("CGR79-77_FD_zcatalog_withLaigle+16_withFAST_withnewPLATEFIT_totalflux_nov18_withFOF_withGALFIT_withGALKIN_jan19.vot" in fullFileName or 
              "CGR32-32-M123_FD_zcatalog_withLaigle+16_withFAST_withnewPLATEFIT_totalflux_withnewz_jan19_withFOF_withGALFIT_withGALKIN_jan19_COSMOSGroupNumberOldCorrected.vot" in fullFileName):
            newArray = rec.rename_fields(newArray, {'TYPE_2':'TYPE', 'secure_z_ss':'secure_z', 'unsecure_z_ss':'unsecure_z', 'no_z_ss':'no_z', 
                                                    'group_secure_z_ss':'group_secure_z', 'group_unsecure_z_ss':'group_unsecure_z'})
    
        elif "CGR32-32-M123_FD_zcatalog_withLaigle+16_withFAST_withnewPLATEFIT_totalflux_withnewz_jan19_withFOF_withGALFIT_withGALKIN_jan19_COSMOSGroupNumberOldCorrected.vot" in fullFileName:
            newArray = rec.rename_fields(newArray, {'TYPE_2':'TYPE'})
            
        elif "CGR114_116_zcatalog_withLaigle+16_withFAST_withPLATEFIT_weightedflux_oct18_withFOF_withGALFIT_withGALKIN_jan19.vot" in fullFileName:
            newArray = rec.rename_fields(newArray, {'TYPE_2':'TYPE', 'COSMOS_Group_number':'COSMOS_Group_Number',
                                                    'COSMOS_Group_number__old_':'COSMOS_Group_Number__old_', 'FLAG_COSMOS_1':'FLAG_COSMOS'})
    
        if "CGR30-28_FD_zcatalog_withLaigle+16_withFAST_withnewPLATEFIT_totalflux_nov18_withFOF_withGALFIT_withGALKIN_jan19.vot" in fullFileName:
            newArray = rec.rename_fields(newArray, {'TYPE_2':'TYPE', 'ID_Laigle_16_or_ORIGIN':'ID_Laigle_16'})

        array = newArray[fields].copy()
        
    #Moving to the bottom the fields of interest and changing their type according to those specified
    if fieldsToDrop is not None and typesToDrop is not None:
        if len(fieldsToDrop)==len(typesToDrop):
            array = move_bad_fields_to_bottom(array, fieldsToDrop, typesToDrop)
        else:
            print('The field names list and the type list should have the same length in order to drop them to the bottom of the table.')
    else:
        print('No field and/or no type list was given, thus no field was dropped to the bottom of the table.')

    #Checking that field management went fine
    if not isFirstArray:
        typeOld = oldArray.dtype
        typeNew = array.dtype
        sz      = len(typeOld)
        if sz != len(typeNew):
            raise Exception("ERROR: old and new arrays do not have the same number of fields.")
        
        for i in range(sz):
            if typeOld[i] != typeNew[i]:
                print('The following type change was applied %s -> %s' %(typeOld.names[i], typeNew.names[i]))
    
    if oldArray is not None:
        outArray = np.append(oldArray, array)
    else:
        outArray = array
    
    return outArray


