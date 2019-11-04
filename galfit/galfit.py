#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:29:00 2019

@author: Wilfried Mercier - IRAP

Functions related to automating galfit modelling.
"""

from wilfried.galfit.models import gendeVaucouleur, genEdgeOnDisk, genExpDisk, genFerrer, genGaussian, genKing, genMoffat, genNuker, genPSF, genSersic, genSky, bendingModes, boxy_diskyness, fourierModes
from wilfried.utilities.dictionaries import checkDictKeys, removeKeys, setDict
from wilfried.utilities.strings import putStringsTogether, toStr, maxStringsLen
from os.path import isdir
from numpy import unique


##########################################
#          Global variables              #
##########################################

# If you change a parameter name in a function declaration, you must also change the name here
fullKeys = {'header': {'parameters':['outputImage', 'xmin', 'xmax', 'ymin', 'ymax', 'inputImage', 'sigmaImage', 'psfImage', 'maskImage', 'couplingFile', 'psfSamplingFactor', 'zeroPointMag', 'arcsecPerPixel', 'sizeConvX', 'sizeConvY', 'displayType', 'option'], 
                       'default':['output.fits', 0, 100, 0, 100, "none", "none", "none", "none", "none", 1, 30.0, [0.03, 0.03], None, None, "regular", 0]},
                                   
            'deVaucouleur': {'parameters':['posX', 'posY', 'magTot', 're', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                             'default':[50, 50, 25.0, 10.0, 1.0, 0.0, False, [], None, False]},
                                         
            'edgeOnDisk': {'parameters':['posX', 'posY', 'mu', 'diskScaleLength', 'diskScaleHeight', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                           'default':[50, 50, 20.0, 10.0, 2.0, 0.0, False, [], None, False]},
                                       
            'expDisk': {'parameters':['posX', 'posY', 'magTot', 'rs', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                        'default':[50, 50, 25.0, 8.0, 1.0, 0.0, False, [], None, False]},
                                    
            'ferrer': {'parameters':['posX', 'posY', 'mu', 'rt', 'alphaFerrer', 'betaFerrer', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                       'default':[50, 50, 20.0, 5.0, 3.0, 2.5, 1.0, 0.0, False, [], None, False]},
                                   
            'gaussian': {'parameters':['posX', 'posY', 'magTot', 'FWHM', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                         'default':[50, 50, 25.0, 3.0, 1.0, 0.0, False, [], None, False]},

            'king': {'parameters':['posX', 'posY', 'mu0', 'rc', 'rt', 'powerlaw', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                     'default':[50, 50, 20.0, 3.0, 30.0, 2.0, 1.0, 0.0, False, [], None, False]},

            'moffat': {'parameters':['posX', 'posY', 'magTot', 'FWHM', 'powerlaw', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                       'default':[50, 50, 25.0, 3.0, 1.0, 1.0, 0.0, False, [], None, False]},

            'nuker': {'parameters':['posX', 'posY', 'mu', 'rb', 'alpha', 'beta', 'gamma', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                      'default':[50, 50, 20.0, 10.0, 1.0, 0.5, 0.7, 1.0, 0.0, False, [], None, False]},

            'psf': {'parameters':['posX', 'posY', 'magTot', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                    'default':[50, 50, 25.0, False, [], None, False]},
                    
            'sersic': {'parameters':['posX', 'posY', 'magTot', 're', 'n', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                       'default':[50, 50, 25.0, 10.0, 4, 1.0, 0.0, False, [], None, False]},
                       
            'sky': {'parameters':['background', 'xGradient', 'yGradient', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                    'default':[1.0, 0.0, 0.0, False, [], None, False]}
            }
            
# Every unique parameter
everyParam = []
for i in [fullKeys[name]['parameters'] for name in fullKeys]:
    everyParam += i
everyParam = unique(everyParam)
            
# Keeping track of the model functions
modelFunctions = {  'deVaucouleur': gendeVaucouleur, 
                    'edgeOnDisk'  : genEdgeOnDisk,     
                    'expDisk'     : genExpDisk,    
                    'ferrer'      : genFerrer,
                    'gaussian'    : genGaussian,
                    'king'        : genKing,
                    'moffat'      : genMoffat,
                    'nuker'       : genNuker,
                    'psf'         : genPSF,        
                    'sersic'      : genSersic,
                    'sky'         : genSky, 
                    'fourier'     : fourierModes,
                    'bending'     : bendingModes,
                    'boxyness'    : boxy_diskyness
                }

# Additional tag functions
tags = ['fourier', 'bending', 'boxyness']


def writeConfigs(header, listProfiles, inputNames, outputNames=[], feedmeNames=[], constraintNames=[], constraints=None,
                 pathFeedme="./feedme/", pathIn="./inputs/", pathOut="./outputs/", pathConstaints="./constraints"):
    """
    Make galfit.feedme files using the same profiles.
    
    Mandatory inputs
    ----------------
        header : dict
            dictionnary with key names corresponding to input parameter names in genHeader function. This is used to generate the header part of the file.
            You do not need to provide an input and an output image file name as this is given with the inputNames keyword.
            
        inputNames : list of str
            list of galaxies .fits files input names in the header
            
        listProfiles : list of dict
            list of dictionaries. Each dictionnary corresponds to a profile:
                - in order for the function to know which profile to use, you must provide a key 'name' whose value is one of the following:
                    'deVaucouleur', 'edgeOnDisk', 'expDisk', 'ferrer', 'gaussian', 'king', 'moffat', 'nuker', 'psf', 'sersic', 'sky'
                - available key names coorespond to the input parameter names. See each profile description, to know which key to use
                - only mandatory inputs can be provided as keys if the default values in the function declarations are okay for you
            
            You can also add fourier modes, bending modes and/or a boxyness-diskyness parameter to each profile. To do so, provide one of the following keys:
                'fourier', 'bending', 'boxyness'
            These keys must contain a dictionnary whose keys are the input parameters names of the functions fourierModes, bendingModes and boxy_diskyness.
            
            Example
            -------
                Say one wants to make a galfit configuration for the galaxies gal1.fits, gal2.fits and galaxy.fits with a:
                    - output image output.fits and a zeroPointMag = 25.4 mag
                    - Sersic profile with a centre position at x=45, y=56, a magnitude of 25 mag and an effective radius of 10 pixels, fixing only n=1, with a PA of 100 (letting other parameters to default values)
                    - Nuker profile with gamma=1.5 and the surface brightness fixed to be 20.1 mag/arcsec^2
                    - bending modes 1 and 3 with values 0.2 and 0.4 respectively added to the Nuker profile
                    
                Then one may write something like
                    >>> header  = {'outputImage':'output.fits', 'zeroPointMag':25.5}
                    >>> sersic  = {'name':'sersic', 'posX':45, 'posY':56, 'magTot':25, 're':10, 'n':1, 'PA':100, 'fixedParams':['n']}
                    
                    >>> bending = {'listModes':[1, 3], 'listModesValues':[0.2, 0.4]}
                    >>> nuker  = {'name':'nuker', 'gamma':1.5, 'mu':20.1, 'bending':bending}
                    >>> galaxies = ["gal1.fits", "gal2.fits", "galaxy.fits"]
                    
                    >>> writeFeedmes(header, [sersic, nuker], galaxies)
                
    Optional inputs
    ---------------
        feedmeNames : list of str
            list of .feedme galfit configuration files. If not provided, the feedme files will have the same name as the input ones but with .feedme extensions at the end.    
        outputNames: list of str
            list of galaxies .fits files output names in the header. If not provided, the output files will have the same name as the input ones with _out appended before the .fits extension.
        pathFeedme : str
            location of the feedme file names relative to the current folder or in absolute
        pathIn : str
            location of the input file names relative to the current folder or in absolute
        pathOut : str
            location of the output file names relative to the current folder or in absolute
            
    Additional input
    ----------------
        This additional parameter can be used to set constraints between components parameters (such as fixing some parameter range).
    
        constraints : dict
            list of dictionaries used to generate the constraints. See below for an explanation on how to use it.
            
            Each dictionary must contain three keys, namely 'components', 'parameter' and 'constraint'
                
                Dictionaries keys
                -----------------
                
                'constraint' : dict with keys 'type' and 'value'
                    set the type of constraint one wants to use and potentially the related range. 
                    The possible 'type' of constraint are:
                        - 'offset' to fix the value of some parameter between different profiles relative to one another (based on the initial values provided in the .feedme file)
                        - 'ratio' to fix the ratio of some parameter between different profiles (based on the initial values provided in the .feedme file)
                        - 'absoluteRange' to set an asbolute range of values for some parameter of a single profile
                        - 'relativeRange' to set a range of possible values around the initial value given in the .feedme file
                        - 'componentWiseRange' to set a range of possible values around the initial value of the same parameter but of another component
                        - 'componentWiseRatio' to set a range of possible values for the ratio of the same parameter in two components
                    
                    And the corresponding values are:
                        - 'offset' for an offset and 'ratio' for a ratio
                        - a list of two float to define bounds for the other types
                
                'components' : int/list of int
                    number (of appearance in the listProfiles) of the galfit models one wants to constrain.:
                        - for 'offset' and 'ratio' constraint types a list of an indefinite number of int can be given. 
                        - for both relative and absolute ranges, a single profile number (int) must be given
                        - for component wise ranges or ratios, a list of two numbers (int) must be provided
                        
                'parameter' : str
                    name of the parameter to contrain
    
    Write full galfit.feedme configuration files for many galaxies.
    """
    
    ################################################
    #          Checking input parameters           #
    ################################################
    
    if feedmeNames == []:
        feedmeNames = [i.replace('.fits', '.feedme') for i in inputNames]
    
    if outputNames == []:
        outputNames = [i.replace('.fits', '_out.fits') for i in inputNames]
        
    if constraintNames == []:
        constraintNames = [i.replace('.fits', '.constraints') for i in inputNames]
    
    if len(inputNames) != len(outputNames) or len(inputNames) != len(feedmeNames) or len(inputNames) != len(constraintNames):
        raise ValueError("Lists intputNames, outputNames, feedmeNames and constraintNames do not have the same length. Please provide lists with similar length in order to know how many feedme files to generate. Cheers !")
    
    if not isdir(pathIn):
        raise OSError("Given path directory %s does not exist or is not a directory. Please provide an existing directory before making the .feedme files. Cheers !" %pathIn)
    if not isdir(pathOut):
        raise OSError("Given path directory %s does not exist or is not a directory. Please provide an existing directory before making the .feedme files. Cheers !" %pathOut)
    if not isdir(pathFeedme):
        raise OSError("Given path directory %s does not exist or is not a directory. Please provide an existing directory before making the .feedme files. Cheers !" %pathFeedme)
    if not isdir(pathConstaints):
        raise OSError("Given path directory %s does not exist or is not a directory. Please provide an existing directory before making the .feedme files. Cheers !" %pathConstaints)
    
    #################################################################
    #                Writing feedme and constraint files            #
    #################################################################
    
    for inp, out, fee, con in zip(inputNames, outputNames, feedmeNames, constraintNames):

        # output .constraint file
        if constraints is not None:
            fname = pathConstaints + con
            with open(fname, "w") as file:
                # Get formatted text
                tmp = genConstraint(constraints)
                file.write(tmp)
                
                # set header key to add the file name into the .feedme file
                header['couplingFile'] = fname
        
        # output .feedme file
        with open(pathFeedme + fee, "w") as file:
            
            # Set input and ouput file names
            header['inputImage']  = pathIn + inp
            header['outputImage'] = pathOut + out
            
            # Get formatted text, check in genFeedme that dict keys are okay and write into file if so
            tmp = genFeedme(header, listProfiles)
            file.write(tmp)


#####################################################################
#              General functions for galfit config files            #
#####################################################################

def genFeedme(header, listProfiles):
    """
    Make a galfit.feedme configuration using the profiles listed in models.py.
    
    Mandatory inputs
    ----------------
        header : dict
            dictionnary with key names corresponding to input parameter names in genHeader function. This is used to generate the header part of the file
        listProfiles : list of dict
            list of dictionaries. Each dictionnary corresponds to a profile:
                - in order for the function to know which profile to use, you must provide a key 'name' whose value is one of the following:
                    'deVaucouleur', 'edgeOnDisk', 'expDisk', 'ferrer', 'gaussian', 'king', 'moffat', 'nuker', 'psf', 'sersic', 'sky'
                - available key names coorespond to the input parameter names. See each profile description, to know which key to use
                - only mandatory inputs can be provided as keys if the default values in the function declarations are okay for you
            
            You can also add fourier modes, bending modes and/or a boxyness-diskyness parameter to each profile. To do so, provide one of the following keys:
                'fourier', 'bending', 'boxyness'
            These keys must contain a dictionnary whose keys are the input parameters names of the functions fourierModes, bendingModes and boxy_diskyness.
            
            Example
            -------
                Say one wants to make a galfit configuration with a:
                    - output image output.fits and a zeroPointMag = 25.4 mag
                    - Sersic profile with a centre position at x=45, y=56, a magnitude of 25 mag and an effective radius of 10 pixels, fixing only n=1, with a PA of 100 (letting other parameters to default values)
                    - Nuker profile with gamma=1.5 and the surface brightness fixed to be 20.1 mag/arcsec^2
                    - bending modes 1 and 3 with values 0.2 and 0.4 respectively added to the Nuker profile
                    
                Then one may write something like
                    >>> header  = {'outputImage':'output.fits', 'zeroPointMag':25.5}
                    >>> sersic  = {'name':'sersic', 'posX':45, 'posY':56, 'magTot':25, 're':10, 'n':1, 'PA':100, 'isFixed':['n']}
                    
                    >>> bending = {'listModes':[1, 3], 'listModesValues':[0.2, 0.4]}
                    >>> nuker  = {'name':'nuker', 'gamma':1.5, 'mu':20.1, 'bending':bending}
                    
                    >>> genFeedme(header, [sersic, nuker])
                
    Return a full galfit.feedme configuration with header and body as formatted text.
    """
    
    header['name'] = 'header'
    correctNames   = fullKeys.keys()
    
    for pos, dic in enumerate([header] + listProfiles):
        # Check that name is okay in dictionaries
        try:
            if dic['name'] not in correctNames:
                raise ValueError("Given name %s is not correct. Please provide a name among the list %s. Cheers !" %correctNames)
        except KeyError:
            raise KeyError("Key 'name' is not provided in one of the dictionaries.")
        
        # Check that the given dictionnary only has valid keys
        checkDictKeys(removeKeys(dic, keys=tags + ['name']), keys=fullKeys[dic['name']]['parameters'], dictName='header or listProfiles')
    
        # Set default values if not provided
        if pos==0:
            header = setDict(dic, keys=fullKeys[dic['name']]['parameters'], default=fullKeys[dic['name']]['default'])
        else:
            listProfiles[pos-1] = setDict(dic, keys=fullKeys[dic['name']]['parameters'], default=fullKeys[dic['name']]['default'])
    
    # Append each profile configuration into a variable of type str
    out      = genHeader(**removeKeys(header, keys=['name']))
    for dic in listProfiles:
        out += "\n\n" + modelFunctions[dic['name']](**removeKeys(dic, keys=['name', 'fourier', 'bending', 'boxyness', 'name']))
        
        # Append tags such as fourier or bending modes if they are provided in the profile
        for t in tags:
            if t in dic:
                out += "\n" + modelFunctions[t](**dic[t])
        
    return out
        
        
def genConstraint(dicts):
    """
    Make a galfit .constraint file.
    
    Mandatory inputs
    -----------------
        dicts : list of dictionaries
            list of dictionaries used to generate the constraints. See below for an explanation on how to use it.
            
            Each dictionary must contain three keys, namely 'components', 'parameter' and 'constraint'
                
                Dictionaries keys
                -----------------
                
                'constraint' : dict with keys 'type' and 'value'
                    set the type of constraint one wants to use and potentially the related range. 
                    The possible 'type' of constraint are:
                        - 'offset' to fix the value of some parameter between different profiles relative to one another (based on the initial values provided in the .feedme file)
                        - 'ratio' to fix the ratio of some parameter between different profiles (based on the initial values provided in the .feedme file)
                        - 'absoluteRange' to set an asbolute range of values for some parameter of a single profile
                        - 'relativeRange' to set a range of possible values around the initial value given in the .feedme file
                        - 'componentWiseRange' to set a range of possible values around the initial value of the same parameter but of another component
                        - 'componentWiseRatio' to set a range of possible values for the ratio of the same parameter in two components
                    
                    And the corresponding values are:
                        - 'offset' for an offset and 'ratio' for a ratio
                        - a list of two float to define bounds for the other types
                
                'components' : int/list of int
                    number (of appearance in the .feedme file) of the galfit models one wants to constrain.:
                        - for 'offset' and 'ratio' constraint types a list of an indefinite number of int can be given. 
                        - for both relative and absolute ranges, a single profile number (int) must be given
                        - for component wise ranges or ratios, a list of two numbers (int) must be provided
                        
                'parameter' : str
                    name of the parameter to contrain
    
    Return a galfit .constraint file as formatted text.
    """

    if type(dicts) is not list:
        raise TypeError("Given parameter 'dicts' is not a list. A list of dictionaries must be provided. Cheers !")
    
    everyConstraint     = ['offset', 'ratio', 'absoluteRange', 'relativeRange', 'componentWiseRange', 'componentWiseRatio']
    
    compList            = []
    paramList           = []
    constraintList      = []
    for d in dicts:
        
        # Retrieve the three information (components, parameter and constraint type)
        try:
            param       = d['parameter']
            constraint  = d['constraint']
            components  = d['components']
        except KeyError:
            raise KeyError("One of the keys 'parameter', 'constraint' or 'components' is missing in one of the dictionaries.")

        
        ############################################################
        #           Check that given parameters are okay           #
        ############################################################
        
        if param not in everyParam:
            raise ValueError('Given parameter %s is not correct. Possible values are %s. Please provide one of these values if you want to put constraints on one of these parameters. Cheers !' %(param, everyParam))

        if constraint['type'] not in everyConstraint:
            raise ValueError('Given constraint type %s is not correct. Possible values are %s. Please provide one of these values if you want to put a constraint on parameters. Cheers !' %(constraint['type'], constraintList))
        
        if constraint['type'] in ['offset', 'ratio', 'componentWiseRange', 'componentWiseRatio']:
            if type(components) is not list:
                raise TypeError("Given components %s should be a list. Please provide a list. Cheers !" %components)
            if len(components)<2:
                raise ValueError("Given components list %s should contain at least two values. Please provide two component numbers at the very least. Cheers !" %components)
            if constraint in ['componentWiseRange', 'componentWiseRatio'] and len(components) != 2:
                raise ValueError("Given components list %s should have exactly two values. Please provide only two component numbers. Cheers !" %components)
        else:
            if type(components) is not int:
                raise ValueError("Given component %s sould be a single integer value corresponding to the component number. Please provide an integer. Cheers !" %components)
        
        if constraint['type'] in ['absoluteRange', 'relativeRange', 'componentWiseRange', 'componentWiseRatio']:
            if type(constraint['value']) is not list:
                raise TypeError("Given constraint values are not a list. Please provide two values in order to have a range. Cheers !")
            if len(constraint['type'])<2:
                raise ValueError("Constraint list does have the correct amount of elements. Two values must be provided. Cheers !")
        
        
        #######################################
        #        Gather columns values        #
        #######################################
        
        # Set components
        if constraint['type'] in ['offset', 'ratio']:
            tmp = ""
            for num in components:
                tmp += "%d_" %num
            compList.append(tmp)
        elif constraint['type'] == 'componentWiseRange':
            compList.append("%d-%d" %tuple(components))
        elif constraint['type'] == 'componentWiseRatio':
            compList.append("%d/%d" %tuple(components))
        else:
            compList.append("%d" %components)
            
        # Set constraint bounds
        if constraint['type'] in ['offset', 'ratio']:
            constraintList.append(constraint['type'])
        elif constraint == 'absoluteRange':
            constraintList.append('%.1f to %.1f' %tuple(constraint['value']))
        else:
            constraintList.append('%.1f %.1f' %tuple(constraint['value']))
            
        # set parameter name
        paramList.append(param)
        
    compList       = ["# Component"] + compList
    paramList      = ["parameter"] + paramList
    constraintList = ["constraint"] + constraintList
        
    lineFormat     = "{:^" + "%d" %maxStringsLen(compList) + "s}   {:" + "%d" %maxStringsLen(paramList) + "s}   {:^" + "%d" %maxStringsLen(constraintList) + "s}"
    out            = ""
    for com, par, con in zip(compList, paramList, constraintList):
        if con in ['offset', 'ratio']:
            com    = com[:-1]
        out       += lineFormat.format(com, par, con) + "\n"
        if com == "# Component":
            out += "\n"
    
    return out


def genHeader(inputImage="none", outputImage='output.fits', sigmaImage="none", psfImage="none", maskImage="none", couplingFile="none", 
              xmin=0, xmax=100, ymin=0, ymax=100, sizeConvX=None, sizeConvY=None,
              psfSamplingFactor=1, zeroPointMag=30.0, arcsecPerPixel=[0.03, 0.03],
              displayType="regular", option=0):
    """
    Constructs a galfit.feedme header as general as possible.
    
    Main inputs
    -----------
        inputImage : str
            name of the input file (fits file only). Note that if "none" is given, when running the feedme file, the fitting will be skipped and a model will be generated using the provided parameters.
        outputImage : str
            name of the file within which the image will be stored. Default is 'output.fits'.
            Technically, the output "image" is not an image but a (1+3)D fits file, where the layer 0 is a blank image whose header contains the fits keys, the layer 1 is the original image within the fitting region, layer 2 is the model image and layer 3 is the residual between the model and the original image. Note that the residual is computed by subtracting the layer 2 from the layer 1. Hence, if any galaxy component is missing in the model image (even if it was optimised), the residual will not reflect the "goodness" of fit.
            
        xmax : int
            maximum x coordinate of the fitting region box (in px). Default is 100 px.
        xmin : int
            minimum x coordinate of the fitting region box (in px). Default is 0 px.
        ymax : int
            maximum y coordinate of the fitting region box (in px). Default is 100 px.
        ymin : int
            minimum y coordinate of the fitting region box (in px). Default is 0 px.
        
    Additional inputs
    -----------------
        arcsecPerPixel : list of two floats
            angular resolution (in arcsec) of image pixels. First element is the angular resolution in the x direction, second element in the y direction.
        displayType : "regular", "both" or "curses"
            galfit display mode. "Regular" mode does not allow any interaction. "Both" and "curses" modes allow you to interact with galfit during the fitting routine (both will display in a xterm terminal the possible commands).
        couplingFile : str
            name of the .constraints file used to add constraints on parameters.
        maskImage : str
            name of the image with bad pixels masked. Either a fits file (with the same dimensions as the input image) with a value of 0 for good pixels and >0 for bad pixels, or an ASCII file with two columns separated by a blank space (first column x coordinate, second y coordinate) listing all the bad pixels locations.
        option : 0, 1, 2 or 3
                - if 0, galfit run normally as explained above
                - if 1, the model image only is made using the given parameters
                - if 2, an image block (data, model and residual) is made using the given parameters
                - if 3, an image block with the first slice beeing the data, and the followings ones beeing one (best-fit) component per slice
       
        psfImage : str
             name of the psf image (fits file).
        psfSamplingFactor : int
            multiplicative factor used to scale the image pixel angular scale to the psf pixel angular scale if the psf is oversampled. Technically it is the ratio between the psf platescale (in arcsec/px) and the data platescale assuming the same seeing.
        sigmaImage : str
            name of the so called variance map (technically standard deviation map) where the value of the standard deviation of the underlying distribution of a pixel is given in place of the pixel value in the image (fits file only). If "none", galfit will compute one.
        sizeConvX : int
            x size of the convolution box (in px)
        sizeConvY : int
            y size of the convolution box (in px)
        zeroPointMag : float
            zero point magnitude used in the definition m = -2.5 \log_{10} (ADU/t_{exp}) + zeropoint where t_{exp} if the exposition time (generally found in the input fits file header).
    
    Returns the header as a formatted string.
    """
    
    psfSamplingFactor, zeroPointMag, option = toStr([psfSamplingFactor, zeroPointMag, option])
    
    #define convolution box to full image and convolution center to image center if not given
    xmaxmin         = xmax-xmin
    ymaxmin         = ymax-ymin
    if sizeConvX is None:
        sizeConvolX = xmaxmin
    if sizeConvY is None:
        sizeConvolY = ymaxmin
    
    #Image region to fit
    imageRegion     = "%d %d %d %d" %(xmin, xmax, ymin, ymax)
    #convolution box X and Y size
    convBox         = "%d %d" %(sizeConvolX, sizeConvolY)
    #angular size per pixel in X and Y directions
    plateScale      = "%d %d" %tuple(arcsecPerPixel)
        
    #compute max text length to align columns
    textLenMax      = maxStringsLen([outputImage, inputImage, sigmaImage, psfImage, psfSamplingFactor, maskImage, couplingFile, imageRegion, convBox, zeroPointMag, plateScale, displayType, option])
    formatText      = "%-" + "%d"%textLenMax + "s"
    
    #generate items text 
    itemA = ("A) " + formatText)%inputImage + "   # Input data image (FITS file)"
    itemB = ("B) " + formatText)%outputImage + "   # Name for the output image"
    itemC = ("C) " + formatText)%sigmaImage + '   # Noise image name (made from data if blank or "none")'
    itemD = ("D) " + formatText)%psfImage + "   # Input PSF image for convolutFITS file)"
    itemE = ("E) " + formatText)%psfSamplingFactor + "   # PSF Fine-Sampling Factor"
    itemF = ("F) " + formatText)%maskImage + "   # Pixel mask (ASCII file or FITS file with non-0 values)"
    itemG = ("G) " + formatText)%couplingFile + "   # File with parameter ion (constraints (ASCII file)"
    itemH = ("H) " + formatText)%imageRegion + "   # Image region to fit (xmin xmax ymin ymax)"
    itemI = ("I) " + formatText)%convBox + "   # Size of convolution box (x y)"
    itemJ = ("J) " + formatText)%zeroPointMag + "   # Magnitude photometric zeropoint"
    itemK = ("K) " + formatText)%plateScale + "   # Plate scale (dx dy). Relevant only for Nuker/King models."
    itemO = ("O) " + formatText)%displayType + "   # Display type (regular, curses, both)"
    itemP = ("P) " + formatText)%option + "   # Options: 0=normal run; 1,2=make model/imgblock & quit; 3=normal run and separate components"
    
    textLenMax      = maxStringsLen([itemA, itemB, itemC, itemD, itemE, itemF, itemG, itemH, itemI, itemJ, itemK, itemO, itemP])
    
    out = putStringsTogether(["="*textLenMax, "# IMAGE PARAMETERS",
                           itemA, itemB, itemC, itemD, itemE, itemF, itemG, itemH, itemI, itemJ, itemK, itemO, itemP,
                           "="*textLenMax])
    return out