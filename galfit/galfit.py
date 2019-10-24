#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:29:00 2019

@author: Wilfried Mercier - IRAP

Functions related to automating galfit modelling.
"""

from wilfried.galfit.models import gendeVaucouleur, genEdgeOnDisk, genExpDisk, genFerrer, genGaussian, genKing, genMoffat, genNuker, genPSF, genSersic, genSky
from wilfried.utilities.dictionnaries import checkDictKeys, checkInDict, removeKeys, setDict
from wilfried.utilities.strings import putStringsTogether, toStr, maxStringsLen


##########################################
#          Global variables              #
##########################################

# If you change a variable name in a function declaration, you must also change the name here
fullKeys = {'header': {'mandatory':['outputImage', 'xmin', 'xmax', 'ymin', 'ymax'], 
                       'optional':[['inputImage', 'sigmaImage', 'psfImage', 'maskImage', 'couplingFile', 'psfSamplingFactor', 'zeroPointMag', 'arcsecPerPixel', 'sizeConvX', 'sizeConvY', 'displayType', 'option'],
                                   ["none", "none", "none", "none", "none", 1, 30.0, [0.03, 0.03], None, None, "regular", 0]]},
                                   
            'deVaucouleur': {'mandatory':['posX', 'posY', 'magTot', 're'], 
                             'optional':[['bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                                         [1.0, 0.0, False, [], None, False]]},
                                         
            'edgeOnDisk': {'mandatory':['posX', 'posY', 'mu', 'diskScaleLength', 'diskScaleHeight'], 
                           'optional':[['PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                                       [0.0, False, [], None, False]]},
                                       
            'expDisk': {'mandatory':['posX', 'posY', 'magTot', 'rs'], 
                        'optional':[['bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                                    [1.0, 0.0, False, [], None, False]]},
                                    
            'ferrer': {'mandatory':['posX', 'posY', 'mu', 'rt'], 
                       'optional':[['alphaFerrer', 'betaFerrer', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                                   [3.0, 2.5, 1.0, 0.0, False, [], None, False]]},
                                   
            'gaussian': {'mandatory':['posX', 'posY', 'magTot', 'FWHM'], 
                         'optional':[['bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                                    [1.0, 0.0, False, [], None, False]]},

            'king': {'mandatory':['posX', 'posY', 'mu0', 'rc', 'rt'], 
                     'optional':[['powerlaw', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                                [2.0, 1.0, 0.0, False, [], None, False]]},

            'moffat': {'mandatory':['posX', 'posY', 'magTot', 'FWHM'], 
                       'optional':[['powerlaw', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                                  [1.0, 1.0, 0.0, False, [], None, False]]},

            'nuker': {'mandatory':['posX', 'posY', 'mu', 'rb'], 
                      'optional':[['alpha', 'beta', 'gamma', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                                 [1.0, 0.5, 0.7, 1.0, 0.0, False, [], None, False]]},

            'psf': {'mandatory':['posX', 'posY', 'magTot'], 
                    'optional':[['skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                               [False, [], None, False]]},
                    
            'sersic': {'mandatory':['posX', 'posY', 'magTot', 're'], 
                       'optional':[['n', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                                  [4, 1.0, 0.0, False, [], None, False]]},
                       
            'sky': {'mandatory':['background', 'xGradient', 'yGradient'], 
                    'optional':[['skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                               [False, [], None, False]]}
            }
            
# Keeping track of the model functions
modelFunctions = {  'deVaucouleur': gendeVaucouleur, 
                    'edgeOnDisk': genEdgeOnDisk,     
                    'expDisk': genExpDisk,    
                    'ferrer': genFerrer,
                    'gaussian': genGaussian,
                    'king': genKing,
                    'moffat': genMoffat,
                    'nuker': genNuker,
                    'psf': genPSF,        
                    'sersic': genSersic,
                    'sky': genSky}


#####################################################################
#              General functions for galfit feedme files            #
#####################################################################

def genFeedme(header, listProfiles):
    """
    Make a galfit.feedme configuration using the profiles listed in models.py.
    
    Mandatory inputs
    ----------------
        header : dict
            dictionnary with key names corresponding to input parameter names in genHeader function. This is used to generate the header part of the file
        listProfiles : list of dict
            list of dictionnaries. Each dictionnary corresponds to a profile:
                - in order for the function to know which profile to use, you must provide a key 'name' whose value is one of the following:
                    'deVaucouleur', 'edgeOnDisk', 'expDisk', 'ferrer', 'gaussian', 'king', 'moffat', 'nuker', 'psf', 'sersic', 'sky'
                - key names are input parameter names. See each profile description, to know which key to use
                - only mandatory inputs can be provided as keys if the default values in the function declarations are okay for you
                
    Return a full galfit.feedme configuration with header and body as formatted text.
    """
    
    header['name'] = 'header'
    correctNames   = fullKeys.keys()
    
    for pos, dic in enumerate([header] + listProfiles):
        # Check that name is okay in dictionnaries
        try:
            if dic['name'] not in correctNames:
                raise ValueError("given name %s is not correct. Please provide a name among the list %s. Cheers !" %correctNames)
        except KeyError:
            raise KeyError("key 'name' is not provided in one of the dictionnaries.")
        
        # Check that the given keys contain at the very least the mandatory ones
        checkInDict(dic, keys=fullKeys[dic['name']]['mandatory'] + ['name'], dictName='header or listProfiles')
        
        # Check that the given dictionnary only has valid keys
        checkDictKeys(dic, keys=fullKeys[dic['name']]['mandatory'] + fullKeys[dic['name']]['optional'][0] + ['name'], dictName='header or listProfiles')
    
        # Set default values if not provided
        if pos==0:
            header = setDict(dic, keys=fullKeys[dic['name']]['mandatory'] + fullKeys[dic['name']]['optional'][0] + ['name'], default=fullKeys[dic['name']]['mandatory'] + fullKeys[dic['name']]['optional'][1] + [dic['name']])
        else:
            listProfiles[pos-1] = setDict(dic, keys=fullKeys[dic['name']]['mandatory'] + fullKeys[dic['name']]['optional'][0] + ['name'], default=fullKeys[dic['name']]['mandatory'] + fullKeys[dic['name']]['optional'][1] + [dic['name']])
    
    # Append each profile configuration into a variable of type str
    out      = genHeader(**removeKeys(header, keys=['name']))
    for dic in listProfiles:
        out += "\n\n" + modelFunctions[dic['name']](**removeKeys(dic, keys=['name']))
        
    return out


def writeFeedmes(header, listProfiles, inputNames=[], outputNames=[], feedmeNames=[]):
    """
    Make galfit.feedme files using the same profiles.
    
    Mandatory inputs
    ----------------
        header : dict
            dictionnary with key names corresponding to input parameter names in genHeader function. This is used to generate the header part of the file.
            You do not need to provide an input and an output image file name as this is given with the inputNames keyword.
            
        listProfiles : list of dict
            list of dictionnaries. Each dictionnary corresponds to a profile:
                - in order for the function to know which profile to use, you must provide a key 'name' whose value is one of the following:
                    'deVaucouleur', 'edgeOnDisk', 'expDisk', 'ferrer', 'gaussian', 'king', 'moffat', 'nuker', 'psf', 'sersic', 'sky'
                - key names are input parameter names. See each profile description, to know which key to use
                - only mandatory inputs can be provided as keys if the default values in the function declarations are okay for you
                
    Optional inputs
    ---------------
        inputNames : list of str
            list of galaxies .fits files input names in the header
        feedmeNames : list of str
            list of .feedme galfit configuration files
        outputNames: list of str
            list of galaxies .fits files output names in the header
    
    Write full galfit.feedme configuration files for many galaxies.
    """
    
    if len(inputNames) != len(outputNames) or len(inputNames) != len(feedmeNames):
        raise ValueError("Lists intputNames, outputNames and feedmeNames do not have the same length. Please provide lists with similar length in order to know how many feedme files to generate. Cheers !")
    

def genHeader(outputImage, xmin, xmax, ymin, ymax,
              inputImage="none", sigmaImage="none", psfImage="none", maskImage="none", couplingFile="none",
              psfSamplingFactor=1, zeroPointMag=30.0, arcsecPerPixel=[0.03, 0.03],
              sizeConvX=None, sizeConvY=None, displayType="regular", option=0):
    """
    Constructs a galfit.feedme header as general as possible.
    
    Mandatory inputs
    ----------------
        outputImage : string
            name of the file within which the image will be stored. Technically, the output "image" is not an image but a (1+3)D fits file, where the layer 0 is a blank image whose header contains the fits keys, the layer 1 is the original image within the fitting region, layer 2 is the model image and layer 3 is the residual between the model and the original image. Note that the residual is computed by subtracting the layer 2 from the layer 1. Hence, if any galaxy component is missing in the model image (even if it was optimised), the residual will not reflect the "goodness" of fit.
        xmax : int
            maximum x coordinate of the fitting region box (in px)
        xmin : int
            minimum x coordinate of the fitting region box (in px)
        ymax : int
            maximum y coordinate of the fitting region box (in px)
        ymin : int
            minimum y coordinate of the fitting region box (in px)
        
    Optional inputs
    ---------------
        arcsecPerPixel : list of two floats
            angular resolution (in arcsec) of image pixels. First element is the angular resolution in the x direction, second element in the y direction.
        displayType : "regular", "both" or "curses"
            galfit display mode. "Regular" mode does not allow any interaction. "Both" and "curses" modes allow you to interact with galfit during the fitting routine (both will display in a xterm terminal the possible commands).
        inputImage : str
            name of the input file (fits file only). Note that if "none" is given, when running the feedme file, the fitting will be skipped and a model will be generated using the provided parameters.
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