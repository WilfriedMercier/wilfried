#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:29:00 2019

@author: Wilfried Mercier - IRAP

Functions related to automating galfit modelling.
"""

from wilfried.strings.strings import putStringsTogether, toStr, computeStringsLen, maxStringsLen

##########################################
#          Global variables              #
##########################################

posFormatX, posFormatY, PAFormat                 = ["%.1f"]*3
bOveraFormat, nFormat                            = ["%.2f"]*2

# mag is the total magnitude (Sersic profile for instance)
# mu is the surface brightness at radius rb (Ferrer profile)
# mu0 is the central surface brightness (King profile)
magFormat, muFormat, mu0Format                   = ["%.1f"]*3

# re is a Sersic effective radius
# rb is a Nuker profile break radius
# rs is a scale length for an exponential disk (rs = re/1.678)
# rt is the truncation radius of a Ferrer/king profile
# rc is the core radius of a King profile
reFormat, rbFormat, rsFormat, rtFormat, rcFormat = ["%.2f"]*5

# diskScaleLength and diskScaleHeight are disk scale-length and scale-height for an edge-on disk
diskScaleLengthFormat, diskScaleHeightFormat     = ["%.2f"]*2

# alpha, beta and gamma are for a Nuker profile
alphaFormat, betaFormat, gammaFormat             = ["%.1f"]*3

# Moffat/Gaussian profile parameters format
FWHMFormat, powerlawFormat                       = ["%.2f"]*2

# alpha and beta parameters for a Ferrer profile
alphaFerrerFormat, betaFerrerFormat              = ["%.1f"]*2

defaultComments = {'object':'Object type', 
                   'pos':'position x, y            [pixel]',
                   'magTot':'total magnitude',
                   're':'effective radius R_e         [pixel]',
                   'n':'Sersic exponent (deVauc=4, expdisk=1)',
                   'bOvera':'axis ratio (b/a)',
                   'PA':'position angle (PA)      [Degrees: Up=0, Left=90]',
                   'skipComponentInResidual':'Skip this model in output image?  (yes=1, no=0)',
                   'mu':'mu(Rb)                   [surface brightness mag. at Rb]',
                   'rb':'break radius Rb                   [pixel]',
                   'alpha':'alpha  (sharpness of transition)',
                   'beta':'beta   (outer powerlaw slope)',
                   'gamma':'gamma  (inner powerlaw slope)',
                   'rs':'scale-length R_s (R_e = 1.678R_s)        [pixel]',
                   'diskScaleLength':'disk scale-length        [pixel]',
                   'diskScaleHeight':'disk scale-height        [pixel]',
                   'FWHM':'FWHM                     [pixel]',
                   'powerlaw':'powerlaw',
                   'rt':'Outer truncation radius  [pixel]',
                   'alphaFerrer':'Alpha (outer truncation sharpness)',
                   'betaFerrer':'Beta (central slope)',
                   'mu0':'mu(0) (Central surface brightness in mag/arcsec^2)',
                   'rc':'Core radius Rc           [pixel]'}


#####################################################################
#              General function for galfit feedme files             #
#####################################################################

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
        if 0, galfit run normally as explained above
        if 1, the model image only is made using the given parameters
        if 2, an image block (data, model and residual) is made using the given parameters
        if 3, an image block with the first slice beeing the data, and the followings ones beeing one (best-fit) component per slice
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
    textLenMax      = max(len(outputImage), len(inputImage), len(sigmaImage), len(psfImage), 
                          len(psfSamplingFactor), len(maskImage), len(couplingFile), len(imageRegion), 
                          len(convBox), len(zeroPointMag), len(plateScale), len(displayType), len(option))
    formatText      = "%-" + "%d"%textLenMax + "s"
    
    #generate items text
    itemA = ("A) " + formatText)%inputImage + "   # Input data image (FITS file)"
    itemB = ("B) " + formatText)%outputImage + "   # Name for the output image"
    itemC = ("C) " + formatText)%sigmaImage + '   # Noise image name (made from data if blank or "none")'
    itemD = ("D) " + formatText)%psfImage + "   # Input PSF image for convolution (FITS file)"
    itemE = ("E) " + formatText)%psfSamplingFactor + "   # PSF Fine-Sampling Factor"
    itemF = ("F) " + formatText)%maskImage + "   # Pixel mask (ASCII file or FITS file with non-0 values)"
    itemG = ("G) " + formatText)%couplingFile + "   # File with parameter constraints (ASCII file)"
    itemH = ("H) " + formatText)%imageRegion + "   # Image region to fit (xmin xmax ymin ymax)"
    itemI = ("I) " + formatText)%convBox + "   # Size of convolution box (x y)"
    itemJ = ("J) " + formatText)%zeroPointMag + "   # Magnitude photometric zeropoint"
    itemK = ("K) " + formatText)%plateScale + "   # Plate scale (dx dy). Relevant only for Nuker/King models."
    itemO = ("O) " + formatText)%displayType + "   # Display type (regular, curses, both)"
    itemP = ("P) " + formatText)%option + "   # Options: 0=normal run; 1,2=make model/imgblock & quit; 3=normal run and separate components"
    
    textLenMax      = max(computeStringsLen([itemA, itemB, itemC, itemD, itemE, itemF, itemG, itemH, itemI, itemJ, itemK, itemO, itemP]))
    
    out = putStringsTogether(["="*textLenMax, "# IMAGE PARAMETERS",
                           itemA, itemB, itemC, itemD, itemE, itemF, itemG, itemH, itemI, itemJ, itemK, itemO, itemP,
                           "="*textLenMax])
    return out


def genModel(modelName, listLineIndex, params, fixedOrNot, paramsFormat, comments=None, noComments=False, mainComment=None):
    """
    Very general function which can output any galfit model configuration. This function is not supposed to be used directly by the user, but should be called by a function whose goal is to output the galfilt configuration of a specific profile. 
    This should only be seen as a bare skeleton which is used in function calls in order to build specific galfit model configuration for feedme files.
    
    Mandatory inputs
    ----------------
        fixedOrNot : list of boolean
            whether each parameter given in params must be fixed during galfit fitting routine or not. This takes the SAME SHAPE as params. See params description for more information.
        listLineIndex : list of str
            list of galfit indices for each line (generally from 1 to 10, including Z at the end)
        modelName : str
            name of the model used
        params : list
            parameters value for each line given in the same order as the indices. 
            If, for a certain index, there are more than one parameter (ex: model center), then provide these as a list.
            
            Example:
                Say two indices are given such that listLineIndex=[1, 2], where the first one is for the Sersic index n=1 and the other for the effective radius re=10, then we would have
                    params=[1, 10]
                Say three indices are given (listLineIndex=[1, 2, 3] for example), with the first two as before and the last one as the model center position (let us assume it lies at position 3, 5), then we would have
                    params = [1, 10 [3, 5]]
                This generalises for any number of parameters.
                
        paramsFormat : list of str
            format of the parameters used for the output. This uses old Python format. If you do not care about the output format, just provide a simple format for each argument ("%d" for int, "%f" for float, "%e" for scientific notation, etc.). Otherwise, you can use more complex notation to truncate values, align strings to the left or the right and so on.
            The SHAPE of this parameter should be IDENTICAL as that of params. See params description for more information.
                
    Optional inputs
    ---------------
        comments : list of str
            comments to add at the end of the lines. If you do not want to provide a comment for a specific index, just give None.
        mainComment : str
            main comment which appears before the configuration line (generally the full model name)
        noComments : boolean
            whether to not provide any comments or not.
        
    Return the galfit model configuration as formatted text.
    """

    length        = len(listLineIndex)  
    listLineIndex = toStr(listLineIndex)
    
    #first line with the model name
    if mainComment is not None:
        firstLine     = "# " + mainComment + "\n"
    else:
        firstLine     = ""
    
    #define comment as an empty string if not provided
    if comments is None:
        comments  = [""]*length
    elif len(comments) != (length+1) and not noComments:
        print("InputError: optional argument 'comment' does not have the same length as listLineIndex. Either provide no comment at all, or one for every line. Cheers !")
        return None
    
    # This is the maximum string length in listLineIndex (it is used to align the first column only)
    lenIndex      = max(computeStringsLen(listLineIndex))
    formatIndex   = "%" + "%d" %lenIndex + "s" + ") "
    
    # Create a list with all lines as an element
    # First line is model name which is not provided in the index list
    allLines      = [formatIndex%"0" + modelName]
    
    # Generate each (other) line separately
    for num, idx, pm, fx, pf in zip(range(1, length+1) , listLineIndex, params, fixedOrNot, paramsFormat):
        #First add the index value
        allLines.append(formatIndex %idx)
        
        #Then check whether there are more than one parameter on the line (for instance a model center)
        #If so, loop over them, otherwise just add the single parameter and its fixedOrNot corresponding flag
        try:
            for valpm, valpf in zip(pm, pf):
                allLines[num] += valpf %valpm + " "
            for valfx in fx:
                allLines[num] += str(valfx) + " "
        except:
            allLines[num]     += pf %pm + " " + str(fx) + " "
    
    if not noComments:
        # Retrieve the length of the longest line in order to align the comments
        maxLineLen             = maxStringsLen(allLines)
        for num, com in zip(range(length+1), comments):
            if com is not None:
                allLines[num]  = ("%-" + "%d" %maxLineLen + "s") %allLines[num] + "  # " + com
            
    return putStringsTogether([firstLine] + allLines)
    

##############################################################
#               Profiles avaiblable in galfit                #
##############################################################
    

def gendeVaucouleur(posX, posY, magTot, re, bOvera=1.0, PA=0.0, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    Construct a de Vaucouleur function configuration.
    
    Mandatory inputs
    ----------------
        magTot : float
            total integrated magnitude of the profile
        posX : int/float
            X position of the de Vaucouleur profile center (in px)
        poxY : int/float
            Y position of the de Vaucouleur profile center  (in px)
        re : float
            half-light (effective) radius of the profile (in px)
            
    Optional inputs
    ---------------
        bOvera : float between 0 and 1
            axis ratio b/a of the minor over major axes
        comments : dict
            dictionnary which contains a comment for each line. By default, comments is set to None, and default comments will be used instead.
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, magTot for total magnitude, 'bOvera' for b/a ratio, etc.).
            The key value is the comment you want.
            
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
            
            WARNINGS:
                - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'posX' or 'posY' as both values appear on the same line
                - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
                - to add a comment to the line with index Z, use the key name 'Zline'
            
        noComments : boolean
            whether to not provide any comments or not
        PA : int/float
            position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
        fixedParams : list
            list of parameters names which must be fixed during galfit fitting routine. BY DEFAULT, ALL PARAMETERS ARE SET FREE.
            For instance, if one wants to fix re and magTot, one may provide fixedParams=["re", "magTot"] in the function call.
        skipComponentInResidual : boolean
            whether to to not take into account this component when computing the residual or not. If False, the residual will be computed using the best fit model taking into account all the components and the input data. If False, the residual will skip this component in the best-fit model.
        
    Returns a complete de Vaucouleur function galfit configuration as formatted text.
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["posX", "posY", "magTot", "re", "n", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "magTot", "re", "n", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("devauc",
                    [1, 3, 4, 9, 10, 'Z'],
                    [[posX, posY], magTot, re, bOvera, PA, skipComponentInResidual],
                    [[isFixed["posX"], isFixed["posY"]], isFixed["magTot"], isFixed["re"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[posFormatX, posFormatY], magFormat, reFormat, bOveraFormat, PAFormat, "%d"],
                    comments=[comments['object'], comments["pos"], comments["magTot"], comments["re"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='de Vaucouleur function')


def genEdgeOnDisk(posX, posY, mu, diskScaleLength, diskScaleHeight, PA=0.0, 
                  skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    Construct an Edge-on Disk function configuration.
    
    Mandatory inputs
    ----------------
        diskScaleHeight : float
            disk scale-height perpendicular to the disk
        mu : float
            central surface brightness (mag/arcsec^2) of the profile
        posX : int/float
            X position of the edge-on disk profile center (in px)
        poxY : int/float
            Y position of the edge-on disk profile center  (in px)
        diskScaleLength : float
            major axis disk scale-length (in px)

            
    Optional inputs
    ---------------
        comments : dict
            dictionnary which contains a comment for each line. By default, comments is set to None, and default comments will be used instead.
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, mu for central surface brightness, etc.).
            The key value is the comment you want.
            
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
            
            WARNINGS:
                - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'posX' or 'posY' as both values appear on the same line
                - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
                - to add a comment to the line with index Z, use the key name 'Zline'
            
        noComments : boolean
            whether to not provide any comments or not
        PA : int/float
            position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
        fixedParams : list
            list of parameters names which must be fixed during galfit fitting routine. BY DEFAULT, ALL PARAMETERS ARE SET FREE.
            For instance, if one wants to fix diskScaleLength and mu, one may provide fixedParams=["diskScaleLength", "mu"] in the function call.
        skipComponentInResidual : boolean
            whether to to not take into account this component when computing the residual or not. If False, the residual will be computed using the best fit model taking into account all the components and the input data. If False, the residual will skip this component in the best-fit model.
        
    Returns a complete Edge-on disk function galfit configuration as formatted text.
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["posX", "posY", "mu", "diskScaleHeight", "diskScaleLength", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mu", "diskScaleHeight", "diskScaleLength", "PA", "skipComponentInResidual"], comments)
            
    return genModel("edgedisk",
                    [1, 3, 4, 5, 10, 'Z'],
                    [[posX, posY], mu, diskScaleHeight, diskScaleLength, PA, skipComponentInResidual],
                    [[isFixed["posX"], isFixed["posY"]], isFixed["mu"], isFixed["diskScaleHeight"], isFixed["diskScaleLength"], isFixed["PA"], ""],
                    [[posFormatX, posFormatY], muFormat, diskScaleHeightFormat, diskScaleLengthFormat, PAFormat, "%d"],
                    comments=[comments['object'], comments["pos"], comments["mu"], comments["diskScaleHeight"], comments["diskScaleLength"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Edge-on disk function')


def genExpDisk(posX, posY, magTot, rs, bOvera=1.0, PA=0.0, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    Construct an exponential disk function configuration.
    
    Mandatory inputs
    ----------------
        magTot : float
            total integrated magnitude of the profile
        posX : int/float
            X position of the exponential disk profile center (in px)
        poxY : int/float
            Y position of the exponential disk profile center  (in px)
        rs : float
            disk scale-length (in px) such that rs = re/1.678, with re the effective radius of an equivalent n=1 Sersic profile

            
    Optional inputs
    ---------------
        bOvera : float between 0 and 1
            axis ratio b/a of the minor over major axes
        comments : dict
            dictionnary which contains a comment for each line. By default, comments is set to None, and default comments will be used instead.
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, magTot for total magnitude, 'bOvera' for b/a ratio, etc.).
            The key value is the comment you want.
            
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
            
            WARNINGS:
                - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'posX' or 'posY' as both values appear on the same line
                - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
                - to add a comment to the line with index Z, use the key name 'Zline'
            
        noComments : boolean
            whether to not provide any comments or not
        PA : int/float
            position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
        fixedParams : list
            list of parameters names which must be fixed during galfit fitting routine. BY DEFAULT, ALL PARAMETERS ARE SET FREE.
            For instance, if one wants to fix rs and PA, one may provide fixedParams=["rs", "PA"] in the function call.
        skipComponentInResidual : boolean
            whether to to not take into account this component when computing the residual or not. If False, the residual will be computed using the best fit model taking into account all the components and the input data. If False, the residual will skip this component in the best-fit model.
        
    Returns a complete Exponential disk function galfit configuration as formatted text.
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["posX", "posY", "magTot", "rs", "n", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "magTot", "rs", "n", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("expdisk",
                    [1, 3, 4, 9, 10, 'Z'],
                    [[posX, posY], magTot, rs, bOvera, PA, skipComponentInResidual],
                    [[isFixed["posX"], isFixed["posY"]], isFixed["magTot"], isFixed["rs"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[posFormatX, posFormatY], magFormat, rsFormat, bOveraFormat, PAFormat, "%d"],
                    comments=[comments['object'], comments["pos"], comments["magTot"], comments["rs"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Exponential function')


def genFerrer(posX, posY, mu, rt, alphaFerrer=3.0, betaFerrer=2.5, bOvera=1.0, PA=0.0, 
             skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    Construct a Ferrer function (generally used to fit bars) configuration.
    
    Mandatory inputs
    ----------------
        mu : float
            surface brightness (mag/arcsec^2) at radius rb
        posX : int/float
            X position of the Ferrer profile center (in px)
        poxY : int/float
            Y position of the Ferrer profile center  (in px)
        rt : float
            outer truncation radius (in px)
            
    Optional inputs
    ---------------
        alphaFerrer : float
            sharpness of the truncation
        betaFerrer : float
            central slope
        bOvera : float between 0 and 1
            axis ratio b/a of the minor over major axes
        comments : dict
            dictionnary which contains a comment for each line. By default, comments is set to None, and default comments will be used instead.
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, mu for central surface brightness, 'bOvera' for b/a ratio, etc.).
            The key value is the comment you want.
            
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
            
            WARNINGS:
                - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'posX' or 'posY' as both values appear on the same line
                - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'
                - to add a comment to the line with index Z, use the key name 'Zline'
            
        noComments : boolean
            whether to not provide any comments or not   
        PA : int/float
            position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
        fixedParams : list
            list of parameters names which must be fixed during galfit fitting routine. BY DEFAULT, ALL PARAMETERS ARE SET FREE.
            For instance, if one wants to fix rt and mu, one may provide fixedParams=["rt", "mu"] in the function call.
        skipComponentInResidual : boolean
            whether to to not take into account this component when computing the residual or not. If False, the residual will be computed using the best fit model taking into account all the components and the input data. If False, the residual will skip this component in the best-fit model.
        
    Returns a complete Ferrer function galfit configuration as formatted text.
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["posX", "posY", "mu", "rt", "alphaFerrer", "betaFerrer", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mu", "rt", "alphaFerrer", "betaFerrer", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("nuker", 
                    [1, 3, 4, 5, 6, 9, 10, 'Z'],
                    [[posX, posY], mu, rt, alphaFerrer, betaFerrer, bOvera, PA, skipComponentInResidual],
                    [[isFixed["posX"], isFixed["posY"]], isFixed["mu"], isFixed["rt"], isFixed["alphaFerrer"], isFixed["betaFerrer"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[posFormatX, posFormatY], muFormat, rtFormat, alphaFerrerFormat, betaFerrerFormat, bOveraFormat, PAFormat, "%d"],
                    comments=[comments['object'], comments["pos"], comments["mu"], comments["rt"], comments["alphaFerrer"], comments["betaFerrer"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Nuker function')
    
    
def genGaussian(posX, posY, magTot, FWHM, bOvera=1.0, PA=0.0, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    Construct a Gaussian function configuration.
    
    Mandatory inputs
    ----------------
        FWHM : float
            full width at half maximum of the PSF (in px)
        magTot : float
            total integrated magnitude of the profile
        posX : int/float
            X position of the Gaussian profile center (in px)
        poxY : int/float
            Y position of the Gaussian profile center  (in px)
            
    Optional inputs
    ---------------
        bOvera : float between 0 and 1
            axis ratio b/a of the minor over major axes
        comments : dict
            dictionnary which contains a comment for each line. By default, comments is set to None, and default comments will be used instead.
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, magTot for total magnitude, 'bOvera' for b/a ratio, etc.).
            The key value is the comment you want.
            
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
            
            WARNINGS:
                - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'posX' or 'posY' as both values appear on the same line
                - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
                - to add a comment to the line with index Z, use the key name 'Zline'
            
        noComments : boolean
            whether to not provide any comments or not
        PA : int/float
            position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
        fixedParams : list
            list of parameters names which must be fixed during galfit fitting routine. BY DEFAULT, ALL PARAMETERS ARE SET FREE.
            For instance, if one wants to fix FWHM and posX, one may provide fixedParams=["FWHM", "posX"] in the function call.
        skipComponentInResidual : boolean
            whether to to not take into account this component when computing the residual or not. If False, the residual will be computed using the best fit model taking into account all the components and the input data. If False, the residual will skip this component in the best-fit model.
        
    Returns a complete Gaussian function galfit configuration as formatted text.
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["posX", "posY", "magTot", "FWHM", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "magTot", "FWHM", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("gaussian",
                    [1, 3, 4, 9, 10, 'Z'],
                    [[posX, posY], magTot, FWHM, bOvera, PA, skipComponentInResidual],
                    [[isFixed["posX"], isFixed["posY"]], isFixed["magTot"], isFixed["FWHM"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[posFormatX, posFormatY], magFormat, FWHMFormat, bOveraFormat, PAFormat, "%d"],
                    comments=[comments['object'], comments["pos"], comments["magTot"], comments["FWHM"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Gaussian function')
    
    
def genKing(posX, posY, mu0, rc, rt, powerlaw=2.0, bOvera=1.0, PA=0.0, 
            skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    Construct an empirical King profile (generally used to fit globular clusters) configuration.
    
    Mandatory inputs
    ----------------
        mu0 : float
            central surface brightness (mag/arcsec^2)
        powerlaw : float
            powerlaw (powerlaw=2.0 for a standard King profile)
        posX : int/float
            X position of the King profile center (in px)
        poxY : int/float
            Y position of the King profile center  (in px)
        rc : float
            core radius (in px) 
        rt : float
            truncation radius (in px)
        
    Optional inputs
    ---------------
        bOvera : float between 0 and 1
            axis ratio b/a of the minor over major axes
        comments : dict
            dictionnary which contains a comment for each line. By default, comments is set to None, and default comments will be used instead.
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, mu for central surface brightness, 'bOvera' for b/a ratio, etc.).
            The key value is the comment you want.
            
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
            
            WARNINGS:
                - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'posX' or 'posY' as both values appear on the same line
                - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'
                - to add a comment to the line with index Z, use the key name 'Zline'
            
        noComments : boolean
            whether to not provide any comments or not   
        PA : int/float
            position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
        fixedParams : list
            list of parameters names which must be fixed during galfit fitting routine. BY DEFAULT, ALL PARAMETERS ARE SET FREE.
            For instance, if one wants to fix rb and mu, one may provide fixedParams=["rb", "mu"] in the function call.
        skipComponentInResidual : boolean
            whether to to not take into account this component when computing the residual or not. If False, the residual will be computed using the best fit model taking into account all the components and the input data. If False, the residual will skip this component in the best-fit model.
        
    Returns a complete empirical King profile galfit configuration as formatted text.
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["posX", "posY", "mu0", "rc", "rt", "powerlaw", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mu0", "rc", "rt", "powerlaw", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("king", 
                    [1, 3, 4, 5, 6, 9, 10, 'Z'],
                    [[posX, posY], mu0, rc, rt, powerlaw, bOvera, PA, skipComponentInResidual],
                    [[isFixed["posX"], isFixed["posY"]], isFixed["mu0"], isFixed["rc"], isFixed["rt"], isFixed["powerlaw"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[posFormatX, posFormatY], mu0Format, rcFormat, rtFormat, powerlawFormat, bOveraFormat, PAFormat, "%d"],
                    comments=[comments['object'], comments["pos"], comments["mu0"], comments["rc"], comments["rt"], comments["powerlaw"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='The Empirical King Profile')

    
def genMoffat(posX, posY, magTot, FWHM, powerlaw=1.0, bOvera=1.0, PA=0.0, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    Construct a de Vaucouleur function configuration.
    
    Mandatory inputs
    ----------------
        FWHM : float
            full width at half maximum of the PSF (in px)
        magTot : float
            total integrated magnitude of the profile
        posX : int/float
            X position of the Moffat profile center (in px)
        poxY : int/float
            Y position of the Moffat profile center  (in px)
            
    Optional inputs
    ---------------
        bOvera : float between 0 and 1
            axis ratio b/a of the minor over major axes
        comments : dict
            dictionnary which contains a comment for each line. By default, comments is set to None, and default comments will be used instead.
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, magTot for total magnitude, 'bOvera' for b/a ratio, etc.).
            The key value is the comment you want.
            
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
            
            WARNINGS:
                - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'posX' or 'posY' as both values appear on the same line
                - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
                - to add a comment to the line with index Z, use the key name 'Zline'
            
        noComments : boolean
            whether to not provide any comments or not
        PA : int/float
            position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
        powerlaw : float
            powerlaw/concentration index in the Moffat profile
        fixedParams : list
            list of parameters names which must be fixed during galfit fitting routine. BY DEFAULT, ALL PARAMETERS ARE SET FREE.
            For instance, if one wants to fix FWHM and posX, one may provide fixedParams=["FWHM", "posX"] in the function call.
        skipComponentInResidual : boolean
            whether to to not take into account this component when computing the residual or not. If False, the residual will be computed using the best fit model taking into account all the components and the input data. If False, the residual will skip this component in the best-fit model.
        
    Returns a complete de Vaucouleur function galfit configuration as formatted text.
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["posX", "posY", "magTot", "FWHM", "powerlaw", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "magTot", "FWHM", "powerlaw", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("moffat",
                    [1, 3, 4, 5, 9, 10, 'Z'],
                    [[posX, posY], magTot, FWHM, powerlaw, bOvera, PA, skipComponentInResidual],
                    [[isFixed["posX"], isFixed["posY"]], isFixed["magTot"], isFixed["FWHM"], isFixed["powerlaw"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[posFormatX, posFormatY], magFormat, FWHMFormat, powerlawFormat, bOveraFormat, PAFormat, "%d"],
                    comments=[comments['object'], comments["pos"], comments["magTot"], comments["FWHM"], comments["powerlaw"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Moffat function')
    
    
def genNuker(posX, posY, mu, rb, alpha=1.0, beta=0.5, gamma=0.7, bOvera=1.0, PA=0.0, 
             skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    Construct a Nuker function (introduced to fit the nuclear region of nearby galaxies) configuration.
    
    Mandatory inputs
    ----------------
        mu : float
            surface brightness (mag/arcsec^2) at radius rb
        posX : int/float
            X position of the Nuker profile center (in px)
        poxY : int/float
            Y position of the Nuker profile center  (in px)
        rb : float
            break radius (in px) where the slope is the average between \beta and \gamma and where the maximum curvature (in log space) is reached. It roughly corresponds to the radius of transition between the inner and outer powerlaws.
            
    Optional inputs
    ---------------
        alpha : float
            sharpness of transition between the inner part and the outer part
        beta : float
            outer powerlaw slope
        bOvera : float between 0 and 1
            axis ratio b/a of the minor over major axes
        comments : dict
            dictionnary which contains a comment for each line. By default, comments is set to None, and default comments will be used instead.
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, mu for central surface brightness, 'bOvera' for b/a ratio, etc.).
            The key value is the comment you want.
            
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
            
            WARNINGS:
                - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'posX' or 'posY' as both values appear on the same line
                - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'
                - to add a comment to the line with index Z, use the key name 'Zline'
            
        gamma: float
            inner powerlaw slope
        noComments : boolean
            whether to not provide any comments or not   
        PA : int/float
            position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
        fixedParams : list
            list of parameters names which must be fixed during galfit fitting routine. BY DEFAULT, ALL PARAMETERS ARE SET FREE.
            For instance, if one wants to fix rb and mu, one may provide fixedParams=["rb", "mu"] in the function call.
        skipComponentInResidual : boolean
            whether to to not take into account this component when computing the residual or not. If False, the residual will be computed using the best fit model taking into account all the components and the input data. If False, the residual will skip this component in the best-fit model.
        
    Returns a complete Nuker function galfit configuration as formatted text.
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["posX", "posY", "mu", "rb", "alpha", "beta", "gamma", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mu", "rb", "alpha", "beta", "gamma", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("nuker", 
                    [1, 3, 4, 5, 6, 7, 9, 10, 'Z'],
                    [[posX, posY], mu, rb, alpha, beta, gamma, bOvera, PA, skipComponentInResidual],
                    [[isFixed["posX"], isFixed["posY"]], isFixed["mu"], isFixed["rb"], isFixed["alpha"], isFixed["beta"], isFixed["gamma"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[posFormatX, posFormatY], muFormat, rbFormat, alphaFormat, betaFormat, gammaFormat, bOveraFormat, PAFormat, "%d"],
                    comments=[comments['object'], comments["pos"], comments["mu"], comments["rb"], comments["alpha"], comments["beta"], comments["gamma"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Nuker function')


def genPSF(posX, posY, magTot, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    Construct PSF function configuration. This PSF is technically not a function, but uses the psf image provided in the header to fit a given point source.
    
    Mandatory inputs
    ----------------
        magTot : float
            total integrated magnitude of the profile
        posX : int/float
            X position of the profile center (in px)
        poxY : int/float
            Y position of the profile center  (in px)
            
    Optional inputs
    ---------------
        comments : dict
            dictionnary which contains a comment for each line. By default, comments is set to None, and default comments will be used instead.
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, magTot for total magnitude, 'bOvera' for b/a ratio, etc.).
            The key value is the comment you want.
            
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
            
            WARNINGS:
                - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'posX' or 'posY' as both values appear on the same line
                - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
                - to add a comment to the line with index Z, use the key name 'Zline'
            
        noComments : boolean
            whether to not provide any comments or not
        fixedParams : list
            list of parameters names which must be fixed during galfit fitting routine. BY DEFAULT, ALL PARAMETERS ARE SET FREE.
            For instance, if one wants to fix FWHM and posX, one may provide fixedParams=["FWHM", "posX"] in the function call.
        skipComponentInResidual : boolean
            whether to to not take into account this component when computing the residual or not. If False, the residual will be computed using the best fit model taking into account all the components and the input data. If False, the residual will skip this component in the best-fit model.
        
    Returns a complete PSF function galfit configuration as formatted text.
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["posX", "posY", "magTot"], fixedParams)
    comments = createCommentsDict(["object", "pos", "magTot", "skipComponentInResidual"], comments)
            
    return genModel("psf",
                    [1, 3, 'Z'],
                    [[posX, posY], magTot, skipComponentInResidual],
                    [[isFixed["posX"], isFixed["posY"]], isFixed["magTot"], ""],
                    [[posFormatX, posFormatY], magFormat, "%d"],
                    comments=[comments['object'], comments["pos"], comments["magTot"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='PSF fit')


def genSersic(posX, posY, magTot, re, n=4, bOvera=1.0, PA=0.0, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    Construct a Sersic function configuration.
    
    Mandatory inputs
    ----------------
        magTot : float
            total integrated magnitude of the profile
        posX : int/float
            X position of the Sersic profile center (in px)
        poxY : int/float
            Y position of the Sersic profile center  (in px)
        re : float
            half-light (effective) radius of the profile (in px)
            
    Optional inputs
    ---------------
        bOvera : float between 0 and 1
            axis ratio b/a of the minor over major axes
        comments : dict
            dictionnary which contains a comment for each line. By default, comments is set to None, and default comments will be used instead.
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, magTot for total magnitude, 'bOvera' for b/a ratio, etc.).
            The key value is the comment you want.
            
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
            
            WARNINGS:
                - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'posX' or 'posY' as both values appear on the same line
                - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
                - to add a comment to the line with index Z, use the key name 'Zline'
            
        n : int/float
            Sersic index of the profile
        noComments : boolean
            whether to not provide any comments or not
        PA : int/float
            position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
        fixedParams : list
            list of parameters names which must be fixed during galfit fitting routine. BY DEFAULT, ALL PARAMETERS ARE SET FREE.
            For instance, if one wants to fix re and n, one may provide fixedParams=["re", "n"] in the function call.
        skipComponentInResidual : boolean
            whether to to not take into account this component when computing the residual or not. If False, the residual will be computed using the best fit model taking into account all the components and the input data. If False, the residual will skip this component in the best-fit model.
        
    Returns a complete Sersic function galfit configuration as formatted text.
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["posX", "posY", "magTot", "re", "n", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "magTot", "re", "n", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("sersic", 
                    [1, 3, 4, 5, 9, 10, 'Z'],
                    [[posX, posY], magTot, re, n, bOvera, PA, skipComponentInResidual],
                    [[isFixed["posX"], isFixed["posY"]], isFixed["magTot"], isFixed["re"], isFixed["n"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[posFormatX, posFormatY], magFormat, reFormat, nFormat, bOveraFormat, PAFormat, "%d"],
                    comments=[comments['object'], comments["pos"], comments["magTot"], comments["re"], comments["n"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Sersic function')



    
    
################################################################
#                   Miscellanous functions                     #
################################################################

def createIsFixedDict(fullListOfNames, fixedParams):
    """
    Make a dictionnary which tells which parameters should be fixed (value of 0) and which should not (value of 1).
    
    Mandatory inputs
    ----------------
        fixedParams : list of str
            list of parameters names which must be fixed during galfit fitting routine. BY DEFAULT, ALL PARAMETERS ARE SET FREE.
            For instance, if one wants to fix re and n, one may provide fixedParams=["re", "n"] in the function call.
        fullListOfNames : list of str
            complete list of parameter names which can either be fixed or let free

    Returns a dictionnary with key names equal to parameter names of the model and key values equal to 0 or 1.
    """
    
    isFixed = {}
    for param in fullListOfNames:
        # If parameter must be fixed, put 0
        if param in fixedParams:
            isFixed[param] = 0
        else:
            isFixed[param] = 1
    return isFixed
    

def createCommentsDict(fullListOfNames, comments):
    """
    Make a dictionnary which tells the comments associated to each line in the galfit configuration file
    
    Mandatory inputs
    ----------------
        comments : dict
            dictionnary which contains a comment for each line. By default, comments is set to None, and default comments will be used instead.
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, magTot for total magnitude, 'bOvera' for b/a ratio, etc.).
            The key value is the comment you want.
            
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
            
            WARNINGS:
                - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'posX' or 'posY' as both values appear on the same line
                - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
                
        fullListOfNames : list of str
            complete list of parameter names which can either be fixed or let free
    
    Returns a dictionnary with key names equal to parameter names of the model and key values equal to their comment.
    """
    
    global defaultComments

    if comments is not None:
        for param in fullListOfNames:
            if param not in comments.keys():
                # Set comments to None if not provided
                comments[param] = None
    else:
        comments = {}
        for param in fullListOfNames:
            comments[param]     = defaultComments[param]
            
    return comments
    
    
    
    
    
    
    
              
