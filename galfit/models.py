#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Galfit model functions.
"""

from wilfried.utilities.strings import putStringsTogether, toStr, computeStringsLen, maxStringsLen
from numpy                      import ravel

##########################################
#          Global variables              #
##########################################

formats = {'posFormat':"%.1f",
           'PAFormat':"%.1f",
           'bOveraFormat':"%.2f", 
           'nFormat':"%.2f",
           'magFormat':"%.1f", 
           'muFormat':"%.1f",
           'radiusFormat':"%.2f",
           'powerlawFormat':"%.2f",
           'backgroundFormat':"%.2f",
           'GradientFormat':"%.3f",
           'bendingFormat':"%.2f",
           'ampFourierFormat':"%.2f",
           'phaseFourierFormat':"%.1f",
           'boxy_diskyFormat':"%.2f"
          }

defaultComments = {'object':                    'Object type', 
                   'pos':                       'position x, y                         [pixel]',
                   'mag':                       'total magnitude',
                   're':                        'effective radius R_e                  [pixel]',
                   'n':                         'Sersic exponent (deVauc=4, expdisk=1)',
                   'bOvera':                    'axis ratio (b/a)',
                   'PA':                        'position angle (PA)                   [Degrees: Up=0, Left=90]',
                   'skipComponentInResidual':   'Skip this model in output image?  (yes=1, no=0)',
                   'mu':                        'mu(Rb)                                [surface brightness mag. at Rb]',
                   'rb':                        'break radius Rb                       [pixel]',
                   'alpha':                     'alpha  (sharpness of transition)',
                   'beta':                      'beta   (outer powerlaw slope)',
                   'gamma':                     'gamma  (inner powerlaw slope)',
                   'rs':                        'scale-length R_s (R_e = 1.678R_s)     [pixel]',
                   'diskScaleLength':           'disk scale-length                     [pixel]',
                   'diskScaleHeight':           'disk scale-height                     [pixel]',
                   'FWHM':                      'FWHM                                  [pixel]',
                   'powerlaw':                  'powerlaw',
                   'rt':                        'Outer truncation radius               [pixel]',
                   'alphaFerrer':               'Alpha (outer truncation sharpness)',
                   'betaFerrer':                'Beta (central slope)',
                   'mu0':                       'mu(0) (Central surface brightness in mag/arcsec^2)',
                   'rc':                        'Core radius Rc                        [pixel]',
                   'background':                'sky background                        [ADU counts]',
                   'xGradient':                 'dsky/dx (sky gradient in x)',
                   'yGradient':                 'dsky/dy (sky gradient in y)'
                  }


################################################################
#                   Miscellanous functions                     #
################################################################

def createIsFixedDict(fullListOfNames, fixedParams):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Make a dictionnary which stores which parameters should be fixed (value of 0) and which should not (value of 1).

    :param list[str] fixedParams: list of parameters names which must be fixed during galfit fitting routine. 
    
        .. note::
            
            BY DEFAULT, ALL PARAMETERS ARE SET FREE.
        
        .. rubric:: **Example**
        
        For instance, if one wants to fix :samp:`re` and :samp:`n`, one may provide 
        
        >>> fixedParams=["re", "n"]
        >>> createIsFixedDict(fullListOfNames, fixedParams)
        
    :param list[str] fullListOfNames: complete list of parameter names which can either be fixed or let free

    :returns: dictionnary with key names equal to parameter names of the model and key values equal to 0 or 1
    :rtype: dict
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
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Make a dictionnary which stores the comments associated to each line in the galfit configuration file.

    :param dict comments: dictionnary which contains a comment for each line. If **comments** is None, default comments will be used instead.
        
        .. note::
            
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, 'mag' for total magnitude, 'bOvera' for b/a ratio, etc.). The key value is the comment you want.
        
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
        
        .. warning::
            
            - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'x' or 'y' as both values appear on the same line
            - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
        
    :param list[str] fullListOfNames: complete list of parameter names which can either be fixed or let free
    
    :returns: dictionnary with key names equal to parameter names of the model and key values equal to their comment
    :rtype: dict
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


################################################################
#                   General main function                      #
################################################################

def genModel(modelName, listLineIndex, params, fixedOrNot, paramsFormat, comments=None, noComments=False, mainComment=None, removeLine0=False, prefix=""):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Very general function which can output any GALFIT model configuration. 
    
    .. warning::

        This function is not supposed to be used directly by the user, but should be called by a function whose goal is to output the galfilt configuration of a specific profile. 
    
        This should only be seen as a bare skeleton which is used in function calls in order to build specific galfit model configuration for feedme files.

    :param list[bool] fixedOrNot: whether each parameter given in **params** must be fixed during GALFIT fitting routine or not. This takes the same shape as **params**.
    :param list[str] listLineIndex: list of GALFIT indices for each line (generally from 1 to 10, including Z at the end)
    :param str modelName: name of the model used
    :param list params: parameters value for each line given in the same order as the indices. If, for a certain index, there are more than one parameter (ex: model center), then provide these as a list.
        
        .. rubric:: ***Example**
        
        Say two indices are given such that 
        
        >>> listLineIndex = [1, 2] 
        
        where the first one is for the Sersic index n=1 and the other for the effective radius re=10, then we would have
        
        >>> params = [1, 10]
        
        Say three indices are given such that
        
        >>> listLineIndex = [1, 2, 3] 
        
        with the first two as before and the last one as the model center position (let us assume it lies at position (3, 5)), then we would have
            
        >>> params = [1, 10 [3, 5]]
        
        This generalises for any number of parameters.
            
    :param list[str] paramsFormat: format of the parameters used for the output. The shape of this parameter should be identical as that of params.
    
        .. note::
            
            This uses old Python format. If you do not care about the output format, just provide a simple format for each argument ("%d" for int, "%f" for float, "%e" for scientific notation, etc.). 
            
            Otherwise, you can use more complex notation to truncate values, align strings to the left or the right and so on.
        
    :param list[str] comments: (**Optional**) comments to add at the end of the lines. If you do not want to provide a comment for a specific index, just give None.
    :param str mainComment: (**Optional**) main comment which appears before the configuration line (generally the full model name)
    :param bool noComments: (**Optional**) whether to not provide any comments or not
    :param str prefix: (**Optional**) prefix to add before the line index. For instance the index 1) would become B1) with the prefix "B".
    :param bool removeLine0: whether to remove line number 0 (model name). This is useful when one wants to make an output for additional terms such as bending modes.
        
    :returns: galfit model configuration as formatted text
    :rtype: str
    """

    listLineIndex = toStr(listLineIndex)
    length        = len(listLineIndex)
    
    #this is used to know the size the comments list should have
    if removeLine0:
        size = length
    else:
        size = length+1
    
    #first line with the model name
    if mainComment is not None:
        firstLine     = "# " + mainComment + "\n\n"
    else:
        firstLine     = ""

    #define comment as an empty string if not provided
    if not noComments:
        if comments is None:
            comments  = [""]*size
        elif len(comments) != size:
            print("InputError: optional argument 'comment' does not have the same length as 'listLineIndex'. Either provide no comment at all, or one for every line. Cheers !")
            return None
        
    # This is the maximum string length in listLineIndex (it is used to align the first column only)
    lenIndex      = max(computeStringsLen(listLineIndex))
    formatIndex   = prefix + "%" + "%d" %lenIndex + "s" + ") "
    
    # Create a list with all lines as an element. First line is model name which is not provided in the index list
    if not removeLine0:
        allLines      = [formatIndex%"0" + modelName]
        startPoint    = 1
    else:
        allLines      = []
        startPoint    = 0
    
    # Generate each (other) line separately
    for num, idx, pm, fx, pf in zip(range(startPoint, length+1) , listLineIndex, params, fixedOrNot, paramsFormat):
        
        # First add the index value
        allLines.append(formatIndex %idx)
        
        #T hen check whether there are more than one parameter on the line (for instance a model center)
        # If so, loop over them, otherwise just add the single parameter and its fixedOrNot corresponding flag
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
        for num, com in zip(range(size), comments):
            if com is not None:
                allLines[num]  = ("%-" + "%d" %maxLineLen + "s") %allLines[num] + "  # " + com
            
    return firstLine + putStringsTogether(allLines)


##############################################################
#               Profiles avaiblable in galfit                #
##############################################################

def gendeVaucouleur(x=50, y=50, mag=25.0, re=10.0, bOvera=1.0, PA=0.0, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct a de Vaucouleur function configuration.

    :param float mag: (**Optional**) total integrated magnitude of the profile
    :param x: (**Optional**) X position of the de Vaucouleur profile center (in pixel)
    :type x: int or float
    :param y: (**Optional**) Y position of the de Vaucouleur profile center  (in pixel)
    :type y: int or float
    :param float re: (**Optional**) half-light (effective) radius of the profile (in pixel)
        
    :param float bOvera: (**Optional**) axis ratio b/a of the minor over major axes. Must be between 0 and 1.
    :param dict comments: (**Optional**) dictionnary which contains a comment for each line. By default, it is set to None, and default comments will be used instead.

        .. note::
            
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, 'mag' for total magnitude, 'bOvera' for b/a ratio, etc.). The key value is the comment you want.
        
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
        
        .. warning::
            
            - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'x' or 'y' as both values appear on the same line
            - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
            - to add a comment to the line with index Z, use the key name 'Zline'
            
    :param bool noComments: (**Optional**) whether to not provide any comments or not
    :param PA: (**Optional**) position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
    :type PA: int or float
    :param list fixedParams: (**Optional**) list of parameter names which must be fixed during galfit fitting routine. See :py:func:`genModel` for more information.
    :param bool skipComponentInResidual: (**Optional**) whether to not take into account this component when computing the residual or not. If False, the residual will skip this component in the best-fit model.
    
    :returns: de Vaucouleur GALFIT configuration as formatted text
    :rtype: str
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["x", "y", "mag", "re", "n", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mag", "re", "n", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("devauc",
                    [1, 3, 4, 9, 10, 'Z'],
                    [[x, y], mag, re, bOvera, PA, skipComponentInResidual],
                    [[isFixed["x"], isFixed["y"]], isFixed["mag"], isFixed["re"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[formats['posFormat'], formats['posFormat']], formats['magFormat'], formats['radiusFormat'], formats['bOveraFormat'], formats['PAFormat'], "%d"],
                    comments=[comments['object'], comments["pos"], comments["mag"], comments["re"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='de Vaucouleur function')


def genEdgeOnDisk(x=50, y=50, mu=20.0, diskScaleLength=10.0, diskScaleHeight=2.0, PA=0.0, 
                  skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    r"""
    . codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct an Edge-on disk function configuration.

    :param float diskScaleHeight: (**Optional**) disk scale-height perpendicular to the disk (in pixel)
    :param float mu: (**Optional**) central surface brightness (:math:`\rm{mag/arcsec^2}`) of the profile
    :param x: (**Optional**) X position of the de Vaucouleur profile center (in pixel)
    :type x: int or float
    :param y: (**Optional**) Y position of the de Vaucouleur profile center  (in pixel)
    :type y: int or float
    :param float diskScaleLength: major axis disk scale-length (in pixel)

    :param dict comments: (**Optional**) dictionnary which contains a comment for each line. By default, it is set to None, and default comments will be used instead.

        .. note::
            
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, 'mag' for total magnitude, 'bOvera' for b/a ratio, etc.). The key value is the comment you want.
        
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
        
        .. warning::
            
            - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'x' or 'y' as both values appear on the same line
            - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
            - to add a comment to the line with index Z, use the key name 'Zline'

    :param bool noComments: (**Optional**) whether to not provide any comments or not
    :param PA: (**Optional**) position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
    :type PA: int or float
    :param list fixedParams: (**Optional**) list of parameter names which must be fixed during galfit fitting routine. See :py:func:`genModel` for more information.
    :param bool skipComponentInResidual: (**Optional**) whether to not take into account this component when computing the residual or not. If False, the residual will skip this component in the best-fit model.
    
    :returns: Edge-on disk GALFIT configuration as formatted text
    :rtype: str
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["x", "y", "mu", "diskScaleHeight", "diskScaleLength", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mu", "diskScaleHeight", "diskScaleLength", "PA", "skipComponentInResidual"], comments)
            
    return genModel("edgedisk",
                    [1, 3, 4, 5, 10, 'Z'],
                    [[x, y], mu, diskScaleHeight, diskScaleLength, PA, skipComponentInResidual],
                    [[isFixed["x"], isFixed["y"]], isFixed["mu"], isFixed["diskScaleHeight"], isFixed["diskScaleLength"], isFixed["PA"], ""],
                    [[formats['posFormat'], formats['posFormat']], formats['muFormat'], formats['radiusFormat'], formats['radiusFormat'], formats['PAFormat'], "%d"],
                    comments=[comments['object'], comments["pos"], comments["mu"], comments["diskScaleHeight"], comments["diskScaleLength"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Edge-on disk function')


def genExpDisk(x=50, y=50, mag=25.0, rs=8.0, bOvera=1.0, PA=0.0, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct an exponential disk function configuration.
    
    :param float mag: (**Optional**) total integrated magnitude of the profile
    :param x: (**Optional**) X position of the de Vaucouleur profile center (in pixel)
    :type x: int or float
    :param y: (**Optional**) Y position of the de Vaucouleur profile center  (in pixel)
    :type y: int or float
    :param float rs: (**Optional**) disk scale-length (in pixel) such that :math:`R_{\rm{s}} = R_{\rm{e}}/1.678`, with :math:`R_{\rm{e}}` the effective radius of an equivalent n=1 Sersic profile.
    :param float bOvera: (**Optional**) axis ratio b/a of the minor over major axes (must be between 0 and 1)

    :param dict comments: (**Optional**) dictionnary which contains a comment for each line. By default, it is set to None, and default comments will be used instead.

        .. note::
            
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, 'mag' for total magnitude, 'bOvera' for b/a ratio, etc.). The key value is the comment you want.
        
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
        
        .. warning::
            
            - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'x' or 'y' as both values appear on the same line
            - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
            - to add a comment to the line with index Z, use the key name 'Zline'

    :param bool noComments: (**Optional**) whether to not provide any comments or not
    :param PA: (**Optional**) position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
    :type PA: int or float
    :param list fixedParams: (**Optional**) list of parameter names which must be fixed during galfit fitting routine. See :py:func:`genModel` for more information.
    :param bool skipComponentInResidual: (**Optional**) whether to not take into account this component when computing the residual or not. If False, the residual will skip this component in the best-fit model.
    
    :returns: Exponential disk GALFIT configuration as formatted text
    :rtype: str
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["x", "y", "mag", "rs", "n", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mag", "rs", "n", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("expdisk",
                    [1, 3, 4, 9, 10, 'Z'],
                    [[x, y], mag, rs, bOvera, PA, skipComponentInResidual],
                    [[isFixed["x"], isFixed["y"]], isFixed["mag"], isFixed["rs"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[formats['posFormat'], formats['posFormat']], formats['magFormat'], formats['radiusFormat'], formats['bOveraFormat'], formats['PAFormat'], "%d"],
                    comments=[comments['object'], comments["pos"], comments["mag"], comments["rs"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Exponential function')


def genFerrer(x=50, y=50, mu=20.0, rt=5.0, alphaFerrer=3.0, betaFerrer=2.5, bOvera=1.0, PA=0.0, 
             skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct a Ferrer function configuration.
    
    :param float mu: (**Optional**) surface brightness (:math:`\rm{mag/arcsec^2}`) at radius rb
    :param x: (**Optional**) X position of the de Vaucouleur profile center (in pixel)
    :type x: int or float
    :param y: (**Optional**) Y position of the de Vaucouleur profile center  (in pixel)
    :type y: int or float
    :param float rt: (**Optional**) outer truncation radius (in pixel)
        
    :param float alphaFerrer: (**Optional**) sharpness of the truncation
    :param float betaFerrer: (**Optional**) central slope  
    :param float bOvera: (**Optional**) axis ratio b/a of the minor over major axes (must be between 0 and 1)

    :param dict comments: (**Optional**) dictionnary which contains a comment for each line. By default, it is set to None, and default comments will be used instead.

        .. note::
            
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, 'mag' for total magnitude, 'bOvera' for b/a ratio, etc.). The key value is the comment you want.
        
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
        
        .. warning::
            
            - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'x' or 'y' as both values appear on the same line
            - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
            - to add a comment to the line with index Z, use the key name 'Zline'

    :param bool noComments: (**Optional**) whether to not provide any comments or not
    :param PA: (**Optional**) position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
    :type PA: int or float
    :param list fixedParams: (**Optional**) list of parameter names which must be fixed during galfit fitting routine. See :py:func:`genModel` for more information.
    :param bool skipComponentInResidual: (**Optional**) whether to not take into account this component when computing the residual or not. If False, the residual will skip this component in the best-fit model.
    
    :returns: Ferrer GALFIT configuration as formatted text
    :rtype: str
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["x", "y", "mu", "rt", "alphaFerrer", "betaFerrer", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mu", "rt", "alphaFerrer", "betaFerrer", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("nuker", 
                    [1, 3, 4, 5, 6, 9, 10, 'Z'],
                    [[x, y], mu, rt, alphaFerrer, betaFerrer, bOvera, PA, skipComponentInResidual],
                    [[isFixed["x"], isFixed["y"]], isFixed["mu"], isFixed["rt"], isFixed["alphaFerrer"], isFixed["betaFerrer"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[formats['posFormat'], formats['posFormatY']], formats['muFormat'], formats['radiusFormat'], formats['powerlawFormat'], formats['powerlawFormat'], formats['bOveraFormat'], formats['PAFormat'], "%d"],
                    comments=[comments['object'], comments["pos"], comments["mu"], comments["rt"], comments["alphaFerrer"], comments["betaFerrer"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Nuker function')
    
    
def genGaussian(x=50, y=50, mag=25.0, FWHM=3.0, bOvera=1.0, PA=0.0, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct a Gaussian function configuration.

    :param float FWHM: (**Optional**) full width at half maximum of the PSF (in pixel)
    :param float mag: (**Optional**) total integrated magnitude of the profile
    :param x: (**Optional**) X position of the de Vaucouleur profile center (in pixel)
    :type x: int or float
    :param y: (**Optional**) Y position of the de Vaucouleur profile center  (in pixel)
    :type y: int or float
    :param float bOvera: (**Optional**) axis ratio b/a of the minor over major axes (must be between 0 and 1)

    :param dict comments: (**Optional**) dictionnary which contains a comment for each line. By default, it is set to None, and default comments will be used instead.

        .. note::
            
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, 'mag' for total magnitude, 'bOvera' for b/a ratio, etc.). The key value is the comment you want.
        
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
        
        .. warning::
            
            - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'x' or 'y' as both values appear on the same line
            - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
            - to add a comment to the line with index Z, use the key name 'Zline'

    :param bool noComments: (**Optional**) whether to not provide any comments or not
    :param PA: (**Optional**) position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
    :type PA: int or float
    :param list fixedParams: (**Optional**) list of parameter names which must be fixed during galfit fitting routine. See :py:func:`genModel` for more information.
    :param bool skipComponentInResidual: (**Optional**) whether to not take into account this component when computing the residual or not. If False, the residual will skip this component in the best-fit model.
    
    :returns: Gaussian GALFIT configuration as formatted text
    :rtype: str
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["x", "y", "mag", "FWHM", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mag", "FWHM", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("gaussian",
                    [1, 3, 4, 9, 10, 'Z'],
                    [[x, y], mag, FWHM, bOvera, PA, skipComponentInResidual],
                    [[isFixed["x"], isFixed["y"]], isFixed["mag"], isFixed["FWHM"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[formats['posFormat'], formats['posFormat']], formats['magFormat'], formats['radiusFormat'], formats['bOveraFormat'], formats['PAFormat'], "%d"],
                    comments=[comments['object'], comments["pos"], comments["mag"], comments["FWHM"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Gaussian function')
    
    
def genKing(x=50, y=50, mu0=20.0, rc=3.0, rt=30.0, powerlaw=2.0, bOvera=1.0, PA=0.0, 
            skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct an empirical King profile configuration.

    :param float mu0: (**Optional**) central surface brightness (:math:`\rm{mag/arcsec^2}`)
    :param float powerlaw: (**Optional**) powerlaw (powerlaw=2.0 for a standard King profile)
    :param x: (**Optional**) X position of the de Vaucouleur profile center (in pixel)
    :type x: int or float
    :param y: (**Optional**) Y position of the de Vaucouleur profile center  (in pixel)
    :type y: int or float
    :param float rc: (**Optional**) core radius (in pixel)
    :param float rt: (**Optional**) truncation radius (in pixel)
    :param float bOvera: (**Optional**) axis ratio b/a of the minor over major axes (must be between 0 and 1)

    :param dict comments: (**Optional**) dictionnary which contains a comment for each line. By default, it is set to None, and default comments will be used instead.

        .. note::
            
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, 'mag' for total magnitude, 'bOvera' for b/a ratio, etc.). The key value is the comment you want.
        
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
        
        .. warning::
            
            - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'x' or 'y' as both values appear on the same line
            - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
            - to add a comment to the line with index Z, use the key name 'Zline'

    :param bool noComments: (**Optional**) whether to not provide any comments or not
    :param PA: (**Optional**) position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
    :type PA: int or float
    :param list fixedParams: (**Optional**) list of parameter names which must be fixed during galfit fitting routine. See :py:func:`genModel` for more information.
    :param bool skipComponentInResidual: (**Optional**) whether to not take into account this component when computing the residual or not. If False, the residual will skip this component in the best-fit model.
    
    :returns: King profile GALFIT configuration as formatted text
    :rtype: str
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["x", "y", "mu0", "rc", "rt", "powerlaw", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mu0", "rc", "rt", "powerlaw", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("king", 
                    [1, 3, 4, 5, 6, 9, 10, 'Z'],
                    [[x, y], mu0, rc, rt, powerlaw, bOvera, PA, skipComponentInResidual],
                    [[isFixed["x"], isFixed["y"]], isFixed["mu0"], isFixed["rc"], isFixed["rt"], isFixed["powerlaw"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[formats['posFormat'], formats['posFormat']], formats['muFormat'], formats['radiusFormat'], formats['radiusFormat'], formats['powerlawFormat'], formats['bOveraFormat'], formats['PAFormat'], "%d"],
                    comments=[comments['object'], comments["pos"], comments["mu0"], comments["rc"], comments["rt"], comments["powerlaw"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='The Empirical King Profile')

    
def genMoffat(x=50, y=50, mag=25.0, FWHM=3.0, powerlaw=1.0, bOvera=1.0, PA=0.0, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct a de Vaucouleur function configuration.
    
    :param float FWHM: (**Optional**) full width at half maximum of the PSF (in pixel)
    :param float mag: (**Optional**) total integrated magnitude of the profile
    :param float powerlaw: (**Optional**) powerlaw
    :param x: (**Optional**) X position of the de Vaucouleur profile center (in pixel)
    :type x: int or float
    :param y: (**Optional**) Y position of the de Vaucouleur profile center  (in pixel)
    :type y: int or float
    :param float bOvera: (**Optional**) axis ratio b/a of the minor over major axes (must be between 0 and 1)

    :param dict comments: (**Optional**) dictionnary which contains a comment for each line. By default, it is set to None, and default comments will be used instead.

        .. note::
            
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, 'mag' for total magnitude, 'bOvera' for b/a ratio, etc.). The key value is the comment you want.
        
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
        
        .. warning::
            
            - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'x' or 'y' as both values appear on the same line
            - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
            - to add a comment to the line with index Z, use the key name 'Zline'

    :param bool noComments: (**Optional**) whether to not provide any comments or not
    :param PA: (**Optional**) position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
    :type PA: int or float
    :param list fixedParams: (**Optional**) list of parameter names which must be fixed during galfit fitting routine. See :py:func:`genModel` for more information.
    :param bool skipComponentInResidual: (**Optional**) whether to not take into account this component when computing the residual or not. If False, the residual will skip this component in the best-fit model.
    
    Returns a complete de Vaucouleur function galfit configuration as formatted text.
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["x", "y", "mag", "FWHM", "powerlaw", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mag", "FWHM", "powerlaw", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("moffat",
                    [1, 3, 4, 5, 9, 10, 'Z'],
                    [[x, y], mag, FWHM, powerlaw, bOvera, PA, skipComponentInResidual],
                    [[isFixed["x"], isFixed["y"]], isFixed["mag"], isFixed["FWHM"], isFixed["powerlaw"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[formats['posFormat'], formats['posFormat']], formats['magFormat'], formats['radiusFormat'], formats['powerlawFormat'], formats['bOveraFormat'], formats['PAFormat'], "%d"],
                    comments=[comments['object'], comments["pos"], comments["mag"], comments["FWHM"], comments["powerlaw"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Moffat function')
    
    
def genNuker(x=50, y=50, mu=20.0, rb=10.0, alpha=1.0, beta=0.5, gamma=0.7, bOvera=1.0, PA=0.0, 
             skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct a Nuker function configuration.
    
    :param float mu: (**Optional**) surface brightness (:math:`\rm{mag/arcsec^2}`) at radius rb
    :param x: (**Optional**) X position of the de Vaucouleur profile center (in pixel)
    :type x: int or float
    :param y: (**Optional**) Y position of the de Vaucouleur profile center  (in pixel)
    :type y: int or float
    :param float rb: (**Optional**) break radius (in pixel) where the slope is the average between :math:`\beta` and :math:`\gamma`, and where the maximum curvature (in log space) is reached. It roughly corresponds to the radius of transition between the inner and outer powerlaws.
        
    :param float alpha: (**Optional**) sharpness of transition between the inner part and the outer part
    :param float beta: (**Optional**) outer powerlaw slope
    :param float bOvera: (**Optional**) axis ratio b/a of the minor over major axes (must be between 0 and 1)

    :param dict comments: (**Optional**) dictionnary which contains a comment for each line. By default, it is set to None, and default comments will be used instead.

        .. note::
            
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, 'mag' for total magnitude, 'bOvera' for b/a ratio, etc.). The key value is the comment you want.
        
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
        
        .. warning::
            
            - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'x' or 'y' as both values appear on the same line
            - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
            - to add a comment to the line with index Z, use the key name 'Zline'

    :param float gamma: (**Optional**) inner powerlaw slope
    :param bool noComments: (**Optional**) whether to not provide any comments or not
    :param PA: (**Optional**) position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
    :type PA: int or float
    :param list fixedParams: (**Optional**) list of parameter names which must be fixed during galfit fitting routine. See :py:func:`genModel` for more information.
    :param bool skipComponentInResidual: (**Optional**) whether to not take into account this component when computing the residual or not. If False, the residual will skip this component in the best-fit model.
    
   :returns: Nuker GALFIT configuration as formatted text
   :rtype: str
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["x", "y", "mu", "rb", "alpha", "beta", "gamma", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mu", "rb", "alpha", "beta", "gamma", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("nuker", 
                    [1, 3, 4, 5, 6, 7, 9, 10, 'Z'],
                    [[x, y], mu, rb, alpha, beta, gamma, bOvera, PA, skipComponentInResidual],
                    [[isFixed["x"], isFixed["y"]], isFixed["mu"], isFixed["rb"], isFixed["alpha"], isFixed["beta"], isFixed["gamma"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[formats['posFormat'], formats['posFormat']], formats['muFormat'], formats['radiusFormat'], formats['powerlawFormat'], formats['powerlawFormat'], formats['powerlawFormat'], formats['bOveraFormat'], formats['PAFormat'], "%d"],
                    comments=[comments['object'], comments["pos"], comments["mu"], comments["rb"], comments["alpha"], comments["beta"], comments["gamma"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Nuker function')


def genPSF(x=50, y=50, mag=25.0, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct PSF function configuration. 
    
    .. note::
        
        This PSF is technically not a function, but uses the psf image provided in the header to fit a given point source.
    
    :param float mag: (**Optional**) total integrated magnitude of the profile
    :param float powerlaw: (**Optional**) powerlaw
    :param x: (**Optional**) X position of the de Vaucouleur profile center (in pixel)
    :type x: int or float
    :param y: (**Optional**) Y position of the de Vaucouleur profile center  (in pixel)
    :type y: int or float
    
    :param dict comments: (**Optional**) dictionnary which contains a comment for each line. By default, it is set to None, and default comments will be used instead.

        .. note::
            
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, 'mag' for total magnitude, 'bOvera' for b/a ratio, etc.). The key value is the comment you want.
        
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
        
        .. warning::
            
            - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'x' or 'y' as both values appear on the same line
            - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
            - to add a comment to the line with index Z, use the key name 'Zline'

    :param bool noComments: (**Optional**) whether to not provide any comments or not
    :param list fixedParams: (**Optional**) list of parameter names which must be fixed during galfit fitting routine. See :py:func:`genModel` for more information.
    :param bool skipComponentInResidual: (**Optional**) whether to not take into account this component when computing the residual or not. If False, the residual will skip this component in the best-fit model.
    
    Returns a complete PSF function galfit configuration as formatted text.
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["x", "y", "mag"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mag", "skipComponentInResidual"], comments)
            
    return genModel("psf",
                    [1, 3, 'Z'],
                    [[x, y], mag, skipComponentInResidual],
                    [[isFixed["x"], isFixed["y"]], isFixed["mag"], ""],
                    [[formats['posFormat'], formats['posFormat']], formats['magFormat'], "%d"],
                    comments=[comments['object'], comments["pos"], comments["mag"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='PSF fit')


def genSersic(x=50, y=50, mag=25.0, re=10.0, n=4, bOvera=1.0, PA=0.0, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct a Sersic function configuration.
    
    :param float mag: (**Optional**) total integrated magnitude of the profile
    :param float powerlaw: (**Optional**) powerlaw
    :param x: (**Optional**) X position of the de Vaucouleur profile center (in pixel)
    :type x: int or float
    :param y: (**Optional**) Y position of the de Vaucouleur profile center  (in pixel)
    :type y: int or float
    :param float re: half-light (effective) radius of the profile (in pixel)
    :param float bOvera: (**Optional**) axis ratio b/a of the minor over major axes (must be between 0 and 1)

    :param dict comments: (**Optional**) dictionnary which contains a comment for each line. By default, it is set to None, and default comments will be used instead.

        .. note::
            
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, 'mag' for total magnitude, 'bOvera' for b/a ratio, etc.). The key value is the comment you want.
        
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
        
        .. warning::
            
            - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'x' or 'y' as both values appear on the same line
            - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
            - to add a comment to the line with index Z, use the key name 'Zline'

    :param n: Sersic index
    :type n: int or float
    :param bool noComments: (**Optional**) whether to not provide any comments or not
    :param PA: (**Optional**) position angle of the morphological major axis on the sky (in degrees). Up is 0° and Left is 90°.
    :type PA: int or float
    :param list fixedParams: (**Optional**) list of parameter names which must be fixed during galfit fitting routine. See :py:func:`genModel` for more information.
    :param bool skipComponentInResidual: (**Optional**) whether to not take into account this component when computing the residual or not. If False, the residual will skip this component in the best-fit model.
    
    :returns: Sersic GALFT configuration as formatted text
    :rtype: str
    """
    
    # isFixed is a dictionnary with correct value for fixing parameters in galfit fit
    isFixed  = createIsFixedDict(["x", "y", "mag", "re", "n", "bOvera", "PA"], fixedParams)
    comments = createCommentsDict(["object", "pos", "mag", "re", "n", "bOvera", "PA", "skipComponentInResidual"], comments)
            
    return genModel("sersic", 
                    [1, 3, 4, 5, 9, 10, 'Z'],
                    [[x, y], mag, re, n, bOvera, PA, skipComponentInResidual],
                    [[isFixed["x"], isFixed["y"]], isFixed["mag"], isFixed["re"], isFixed["n"], isFixed["bOvera"], isFixed["PA"], ""],
                    [[formats['posFormat'], formats['posFormat']], formats['magFormat'], formats['radiusFormat'], formats['nFormat'], formats['bOveraFormat'], formats['PAFormat'], "%d"],
                    comments=[comments['object'], comments["pos"], comments["mag"], comments["re"], comments["n"], comments["bOvera"], comments["PA"], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='Sersic function')


def genSky(background=1.0, xGradient=0.0, yGradient=0.0, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct a sky function configuration.

    :param float background: (**Optional**) sky background  (in ADU counts)
    :param float xGradient: (**Optional**) sky gradient along x, :math:`d{\rm{sky}}/dx` (in ADU/pixel)
    :param float yGradient: (**Optional**) sky gradient along y, :math:`d{\rm{sky}}/dy` (in ADU/pixel)
        
    :param dict comments: (**Optional**) dictionnary which contains a comment for each line. By default, it is set to None, and default comments will be used instead.

        .. note::
            
            In general, the dictionnary key name is the parameter name of the galfit configuration line (ex: 'pos' for position, 'mag' for total magnitude, 'bOvera' for b/a ratio, etc.). The key value is the comment you want.
        
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.
        
        .. warning::
            
            - for the POSITION, USE THE KEY 'pos' INSTEAD OF 'x' or 'y' as both values appear on the same line
            - you can also provide a comment for the 0th line (i.e. the model name). To do so, use the key name 'object'.
            - to add a comment to the line with index Z, use the key name 'Zline'

    :param bool noComments: (**Optional**) whether to not provide any comments or not
    :param list fixedParams: (**Optional**) list of parameter names which must be fixed during galfit fitting routine. See :py:func:`genModel` for more information.
    :param bool skipComponentInResidual: (**Optional**) whether to not take into account this component when computing the residual or not. If False, the residual will skip this component in the best-fit model.
    
    Return a complete sky function galfit configuration as formatted text.
    """
    
    isFixed  = createIsFixedDict(["background", "xGradient", "yGradient"], fixedParams)
    comments = createCommentsDict(["object", "background", "xGradient", "yGradient", "skipComponentInResidual"], comments)
            
    return genModel("sky",
                    [1, 2, 3, 'Z'],
                    [background, xGradient, yGradient, skipComponentInResidual],
                    [isFixed['background'], isFixed["xGradient"], isFixed['yGradient'], ""],
                    [formats['backgroundFormat'], formats['GradientFormat'], formats['GradientFormat'], "%d"],
                    comments=[comments['object'], comments["background"], comments["xGradient"], comments['yGradient'], comments['skipComponentInResidual']], 
                    noComments=noComments,
                    mainComment='sky')
    

######################################################################
#                   Additional galfit tag functions                  #
######################################################################
    
def bendingModes(listModes=[1], listModesValues=[0.5], fixedParams=[], comments=None, noComments=False):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct bending modes GALFIT configuration.

    :param list[int] listModes: (**Optional**) mode number
    :param list[float] listModesValues: (**Optional**) value of the corresponding mode
            
    :param dict comments: (**Optional**) dictionary which contains a comment for each line. 
    
        .. note::
            
            By default, **comments** is set to None, and default comments will be used instead.
            
            The dictionnary key names are the mode number. For instance
            
            * 1 for bending mode 1
            * 2 for bending mode 2, and so on...
            
            The key value is the comment you want.
            
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.

    :param bool noComments: (**Optional**) whether to not provide any comments or not
    :param list fixedParams: (**Optional**) list of parameter names which must be fixed during galfit fitting routine. See :py:func:`genModel` for more information.
    
    :returns: bending modes GALFIT configuration as formatted text
    :rtype: str
    """
    
    isFixed        = [0 if i in fixedParams else 1 for i in listModes]
    
    if comments is None:
        comments   = ["Bending mode %d" %i for i in listModes]
    else:
        comment    = createCommentsDict(listModes, comments)
        comments   = [comment[i] for i in listModes]
    
    return genModel(None, listModes, listModesValues, isFixed, [formats['bendingFormat']]*len(listModes), 
                    comments=comments, noComments=noComments, removeLine0=True, prefix="B")
    
    
def boxy_diskyness(value=0.5, isFixed=False, comment=None):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct a boxy/diskyness (genralised ellipses) GALFIT configuration.
    
    :parma bool isFixed: (**Optional**) whether to fix the parameter during galfit fitting routine
    :param float value: value of the diskyness/boxyness parameter (negative = more disky, positive = more boxy)
    :param str comment: comment to append at the end of the line
            
    :returns: boxyness/diskyness GALFIT configuration
    :rtype: str
    """
    
    if comment is None:
        comment = "traditional diskyness(-)/boxyness(+)"
        
    return genModel(None, [0], [value], [int(not isFixed)], [formats['boxy_diskyFormat']], comments=[comment], noComments=False, removeLine0=True, prefix="c")
    
    
def fourierModes(listModes=[1], listModesAmplitudes=[1.0], listModesPhases=[0.0], fixedParams=[], comments=None, noComments=False):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Construct Azimuthal fourier modes GALFIT configuration.

    :param list[int] listModes: mode number
    :param list[float] listModesAmplitudes: amplitudes of the corresponding mode
    :param list[float] listModesPhases: phases of the corresponding mode
            
    :param dict comments: dictionary which contains a comment for each line. By default, comments is set to None, and default comments will be used instead.
    
        .. note::
            
            The dictionnary key names are the modes number followed by A for amplitude and P for phase. For instance 
            
            * '1A' for the amplitude of the fourier mode 1
            * '2P' for the phase of Fourier mode 2, and so on...
            
            The key value is the comment you want.
            
            You only need to provide comments for the parameters you want. Unprovided key names will result in no comment given to the line.

    :param list[str] fixedParams: list of Fourier modes amplitudes and/or phases which must be fixed in GALFIT. 
    
        .. rubric:: **Example**
        
        Say one wants to fix the amplitude of the Fourier mode 1 and the phase of Fourier mode 2, then one may write 
        
        >>> fixedParams=['1A', '2P']

    :param bool noComments: whether to provide no comments or not
    
    :returns: Azimuthal fourier modes GALFIT configuration as formatted text
    :rtype: str
    """
    
    listModesValues       = []
    for amp, ph in zip(listModesAmplitudes, listModesPhases):
        listModesValues.append([amp, ph])
    
    # we use the behaviour that True is 1 and False is 0 in python
    isFixed = [[int(('%dA' %i) not in fixedParams), int(('%dP' %i) not in fixedParams)] for i in listModes]
    
    if comments is None:
        comments          = ["Az. Fourier mode %d, amplitude and phase angle" %i for i in listModes]
    else:
        possibleModesKeys = ravel([["%dA" %i, "%dP" %i] for i in listModes])
        comment           = createCommentsDict(possibleModesKeys, comments)
        comments          = [[comment["%dA" %i], comment["%dP" %i]] for i in listModes]
    
    return genModel(None, listModes, listModesValues, isFixed, [[formats['ampFourierFormat'], formats['phaseFourierFormat']]]*len(listModes), 
                    comments=comments, noComments=noComments, removeLine0=True, prefix="F")