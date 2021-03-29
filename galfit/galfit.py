#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Functions related to automating galfit modelling.
"""

from   .io                              import errorMessage, okMessage, brightMessage, dimMessage
from   .models                          import gendeVaucouleur, genEdgeOnDisk, genExpDisk, genFerrer, genGaussian, genKing, genMoffat, genNuker, genPSF, genSersic, genSky, bendingModes, boxy_diskyness, fourierModes
from   wilfried.utilities.dictionaries  import checkDictKeys, removeKeys, setDict
from   wilfried.utilities.strings       import putStringsTogether, toStr, maxStringsLen
from   wilfried.utilities.plotUtilities import genMeThatPDF
from   os                               import mkdir
import subprocess                       as     sub
from   numpy                            import unique, array
import os.path                          as     opath
import multiprocessing
import colorama                         as     col
col.init()



##########################################
#          Global variables              #
##########################################

# keys allowed per model. If you change a parameter name in a function declaration, you must also change the name here
fullKeys = {'header': {'parameters':['outputImage', 'xmin', 'xmax', 'ymin', 'ymax', 'inputImage', 'sigmaImage', 'psfImage', 'maskImage', 'couplingFile', 'psfSamplingFactor', 'zeroPointMag', 'arcsecPerPixel', 'sizeConvX', 'sizeConvY', 'displayType', 'option'], 
                       'default':['output.fits', 0, 100, 0, 100, "none", "none", "none", "none", "none", 1, 30.0, [0.03, 0.03], None, None, "regular", 0]},
                                   
            'deVaucouleur': {'parameters':['x', 'y', 'mag', 're', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                             'default':[50, 50, 25.0, 10.0, 1.0, 0.0, False, [], None, False]},
                                         
            'edgeOnDisk': {'parameters':['x', 'y', 'mu', 'diskScaleLength', 'diskScaleHeight', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                           'default':[50, 50, 20.0, 10.0, 2.0, 0.0, False, [], None, False]},
                                       
            'expDisk': {'parameters':['x', 'y', 'mag', 'rs', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                        'default':[50, 50, 25.0, 8.0, 1.0, 0.0, False, [], None, False]},
                                    
            'ferrer': {'parameters':['x', 'y', 'mu', 'rt', 'alphaFerrer', 'betaFerrer', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                       'default':[50, 50, 20.0, 5.0, 3.0, 2.5, 1.0, 0.0, False, [], None, False]},
                                   
            'gaussian': {'parameters':['x', 'y', 'mag', 'FWHM', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                         'default':[50, 50, 25.0, 3.0, 1.0, 0.0, False, [], None, False]},

            'king': {'parameters':['x', 'y', 'mu0', 'rc', 'rt', 'powerlaw', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                     'default':[50, 50, 20.0, 3.0, 30.0, 2.0, 1.0, 0.0, False, [], None, False]},

            'moffat': {'parameters':['x', 'y', 'mag', 'FWHM', 'powerlaw', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                       'default':[50, 50, 25.0, 3.0, 1.0, 1.0, 0.0, False, [], None, False]},

            'nuker': {'parameters':['x', 'y', 'mu', 'rb', 'alpha', 'beta', 'gamma', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                      'default':[50, 50, 20.0, 10.0, 1.0, 0.5, 0.7, 1.0, 0.0, False, [], None, False]},

            'psf': {'parameters':['x', 'y', 'mag', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                    'default':[50, 50, 25.0, False, [], None, False]},
                    
            'sersic': {'parameters':['x', 'y', 'mag', 're', 'n', 'bOvera', 'PA', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                       'default':[50, 50, 25.0, 10.0, 4, 1.0, 0.0, False, [], None, False]},
                       
            'sky': {'parameters':['background', 'xGradient', 'yGradient', 'skipComponentInResidual', 'fixedParams', 'comments', 'noComments'], 
                    'default':[1.0, 0.0, 0.0, False, [], None, False]}
            }
            
# Every unique parameter
everyParam = []
for i in [fullKeys[name]['parameters'] for name in fullKeys]:
    everyParam += i
everyParam = unique(everyParam)
            
#: Recognised models
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

#: Additional tag functions
tags = ['fourier', 'bending', 'boxyness']

#: Default pdf viewer
myPDFViewer = 'okular'


def run_galfit(feedmeFiles, header={}, listProfiles=[], inputNames=[], outputNames=[], constraintNames=[],
               pathFeedme="./feedme/", pathIn="./inputs/", pathOut="./outputs/", pathConstraints="./constraints/", pathLog="./log/", pathRecap='./recap/',
               constraints=None, forceConfig=False, noGalfit=False, noPDF=False, showRecapFiles=True, showLog=False,
               divergingNorm=True,
               numThreads=8):
    """
    
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Run GALFIT given some models and a header. If necessary, it creates GALFIT config files. 
    
    If the FEEDME and CONSTRAINT files already exist, just use the following command to run GALFIT on all the galaxies
    
    >>> run_galfit(yourList)
    
    where yourList is a list of FEEDME files.
    
    .. note ::
    
        If at least one FEEDME file does not exist, the following information must be provided
        
        * a header
        * a list of profiles (models) 
        * input (HST) file names
        
        **See** :py:func:`writeConfigs` **function for more information.**
    
        Using the correct optional arguments, one can **skip** the following steps:
            
        * FEEDME and CONSTRAINT files creation by setting **forceConfig** to False. This is done by default in order to avoid to destroy already existing input files.
        * GALFIT modelling by setting **noGalfit** to True. This can be useful if one only wants to make the pdf recap files when the modelling has already been done.
        * generating pdf recap files at the end of the program by setting **noPDF** to True
        * opening of the recap files at the end of the program by setting **showRecapFiles** to False
        * showing the content of the LOG file once the fit is complete by setting **showLog** to False. By default, LOG files are not shown.

    :param list[str] feedmeFiles: name of the GALFIT FILES to run galfit onto
                
    :param bool divergingNorm: (**Optional**) whether to use a diverging norm in the output pdf figure or not
    :param list[dict] constraints: (**Optional**)
        
        list of dictionaries used to generate the constraints. If None, no constraints are applied. 
    
        Each dictionary must contain three keys, namely 'components', 'parameter' and 'constraint' (see below for use cases):
            
        * 'constraint' : dict with keys 'type' and 'value'
            
            This sets the type of constraint one wants and its range. The possible 'type' of constraints are:
                
            * 'offset' to set the value of some parameter between different profiles relative to one another (based on the initial values provided in the FEEDME file)
            * 'ratio' to fix the ratio of some parameter between different profiles (based on the initial values provided in the FEEDME file)
            * 'absoluteRange' to set an asbolute range of values for some parameter
            * 'relativeRange' to set a range of possible values around the initial value given in the FEEDME file
            * 'componentWiseRange' to set a range of possible values around the initial value of the same parameter but from another component
            * 'componentWiseRatio' to set a range of possible values for the ratio of the same parameter in two components
        
            And the corresponding values are:
            
            * single value for 'offset' and 'ratio' types
            * a list of two floats to define bounds for the other types
    
        * 'components' : int or list[int]
        
            The number(s) associated to the model(s) one wants to constrain:
            
            * for 'offset' and 'ratio' types a list of an indefinite number of int can be given
            * for both relative and absolute ranges, a single profile number (int) must be given
            * for component wise ranges or ratios, a list of two numbers (int) must be provided
            
        * 'parameter' : str
        
            Name of the parameter to contrain.
        
        .. rubric:: **Example**
                
        Let us assume we want to model two Sersic components such that we have :samp:`listProfiles=[sersic1, sersic2]` where
        
        >>> sersic1 = {'name':'sersic', 'x':81, 'y':60, 'n':1, 'fixedParams':['n'],                 're':10, 'mag':25, 'PA':40}
        >>> sersic2 = {'name':'sersic', 'x':81, 'y':60, 'n':4, 'fixedParams':['n', 'PA', 'bOvera'], 're':2,  'mag':25}
        
        Assume we want to constrain the X coordinate of :samp:`sersic1` and :samp:`sersic2` to always be similar, then we can do
        
        >>> c1 = {'components':[1, 2], 'parameter':'x', 'constraint':{'type':'offset', 'value':0.0}}
        
        Similarly, for the Y coordinate, we can do
        
        >>> c2 = {'components':[1, 2], 'parameter':'y', 'constraint':{'type':'offset', 'value':0.0}}
        
        In both cases, the number 1 refers to the first model appearing in listProfiles (i.e. :samp:`sersic1`), and the number 2 to :samp:`sersic2`.
        
        Additionnaly, let us say we want to force the X and Y coordinates to be allowed to change by :math:`\pm 10` relative to the initial value. This can be done with
        
        >>> c3 = {'components':1, 'parameter':'x', 'constraint':{'type':'relativeRange', 'value':[-10, 10]}}
        >>> c4 = {'components':1, 'parameter':'y', 'constraint':{'type':'relativeRange', 'value':[-10, 10]}}
        
        Now, it still remains to build the constraint list, so that we write
        
        >>> constraint = [c1, c2, c3, c4]
        
        And we provide :samp:`constraints=constraint` when calling :func:`run_galfit`.
            
    :param bool forceConfig: (**Optional**) whether to make all the configuration files no matter if they exist or not
    :param dict header: (**Optional**) dictionnary with key names corresponding to input parameter names in :py:func:`genHeader` function
        
        .. note::
            
            You do not need to provide an input and an output image files as this is given with the inputNames keyword.
        
    :param list[str] inputNames: (**Optional**) list of galaxies input .fits files
    
    :param list[dict] listProfiles: (**Optional**) list of profiles (models). Each dictionnary corresponds to a profile:
    
        * A key 'name' is mandatory to indentify the profile. Its value is one of the following:
            
            * 'deVaucouleur' for a de Vaucouleur profile
            * 'edgeOnDisk' for an edge-on (razor-thin) disk
            * 'expDisk' for an exponential disk
            * 'ferrer' for a Ferrer profile
            * 'gaussian' for a Gaussian 
            * 'king' for a King profile
            * 'moffat' for a Moffat profile
            * 'nuker' for a Nuker profile
            * 'psf'
            * 'sersic' for a Sersic profile
            * 'sky' for a sky gradient
            
        * Other keys correspond to the parameters corresponding to each profile. **See each profile description, to know which key to use**. Only mandatory parameters can be provided as keys if the default values seem fine to you
        
        .. note::

            You can also add fourier modes, bending modes and/or a boxyness-diskyness parameter to each profile. To do so, provide one of the following keys:
            
            * 'fourier'
            * 'bending'
            * 'boxyness'
                
            These keys must also contain dictionnaries whose keys are the parameters of the functions :py:func:`fourierModes`, :py:func:`bendingModes` and :py:func:`boxy_diskyness`.
            
        .. rubric:: **Example**
                       
        Let us assume we want to model two Sersic components such that we have :samp:`listProfiles=[sersic1, sersic2]`, where :samp:`sersic1` is an exponential disk (Sersic with n=1), and :samp:`sersic2` is a de Vaucouleurs bulge (Sersic n=4) with fixed axis ratio and PA.
        
        If we look at the function declaration for the :py:func:`genSersic` function we find the following parameters
        
        .. code::
            
            def genSersic(x=50, y=50, mag=25.0, re=10.0, n=4, bOvera=1.0, PA=0.0, skipComponentInResidual=False, fixedParams=[], comments=None, noComments=False):
                       
        For the exponential disk, we only need the Sersic index to be fixed, and we can plug typical values for the other parameters
        
        >>> sersic1 = {'name':'sersic', 'x':81, 'y':60, 'n':1, 'fixedParams':['n'], 're':10, 'mag':25, 'PA':40}
        
        For the de Vaucouleur bulge, we use the same centre, a slightly smaller bulge, and we also fix the :samp:`PA` and :samp:`bOvera`
        
        >>> sersic2 = {'name':'sersic', 'x':81, 'y':60, 'n':4, 'fixedParams':['n', 'PA', 'bOvera'], 're':2, 'mag':25}
        
        We do not provide the :samp:`'bOvera'` and :samp:`'PA'` values because default values are :samp:`bOvera=1` and :samp:`PA=0` which is perfect for this example.
            
                
    :param bool noGalfit: (**Optional**) whether to not run galfit or not
    :param bool noPDF: (**Optional**) whether to make the pdf recap files or not
    :param int numThreads: (**Optional**) number of threads used to parallelise
    :param list[str] outputNames: (**Optional**) list of galaxies output .fits files. If not provided, the output files will have the same name as the input ones with _out appended before the .fits extension.
    :param str pathFeedme: (**Optional**) location of the feedme files relative to the current folder or as an absolute path
    :param str pathIn: (**Optional**) location of the input files relative to the current folder or as an absolute path
    :param str pathLog: (**Optional**) location of the log files relative to the current folder or as an absolute path
    :param str pathOut: (**Optional**) location of the output files relative to the current folder or as an absolute path
    :param str pathRecap: (**Optional**) location of the pdf recap files relative to the current folder or as an absolute path
    :param bool showLog: (**Optional**) whether to show log files or not
    :param bool showRecapFiles: (**Optional**) whether to show the recap files made at the end or not
    
    :raises TypeError: if feedmeFiles is not a list
    """
    
    # Enabling colored text
    col.reinit()
    
    #################################################
    #                Local functions                #
    #################################################
    
    def singleGalfit(name, all_procs):
        '''Run galfit for a single image.'''
        
        try:
            # Get galfit output into a variable
            text = sub.check_output(['galfit', opath.join(pathFeedme, name)])
            
            # Only keep the relevant part in the ouput
            text = 'Iteration ' + text.decode('utf8').split('Iteration')[-1].split('COUNTDOWN')[0].rsplit('\n', maxsplit=1)[0]
            
            # Save content into a log file
            log  = opath.join(pathLog, name.replace('.feedme', '.log'))
            with open(log, 'w') as f:
                f.write(text)
            
            print(okMessage('Object %s done ' %name.strip('.feedme')) + '[~%.2f%%]'  %(len(all_procs)/total_num*100))
            
        except sub.CalledProcessError:
            print(errorMessage('Galfit failed:') + ' Probably a segmentation fault caused by a wrong constraint name or value. Try running galfit by hand on one of the feedme files to find out what went wrong.')
            
        semaphore.release()
        return 
    
    def pdf(i, maxImages, splt, ll):
        '''Generate a single pdf file.'''
        
        print("Making pdf file recap%d.pdf" %i)
        mini         = (i-1)*maxImages
        if i == splt or splt == 0:
            maxi     = ll
        else:
            maxi     = i*maxImages
        
        try:
            genMeThatPDF(outputFiles[mini:maxi], opath.join(pathRecap, 'recap%d.pdf' %i), log=False, diverging=divergingNorm)
            print(okMessage("Recap file number %d made." %i))
        except:
            pass
        semaphore.release()
        return
    
    
    ##########################################
    #            Checking section            #
    ##########################################
    
    # Check input data type is okay
    if type(feedmeFiles) is not list:
        raise TypeError(errorMessage('Given feedmeFiles variable is not a list') + '. Please provide a list of galfit .feedme file names. Cheers !')
        
    if not isinstance(numThreads, int) or numThreads<1:
        print(errorMessage('Incorrect given number of threads: %s.' %numThreads) + ' Setting default value to ' + brightMessage('1'))
        numThreads        = 1
    
    # Find where given .feedme files do not exist
    notExists             = [not opath.isfile(pathFeedme + fname) for fname in feedmeFiles]
            
    
    ##################################################
    #          Configuration files creation          #
    ##################################################
    
    if forceConfig or any(notExists):
        
        if forceConfig:
            notExists     = [True]*len(feedmeFiles)
        else:
            answer        = input(brightMessage('Some .feedme files do not exist yet. Do you want to generate automatically the files using the provided input parameters ? [N] or Y: ')).lower()
            
        if forceConfig or answer in ['y', 'yes']:
            print("Making configuration files")
            try:
                # If empty lists are given, do not apply the mask
                if len(outputNames)>0:
                    out   = list(array(outputNames)[notExists])
                else:
                    out   = []
                if len(constraintNames)>0:
                    con   = list(array(constraintNames)[notExists])
                else:
                    con   = []
        
                writeConfigs(header, listProfiles, inputNames=list(array(inputNames)[notExists]), outputNames=out, constraintNames=con, feedmeNames=list(array(feedmeFiles)[notExists]),
                             constraints=constraints, pathFeedme=pathFeedme, pathIn=pathIn, pathOut=pathOut, pathConstraints=pathConstraints)
                    
                notExists = [False]*len(notExists)
            except Exception as e:
                print(errorMessage("An Error occured during galfit configuration file creation.\n"))
                raise e
                
                
    ##########################################################################
    #                      Run galfit using multi processes                  #
    ##########################################################################
    
    if not noGalfit:
        print(brightMessage("Running galfit using multiprocesses") + ' (%d threads used)' %numThreads)
        
        # Make log directory if it does not exist yet
        if not opath.isdir(pathLog):
            mkdir(pathLog)
        
        total_feedmes = array(feedmeFiles)[[not i for i in notExists]]
        total_num     = len(total_feedmes)
        semaphore     = multiprocessing.Semaphore(numThreads)
        all_procs     = []
        for num, fname in enumerate(total_feedmes):
            semaphore.acquire()
            proc      = multiprocessing.Process(name=fname, target=singleGalfit, args=(fname, all_procs))
            all_procs.append(proc)
            proc.start()
            
        for p in all_procs:
            p.join()
    
        # Move the galfit files to log directory as well
        sub.run(['mv galfit.[0-9]* %s' %pathLog], shell=True)
    
    #############################################################
    #                 Generate pdf recap files                  #
    #############################################################
    
    # Matplotlib forces pdf images to have a maximum size. This corresponds to ~100 lines of plots given the plot size taken
    outputFiles      = [opath.join(pathOut, i) for i in outputNames]
    ll               = len(outputFiles)
    maxImages        = 100
    splt             = ll // maxImages
    
    # If there are less figures than the maximum allowed, we want a single pdf file
    if splt < 1:
        splt         = 1
    
    if not noPDF:
        semaphore        = multiprocessing.Semaphore(numThreads)
        all_procs        = []
        print('Making pdf recap files')
        for i in range(1, splt+1):                
            semaphore.acquire()
            proc         = multiprocessing.Process(name=i, target=pdf, args=(i, maxImages, splt, ll))
            all_procs.append(proc)
            proc.start()        
            
            for p in all_procs:
                p.join()    
    
    
    #############################################
    #               Show log file               #
    #############################################
    
    if showLog:
        print("Showing log files")
        
        logFiles = ['%s%s' %(pathLog, name.replace('.feedme', '.log')) for name in feedmeFiles]
        for log in logFiles:
            print(log)
            with open(log, 'r') as f:
                print(f.read(), '\n')   
        
        
    #######################################################
    #                   Show recap files                  # 
    #######################################################
                
    def askAgain(recapName, showTips=True):
        global myPDFViewer
        
        myPDFViewer = input(errorMessage('Opening failed:') + ' Probably cannot find given pdf viewer ' + brightMessage(myPDFViewer) + '. Please provide a correct software name: ')
        
        if showTips:
            print(dimMessage('Tips: for next uses, you can change the myPDFViewer global variable in the Global Variable section of galfit.py and set it to your favourite pdf viewer.'))
        return
    
    if showRecapFiles:
        global myPDFViewer
        
        print('Trying to open recap files located in ' + brightMessage(pathRecap) + ' with ' +  brightMessage(myPDFViewer))
        recap     = opath.join(pathRecap, 'recap')
        showTips  = True
        
        for i in range(1, splt+1):
            recapName        = '%s%d.pdf' %(recap, i)
            print('Showing recap file ' + brightMessage(recapName))
            
            try:
                
                code         = sub.run(myPDFViewer + ' ' + recapName, shell=True).returncode
                
                if code != 0:
                    try:
                        askAgain(recapName, showTips=showTips)
                        code = sub.run(myPDFViewer + ' ' + recapName, shell=True).returncode
                        
                        if code != 0:
                            raise OSError
                    except:
                        raise OSError
            except:
                raise OSError(errorMessage('Could not open pdf recap file %s with pdf viewer %s' %(recapName, myPDFViewer)))
    
    # Disabling future colored text
    col.deinit()
    return True
     

def writeConfigs(header, listProfiles, inputNames, outputNames=[], feedmeNames=[], constraintNames=[], constraints=None,
                 pathFeedme="./feedme/", pathIn="./inputs/", pathOut="./outputs/", pathConstraints="./constraints/"):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Make galfit FEEDME and CONSTRAINT files using the same profiles.
    
    :param dict header: dictionnary with key names corresponding to input parameter names in genHeader function. **See :py:func:run_galfit for more information**.
    :param list[str] inputNames: list of galaxies input .fits files
    :param list[dict] listProfiles: list of profiles (models). **See :py:func:run_galfit for more information**.
    
    :param list[str] constraintNames: (**Optional**) list of CONSTRAINT files. If not provided, the constraint files will have the same name as the input ones but with a .constraints extension at the end.
    :param list[str] feedmeNames: (**Optional**) list of FEEDME files. If not provided, the feedme files will have the same name as the input ones but with .feedme extensions at the end.
    :param list[str] outputNames: (**Optional**) list of galaxies output .fits files. If not provided, the output files will have the same name as the input ones with _out appended before the .fits extension.
    :param str pathFeedme: (**Optional**) location of the feedme files relative to the current folder or as an absolute path
    :param str pathIn: (**Optional**) location of the input files relative to the current folder or as an absolute path
    :param str pathOut: (**Optional**) location of the output files relative to the current folder or as an absolute path
    
    :param dict constraints: (**Optional**) list of dictionaries used to generate the constraints. If None, no constraints are applied. **See :py:func:run_galfit for more information**.
    
    :raises ValueError: if inputNames, outputName, feedmeNames and constraintNames are not iterables of the same length
    :raises OSError: if pathIn, pathOut, pathFeedme or pathConstraints are not correct path
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
  
    if pathIn is None:
        pathIn = ""
    else:
        if not opath.isdir(pathIn):
            raise OSError("Given path %s does not exist or is not a directory. Please provide an existing directory before making the .feedme files. Cheers !" %pathIn)
            
    if not opath.isdir(pathOut):
        raise OSError("Given path %s does not exist or is not a directory. Please provide an existing directory before making the .feedme files. Cheers !" %pathOut)
        
    if not opath.isdir(pathFeedme):
        raise OSError("Given path %s does not exist or is not a directory. Please provide an existing directory before making the .feedme files. Cheers !" %pathFeedme)
        
    if constraints is None:
        print('No constraints provided. Parameters are free to vary as much they want.')
    else:
        if not opath.isdir(pathConstraints):
            raise OSError("Given path %s does not exist or is not a directory. Please provide an existing directory before making the .feedme files. Cheers !" %pathConstraints)

    #################################################################
    #                Writing feedme and constraint files            #
    #################################################################
    
    for inp, out, fee, con in zip(inputNames, outputNames, feedmeNames, constraintNames):

        # output .constraint file
        if constraints is not None:
            fname = opath.join(pathConstraints, con)
            with open(fname, "w") as file:
                # Get formatted text
                tmp = genConstraint(constraints)
                file.write(tmp)
                
                # set header key to add the file name into the .feedme file
                header['couplingFile'] = fname
        
        # output .feedme file
        with open(opath.join(pathFeedme, fee), "w") as file:
            
            # Set input and ouput file names
            header['inputImage']  = opath.join(pathIn,  inp)
            header['outputImage'] = opath.join(pathOut, out)
            
            # Get formatted text, check in genFeedme that dict keys are okay and write into file if so
            tmp = genFeedme(header, listProfiles)
            file.write(tmp)


#####################################################################
#              General functions for galfit config files            #
#####################################################################


def genConstraint(dicts):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Make a galfit CONSTRAINT file.
    
    :param list[dict] constraintNames: list of constraint dictionaries. **See** :py:func:`run_galfit` **for more information**.
    
    :returns: CONSTRAINT file as formatted text
    :rtype: str
    
    :raises TypeError: 
        
        * if **dicts** is not a list
        * if 'type' key in constraint dict is not an acceptable value
        * if 'components' key in one of the dict is not a list
        * if 'value' key in one of the dict is not a list
        
    :raises KeyError: if neither 'parameter', nor 'constraints', nor 'components' keys are found in the dictionaries
    :raises ValueError: 
        
        * if 'parameter' key in one of the dict is not an acceptable value
        * if 'components' key in one of the dict does not have the correct length
        * if 'type' key in one of the dict does not have the correct length
    """
    
    def mapParams(par):
        '''Map the given parameter to its correct name (shall be updated as new segmentation faults are found)'''
        
        if   par.lower() in ['fwhm', 'diskscalelength', 'diskscaleheight', 'rs', 'rt', 'rc']:
            par = 're'
        elif par.lower() == 'bovera':
            par = 'q'
        elif par.lower() == 'pa':
            par = 'pa'
        elif par.lower() in ['alphaferrer', 'powerlaw']:
            par = 'alpha'
        elif par.lower() == 'betaferrer':
            par = 'beta'
        
        return par
    

    if type(dicts) is not list:
        raise TypeError("Given parameter 'dicts' is not a list. Please provide a list of dictionaries. Cheers !")
    
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

        try:
            if constraint['type'] not in everyConstraint:
                raise ValueError('Given constraint type %s is not correct. Possible values are %s. Please provide one of these values if you want to put a constraint on parameters. Cheers !' %(constraint['type'], everyConstraint))
        except TypeError:
            raise TypeError("'constraint' key value in dictionary should be a dictionary with keys 'type' and 'value'. Please provide a dictionary with these keys for the constraint. Cheers !")
            
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
        elif constraint['type'] == 'absoluteRange':
            constraintList.append('%.1f to %.1f' %tuple(constraint['value']))
        else:
            constraintList.append('%.1f %.1f' %tuple(constraint['value']))
            
        # Some parameters must change name between the feedme file and the constraint file (typically FWHM -> re) so we map them
        param = mapParams(param)
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


def genFeedme(header, listProfiles):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Make a galfit FEEDME configuration file using the given profiles.
    
    :param dict header: (**Optional**) dictionnary with key names corresponding to input parameter names in :py:func:genHeader function
    :param list[dict] listProfiles: (**Optional**) list of profiles (models). **See** :py:func:`run_galfit` **for more information**.
                
    :returns: FEEDME configuration file with header and body as formatted text
    :rtype: str
    
    :raises ValueError: if 'name' key in one of the profiles is not an acceptable value
    :raises KeyError: if 'name' key is missing in one of the profiles
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


def genHeader(inputImage="none", outputImage='output.fits', sigmaImage="none", psfImage="none", maskImage="none", couplingFile="none", 
              xmin=0, xmax=100, ymin=0, ymax=100, sizeConvX=None, sizeConvY=None,
              psfSamplingFactor=1, zeroPointMag=30.0, arcsecPerPixel=[0.03, 0.03],
              displayType="regular", option=0):
    r"""
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Constructs a galfit FEEDME header as general as possible.

    :param str inputImage: name of the input .fits file. 
    
        .. note::
            
            If "none" is given, when running the feedme file, the fitting will be skipped and a model will be generated using the provided parameters.
        
    :param str outputImage: name of the file within which the image will be stored. 
    
        .. note::
            
            Technically, the output "image" is not an image but a (1+3)D fits file, where:
            
                * layer 0 is a blank image whose header contains the fits keys
                * layer 1 is the original image within the fitting region
                * layer 2 is the model image
                * layer 3 is the residual between the model and the original image. 
                
            The residual is computed by subtracting layer 2 from layer 1. 
            
            Hence, if any galaxy component is missing in the model image (even if it was optimised), the residual will not reflect the "fitness" of fit.
    
    :param int xmax: maximum x coordinate of the fitting region box (in pixel)
    :param int xmin: minimum x coordinate of the fitting region box (in pixel)
    :param int ymax: maximum y coordinate of the fitting region box (in pixel)
    :param int ymin: minimum y coordinate of the fitting region box (in pixel)
    
    :param [float, float] arcsecPerPixel: (**Optional**) angular resolution (in arcsec) of image pixels. First element is the angular resolution in the x direction, second element in the y direction.
    :param displayType: (**Optional**) galfit display mode
    
        * "Regular" mode does not allow any interaction
        * "Both" and "curses" modes allow you to interact with galfit during the fitting routine (both will display a xterm terminal).
    :type displayType: "regular", "both" or "curses"
    :param str couplingFile: name of the CONSTRAINT file used to add constraints on parameters
    :param str maskImage: name of the image with bad pixels masked. Possible options are:
        
        * a .fits file (with the same dimensions as the input image) with a value of 0 for good pixels and >0 for bad pixels
        * an ASCII file with two columns separated by a blank space (first column x coordinate, second y coordinate) listing all the bad pixels locations
        
    :param int option: option
        
        * if 0, galfit run normally as explained above
        * if 1, the model image only is made using the given parameters
        * if 2, an image block (data, model and residual) is made using the given parameters
        * if 3, an image block with the first slice beeing the data, and the followings ones beeing one (best-fit) component per slice
   
    :param str psfImage: name of the psf image (fits file)
    :param int psfSamplingFactor: multiplicative factor used to scale the image pixel angular scale to the psf pixel angular scale if the psf is oversampled. Technically it is the ratio between the psf platescale (in arcsec/px) and the data platescale assuming the same seeing.
    :param str sigmaImage: name of the so called variance map (technically standard deviation map) where the value of the standard deviation of the underlying distribution of a pixel is given in place of the pixel value in the image (fits file only). If "none", galfit will compute one.
    :param int sizeConvX: x size of the convolution box (in px)
    :param int sizeConvY: y size of the convolution box (in px)
    :param float zeroPointMag: zero point magnitude used in the definition 
    
        .. math::
            
            m = -2.5 \log_{10} ({\rm{ADU}}/t_{\rm{exp}}) + {\rm{zeropoint}},
            
        where :math:`t_{\rm{exp}}` if the exposition time (generally found in the input .fits file header).
    
    :returns: header as a formatted string
    :rtype: str
    """
    
    psfSamplingFactor, zeroPointMag, option = toStr([psfSamplingFactor, zeroPointMag, option])
    
    # Define convolution box to full image and convolution center to image center if not given
    xmaxmin         = xmax-xmin
    ymaxmin         = ymax-ymin
    if sizeConvX is None:
        sizeConvX = xmaxmin
    if sizeConvY is None:
        sizeConvY = ymaxmin
    
    # Image region to fit
    imageRegion     = "%d %d %d %d" %(xmin, xmax, ymin, ymax)
    
    # Convolution box X and Y size
    convBox         = "%d %d" %(sizeConvX, sizeConvY)
    
    # Angular size per pixel in X and Y directions
    plateScale      = "%.3f %.3f" %tuple(arcsecPerPixel)
        
    # Compute max text length to align columns
    textLenMax      = maxStringsLen([outputImage, inputImage, sigmaImage, psfImage, psfSamplingFactor, maskImage, couplingFile, imageRegion, convBox, zeroPointMag, plateScale, displayType, option])
    formatText      = "%-" + "%d"%textLenMax + "s"
    
    # Generate items text 
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