#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

With the help of Lina Issa - IRAP

Functions meant to automatise as much as possible plotting of data of any kind.
"""

import numpy                              as     np
import matplotlib.pyplot                  as     plt
import matplotlib.gridspec                as     gridspec
import astropy.io.fits                    as     fits
from   matplotlib.colors                  import Normalize, LogNorm, SymLogNorm, PowerNorm, TwoSlopeNorm, BoundaryNorm, CenteredNorm
from   astropy.modeling.functional_models import Gaussian2D
from   matplotlib.markers                 import MarkerStyle
from   copy                               import copy
from   os.path                            import isfile


################################################################################################
#                                   Galfit models plots                                        #
################################################################################################


def display_hst_models(file1, fileout='test.pdf', title=None, cmap='spectral', log=False, show=False):
    '''
    .. codeauthor:: Epinat Benoit - LAM <benoit.epinat@lam.fr> & Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    This function enables to display a GALFIT output file containing the galaxy image, the associated model and the residuals.
    
    :param str file1: name of GALFIT output file
        
    :param str cmap: (**Optional**) name of the colormap. Must be understood by matplotlib.
    :param str fileout: (**Optional**) name of the output file
    :param bool log: (**Optional**) whether to have a log scale or not
    :param bool show: (**Optional**) whether to show the image or not
    :param str title: (**Optional**) title of the output image
    
    :raises TypeError:
        
        * if **title** is neither None, nor of type str
        * if **fileout** is not of type str
        * if **cmap** is not of type str
        * if **log** is not of type bool
        * if **show** is not of type bool
    '''
    
    # Perform checks
    if title is not None and not isinstance(title, str):
        raise TypeError('title is of type %s but it must be a string.' %type(title))
        
    if not isinstance(fileout, str):
        raise TypeError('fileout is of type %s but it must be a string.' %type(fileout))
        
    if not isinstance(cmap, str):
        raise TypeError('cmap is of type %s but it must be a string.' %type(cmap))
        
    if not isinstance(log, bool):
        raise TypeError('log is of type %s but it must be a bool.' %type(log))
        
    if not isinstance(show, bool):
        raise TypeError('show is of type %s but it must be a bool.' %type(show))

    # Load data    
    hdul     = fits.open(file1)
    data     = hdul[1].data
    model    = hdul[2].data
    res      = hdul[3].data
    
    # Generate figure
    fig      = plt.figure(figsize=(12, 3))
    
    if title is not None:
        fig.suptitle(title)
    
    maxi     = np.max([np.max([data, model]), np.abs(np.min([data, model]))])
    mini     = -maxi
    
    norm     = None
    if log:
        norm = SymLogNorm(linthresh=1, vmin=mini, vmax=maxi)
        
    fig.add_subplot(131)
    plt.imshow(data, origin='lower', cmap=cmap, interpolation='nearest', 
                         vmin=mini, vmax=maxi, norm=norm)
    plt.colorbar(fraction=0.05, shrink=1.)
    
    fig.add_subplot(132)
    plt.imshow(model, origin='lower', cmap=cmap, interpolation='nearest', 
                         vmin=mini, vmax=maxi, norm=norm)
    plt.colorbar(fraction=0.05, shrink=1.)
    
    fig.add_subplot(133)
    maxi     = np.max([np.max(res), np.abs(np.max(res))])
    mini     = -maxi
    plt.imshow(res, origin='lower', cmap=cmap, interpolation='nearest',
                         vmin=mini, vmax=maxi)
    plt.colorbar(fraction=0.05, shrink=1.)
    
    plt.savefig(fileout)
    
    if show:
        plt.show()
    plt.close()


def genMeThatPDF(fnamesList, pdfOut, readFromFile=False, groupNumbers=None, log=True, diverging=False, zeroPoint=0.0, cmap='bwr'):
    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Generate a pdf file with all the GALFIT images (data, model and residual) side by side.
    
    .. note::
        
        When certain problems arise, a default blank plot is drawn instead. Below is a list of issues where a blank plot may be generated
            
            * if a file is not found
            * if the min and max of an image are equal

    :param list[str] fnamesList: all the file names (paths included) with the GALFIT images. If **readFromFile** is True, provide the name of a file containing all the different names (one per line).
    :param str pdfOut: name of the output pdf file
        
    :param bool diverging: (**Optional**) whether to use a diverging norm or not. If False, a linear norm will be used (overriding any log norm).
    :param str cmap: (**Optional**) colormap to use for the plot
    :param list[str] groupNumbers: (**Optional**) list of groups the galaxies belong to
    :param bool log: (**Optional**) whether to show images as log or not
    :param bool readFromFile: (**Optional**) whether to read the file names from a file or not. If True, the names must be listed one per line.
    :param float zeroPoint: (**Optional**) value at which the diverging norm will split in two
    """
    
    def noModelAvailable():
        '''Generate default values when no model is available for the plot'''
        
        sz        = 60
        data      = np.array([[0]*sz]*sz)
        model     = data
        res       = data
        maxi      = 1
        mini      = 0
        log       = False
        diverging = False
        cmap      = 'Greys_r'
        norm      = BoundaryNorm([0, 0.5, 1], 2)
        noFile    = True
        
        return sz, data, model, res, maxi, mini, log, diverging, cmap, norm, noFile
    
    # Get file names from a file if necessary
    if readFromFile:
        fnamesList, groupNumbers = np.genfromtxt(fnamesList, dtype=str, unpack=True)
    
    # Computing the number of necessary subfigures
    nbfig    = len(fnamesList)
    
    # Stripping path from file names
    names = np.copy(fnamesList)
    for num, name in enumerate(fnamesList):
        names[num] = name.split('/')[-1]
    
    # Creating default values if group numbers are not given
    if (groupNumbers is None) or (len(groupNumbers) != len(fnamesList)):
        groupNumbers = [None]*(nbfig)
    
    fig = plt.figure(figsize=(17, 6*nbfig))
    gs  = gridspec.GridSpec(nbfig, 3, figure=fig)
    
    cmapSave       = cmap
    previousNumber = None
    for num, file, name, gr in zip(range(0, 3*nbfig, 3), fnamesList, names, groupNumbers):
        
        # Re-initialise the color map if it was changed
        cmap           = cmapSave
        
        # Fetching fits file extensions with data, model and residual maps
        try:
            hdul       = fits.open(file)
            data       = hdul[1].data
            sz         = int(hdul[1].header['NAXIS2'])
            model      = hdul[2].data
            res        = hdul[3].data
            
            # Defining a maximum and minimum which are symetrical
            maxi       = np.nanmax([np.nanmax([data, model]), np.abs(np.nanmin([data, model]))])
            mini       = -maxi
            norm       = None
            noFile     = False
            
        except FileNotFoundError:
            # If no file can be found (Galfit failed to model), we generate a default image to place on the plots
            sz, data, model, res, maxi, mini, log, diverging, cmap, norm, noFile = noModelAvailable()
        
        # Defining norm based on input parameter value
        if log:
            norm   = SymLogNorm(linthresh=maxi/1e3, vmin=mini, vmax=maxi)
        if diverging:
            mini   = np.nanmin([data, model])
            maxi   = np.nanmax([data, model])
            
            if mini == maxi:
                sz, data, model, res, maxi, mini, log, diverging, cmap, norm, noFile = noModelAvailable() 
            elif mini == zeroPoint:
                mini = - maxi
                norm = TwoSlopeNorm(vmin=mini, vcenter=zeroPoint, vmax=maxi)
            elif maxi == zeroPoint:
                maxi = -mini
                norm = TwoSlopeNorm(vmin=mini, vcenter=zeroPoint, vmax=maxi)
        
        # Plotting the three plots side by side
        ax1        = plt.subplot(gs[num])
        ax1.title.set_text(name)
        plt.imshow(data, origin='lower', cmap=cmap, interpolation='nearest', vmin=mini, vmax=maxi, norm=norm)
        
        if not noFile:
            plt.grid()
            plt.colorbar(fraction=0.05, shrink=1.)
        
        # Adding group info on the plots if a new group is encountered
        if previousNumber != gr:
            print("group", gr)
            previousNumber = gr
            plt.text(0, sz+30, "Group: %s" %(str(gr)), fontsize=20, fontweight='bold')
        
        ax2 = plt.subplot(gs[num+1])
        ax2.title.set_text('model')
        plt.imshow(model, origin='lower', cmap=cmap, interpolation='nearest', vmin=mini, vmax=maxi, norm=norm)
        
        if not noFile:
            plt.grid()
            plt.colorbar(fraction=0.05, shrink=1.)
        
        ax3 = plt.subplot(gs[num+2])
        ax3.title.set_text('residual')
        
        if not noFile:
            maxi = np.nanmax(res)
            mini = np.nanmin(res)
            print(maxi, mini, zeroPoint)
            
            if mini == maxi:
                sz, data, model, res, maxi, mini, log, diverging, cmap, norm, noFile = noModelAvailable() 
            elif mini == zeroPoint:
                mini = - maxi
                norm = TwoSlopeNorm(vmin=mini, vcenter=zeroPoint, vmax=maxi)
            elif maxi == zeroPoint:
                maxi = -mini
                norm = TwoSlopeNorm(vmin=mini, vcenter=zeroPoint, vmax=maxi)
            
        plt.imshow(res, origin='lower', cmap=cmap, interpolation='nearest', norm=CenteredNorm(vcenter=0))
        
        if not noFile:
            plt.colorbar(fraction=0.05, shrink=1.)
    
    plt.savefig(pdfOut, bbox_inches='tight')
    plt.close()
    

#########################################################################
#                    Automated plotting utilities                       #
#########################################################################

'''
'''
    
def effective_local_density(numPlot, X, Y, xerr, yerr, xmin=None, xmax=None, ymin=None, ymax=None, dx=1, dy=1, nx=None, ny=None):
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Generate an effective local density plot. 
    
    The idea is that, given a set of measurements for which we have values of the data points scatter in X and Y, rather than just plotting those values with error bars, we can instead try to compute the "likelihood" of having data points in a particular location.
    
    
    .. rubric:: **Information**
    
    The "likelihood" is represented as an effective 2D density:
    
        * If we are located infinitely far away from the bulk of the points, then we expect the density to drop to 0.
        * On the other hand, if all the measurements are located at the same location with 0 error assigned to them, then we expect to have a density maximised as N/box size of a point.
    
    This gives an idea of how clustered along a certain relation the points could be given their error bars.
    
    But, error bars are only indicators of how "wrong" the measurement could be. They do not strictly show the lower and upper bounds of the measurement. 
    
    Error bars are sometimes represented as a 1 sigma width of a Gaussian distribution centered at that measurement location. Thus, we are assuming that this is the case here.


    .. warning::
        
        This is purely experimental.

    :param numPlot: plot identifier. It can be 
    
        * an int (format is XYZ with X the number of rows, Y the number of columns and Z the plot position)
        * a list with 3 similar numbers [X, Y, Z]
        * a matplotlib GridSpec instance
        
    :type numPlot: list[3 int] or ndarray[3 int] or int or matplotlib GridSpec instance
    :param X: X position of the data points
    :type X: ndarray[int] or ndarray[float]
    :param xerr: X axis error on the data points
    :type xerr: ndarray[int] or ndarray[float]
    :param Y: y position of the data  points
    :type Y: ndarray[int] or ndarray[float]
    :param yerr: y axis error on the data points
    :type yerr: ndarray[int] or ndarray[float]

    :param dx: (**Optional**) x axis step used to draw the grid. If **nx** is provided, this parameter is overriden.
    :type dx: int or float
    :param dy: (**Optional**) y axis step used to draw the grid. If **ny** is provided, this parameters is overriden
    :type dy: int or floats
    :param int nx: (**Optional**) number of cells along the x axis. If None, **dx** is used instead to compute this value.
    :param int ny: (**Optional**) number of cells along the y axis. If None, **dy** is used instead to compute this value.
    :param xmin: (**Optional**) minimum x axis value for the x axis of the plot. If None, data points values will be used as bound.
    :type xmin: int or float
    :param xmax: (**Optional**) maximum x axis value for the x axis of the plot. If None, data points values will be used as bound.
    :type xmax: int or float
    :param ymin: (**Optional**) minimum y axis value for the y axis of the plot. If None, data points values will be used as bound.
    :type ymin: int or float
    :param ymax: (**Optional**) maximum y axis value for the y axis of the plot. If None, data points values will be used as bound.
    :type ymax: int or float
        
    :returns: matplotlib main axis
    :raises TypeError: if neither **X**, **Y**, **xerr** nor **yerr** is of type ndarray
    :raises ValueError: if either **xmin**, **xmax**, **ymin** or **ymax** is nan
    '''
    
    # Generate subplot
    typ = type(numPlot)
    if isinstance(typ, (list, np.ndarray)) and len(numPlot) == 3:
        ax1 = plt.subplot(numPlot[0], numPlot[1], numPlot[2])
    else:
        ax1 = plt.subplot(numPlot)
    
    for i in [X, Y, xerr, yerr]:
        if not isinstance(i, np.ndarray):
            raise TypeError('One of the mandatory parameters is type %s but mandatory parameters must be numpy arrays only. Cheers !' %type(i))
        
    if xmin is None:
        xmin  = np.nanmin(X)
    if xmax is None:
        xmax  = np.nanmax(X)
    if ymin is None:
        ymin  = np.nanmin(Y)
    if ymax is None:
        ymax  = np.nanmax(Y)
    
    if np.nan in [xmin, xmax, ymin, ymax]:
        raise ValueError('One in xmin, xmax, ymin, ymax is nan, which may be due to having an array full of np.nan. Please check your arrays. Cheers !')
        
    # Add a small error on data if error is 0 or np.nan, otherwise the Gaussian becomes a Dirac
    xerr0   = xerr==0
    yerr0   = yerr==0
    xerrnan = np.isnan(xerr)
    yerrnan = np.isnan(yerr)
    
    if np.any(xerr0) or np.any(xerrnan):
        xerr[np.logical_or(xerr0, xerrnan)] = np.nanmin(xerr[np.logical_and(~xerrnan, ~xerr0)])

    
    if np.any(yerr0) or np.any(yerrnan):
        yerr[np.logical_or(yerr0, yerrnan)] = np.nanmin(yerr[np.logical_and(~yerrnan, ~yerr0)])
        
    print('Minimum errors are %s and %s' %(np.nanmin(xerr), np.nanmin(yerr)))
    
    # Generate the Gaussians
    gaussians = [Gaussian2D(amplitude=1.0, x_mean=x, y_mean=y, x_stddev=xe, y_stddev=ye) for x, y, xe, ye in zip(X, Y, xerr, yerr)]
        
    # Generate the grid
    if nx is None:
        nx    = (xmax - xmin)//dx + 1
    if ny is None:
        ny    = (ymax - ymin)//dy + 1
    
    xarr      = np.linspace(xmin, xmax, nx)
    yarr      = np.linspace(ymin, ymax, ny)
    
    XX, YY    = np.meshgrid(xarr, yarr)
    
    # Compute the product of Gaussians at each grid point
    ZZ        = np.sum([i(XX, YY) for i in gaussians], axis=0)/len(gaussians)
    
    # Compute the most likely path of Y given X
    YgX_x, YgX_y = np.asarray([[x[0], y[np.argmax(z)]] for x, y, z in zip(XX.T, YY.T, ZZ.T)]).T
    
    # Compute the most likely path of X given Y
    XgY_x, XgY_y = np.asarray([[x[np.argmax(z)], y[0]] for x, y, z in zip(XX, YY, ZZ)]).T
    
    # Plotting
    ret       = plt.contourf(XX, YY, ZZ, levels=nx//2, cmap='hot', vmin=0)
    col       = plt.colorbar(ret, label='Probability')
    
    plt.plot(YgX_x, YgX_y, linestyle='--', color='blue', linewidth=2, label=r'$\lbrace (x,y) | y(x) = \max_Y ( Z(X=x, Y) ) \rbrace$')
    plt.plot(XgY_x, XgY_y, linestyle='-', color='magenta', linewidth=2.5, label=r'$\lbrace (x,y) | x(y) = \max_X ( Z(X, Y=y) ) \rbrace$')
    
    return ax1, col, ret
    

def singleContour(X, Y, Z, contours=None, sizeFig=(12, 12), aspect='equal', hideAllTicks=False, cmap='plasma', colorbar=True,
                  norm='log', cut=None, xlim=None, ylim=None, title=None, filled='both'):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Draw a (filled) contour plot.

    :param X: grid containing the x-axis values for each pixel
    :type X: meshgrid ndarray
    :param Y: grid containing the y-axis values for each pixel
    :type Y: meshgrid ndarray
    :param Z: grid containing the z-axis values for each pixel. This will correspond to the contour values.
    :type Z: meshgrid ndarray
    
    :param str aspect: (**Optional**) aspect of the plot
    :param str cmap: (**Optional**) colormap to use
    :param bool colorbar: (**Optional**) whether to draw a colorbar or not
    :param contours: (**Optional**) contours to draw 
    
        * if an int, it must be the number of contours to draw
        * if a list, it must be the contour values 
        * if None, its value will be determined automatically (see matplotlib)
        
    :type contours: int or list[int] or list[float]
    :param float cut: (**Optional**) cut below which values in the Z data are put to np.nan. If None, no cut is applied.
    :param filled: (**Optional**) whether to draw a filled contour (if True) or just contours (if False), or both (if "both")
    :type filled: bool or str
    :param bool hideAllTicks: (**Optional**) whether to hide ticks or not
    :param str norm: (**Optional**) scale to use. Must either be 'log' or 'linear'.
    :param sizeFig: (**Optional**) width and height of the figure
    :type sizeFig: (int, int)
    :param (float, float) xlim: (**Optional**) x-axis bounds. If None, the min and max of X are used.
    :param (float, float) ylim: (**Optional**) y-axis bounds. If None, the min and max of Y are used.
    
    :returns: current axis and plot
    
    :raises ValueError: if **norm** is neither 'linear', nor 'log'
    '''
    
    plt.rcParams["figure.figsize"] = sizeFig
    f             = plt.figure()
    ax            = plt.subplot(111)
    ax.set_aspect(aspect)
    
    if title is not None:
        plt.title(title)
    
    Intensity     = Z.copy()
    if cut is not None:
        Intensity[Intensity<=cut] = np.nan
    
    
    if hideAllTicks:
        plt.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    
    if norm == 'log':
        theNorm   = LogNorm()
        
        if type(contours) is int:
            contours = np.logspace(np.log10(np.nanmin(Z)), np.log10(np.nanmax(Z)), num=contours)
            
        levels    = contours
    elif norm == 'linear':
        levels    = None
        theNorm   = Normalize()
    else:
        raise ValueError('Given norm is neither log nor linear. Please provide one of these options. Cheers !')

    if filled == 'both' or filled is True:
        ret       = plt.contourf(X, Y, Intensity, cmap=cmap, levels=contours, norm=theNorm)
    if filled == 'both' or filled is False:
        plt.contour(X, Y, Intensity, linestyles='dashed', levels=contours, norm=theNorm, cmap='gist_yarg') #colors='k'
    
    if colorbar:
        plt.colorbar(ret, ticks=levels)
        
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
        
    #plt.show()
    
    return ax, ret


def asManyHists(numPlot, data, bins=None, weights=None, hideXlabel=False, hideYlabel=False, hideYticks=False, hideXticks=False,
                placeYaxisOnRight=False, xlabel="", ylabel='', color='black',
                label='', zorder=0, textsize=24, showLegend=False, legendTextSize=24, shadow=True, fancybox=True, framealpha=1, legendEdgeColor=None, frameon=True,
                xlim=[None, None], locLegend='best', tickSize=24, title='', titlesize=24,
                outputName=None, overwrite=False, tightLayout=True, integralIsOne=None,
                align='mid', histtype='stepfilled', alpha=1.0, cumulative=False, legendNcols=1, hatch=None, orientation='vertical', log=False, stacked=False, grid=True):

    """
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Function which plots on a highly configurable subplot grid 1D histograms. A list of data can be given to have multiple histograms on the same subplot.

    :param ndarray data: data to plot as histogram
    :param numPlot: plot identifier. It can be 
    
        * an int (format is XYZ with X the number of rows, Y the number of columns and Z the plot position)
        * a list with 3 similar numbers [X, Y, Z]
        * a matplotlib GridSpec instance
        
    :type numPlot: list[3 int] or ndarray[3 int] or int or matplotlib GridSpec instance

    :param str align: (**Optional**) how to align bars with respect to their value. Must either be 'left', 'mid' or 'right'.
    :param float alpha: (**Optional**) transparency of the bars
    :param bins: (**Optional**) define the bins:
        
        * if an int, must be the number of bins
        * if a list, must be the bins edges
        
    :type bins: int or list[int]
    :param str color: (**Optional**) color for the data. It can either be a string or RGB values.
    :param bool cumulative: (**Optional**) whether to plot the cumulative distribution (where each bin equals the sum of the values in the previous bins up to this one)
    :param bool fancybox: (**Optional**) whether to draw a fancy legend or not
    :param float framealpha: (**Optional**) transparency the legend background
    :param bool frameon: (**Optional**) whether to draw the legend frame or not
    :param bool grid: (**Optional**) whether to show the grid or not
    :param str hatch: (**Optional**) hatching pattern
    :param bool hideXlabel: (**Optional**) whether to hide the x label or not
    :param bool hideXticks: (**Optional**) whether to hide the x ticks or not
    :param bool hideYlabel: (**Optional**) whether to hide the y label or not
    :param bool hideYticks: (**Optional**) whether to hide the y ticks or not
    :param str histtype: (**Optional**) type of histogram. Must either be 'bar' (histograms next to each other), 'barstacked' (stacked histograms), 'step' (unfilled histograms) or 'stepfilled' (filled histograms).
    :param bool integralIsOne: (**Optional**) whether to normalise the integral of the histogram
    :param str label: (**Optional**) legend label
    :param str legendEdgeColor: (**Optional**) color of the legend edges. If None, there is no edge.
    :param int legendNcols: (**Optional**) number of columns in the legend
    :param int legendTextSize: (**Optional**) size for the legend
    :param str locLegend: (**Optional**) position of the legend
    :param str orientation: (**Optional**) orientation of the bars
    :param str outputName: (**Optional**) file name where save the figure. If None, the plot is not saved into a file.
    :param bool overwrite: (**Optional**) whether to overwrite the ouput file or not
    :param bool placeYaxisOnRight: (**Optional**) whether to place the y axis of the plot on the right or not
    :param int textsize: (**Optional**) size for the labels
    :param bool shadow: (**Optional**) whether to draw a shadow around the legend or not
    :param bool showLegend: (**Optional**) whether to show the legend or not
    :param int tickSize: (**Optional**) size of the ticks on both axes
    :param bool tightLayout: (**Optional**) whether to use bbox_inches='tight' if **tightLayout** is True or bbox_inches=None otherwise
    :param ndarray[float] weights:  (**Optional**) weights to apply to each value in **data**
    :param str xlabel: (**Optional**) x axis label
    :param list[float] xlim: (**Optional**) x-axis limits to use. If None, data bounds are used.
    :param str ylabel: (**Optional**) y axis label
    :param list[floats] ylim:  (**Optional**) y-axis limits to use. If None, data bounds are used
    :param int zorder: (**Optional**) plot position. The lower the value, the earlier it will be plotted
        
    :returns: Current axis, hist values, bin values, patches elements and legend elements
    """
    
    ax1 = plt.subplot(numPlot)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.set_title(title, size=titlesize)
    ax1.tick_params(which='both', direction='in', labelsize=tickSize)
    
    if grid:
        plt.grid()
        
    #hiding labels if required
    if hideXlabel:
        ax1.axes.get_xaxis().set_ticklabels([])
    else:
        plt.xlabel(xlabel, size=textsize)    
    if hideXticks:
        ax1.axes.get_xaxis().set_ticklabels([])
    if hideYticks:
        ax1.axes.get_yaxis().set_ticklabels([])
    if not hideYlabel:    
        plt.ylabel(ylabel, size=textsize)
    
    #Place Y axis on the right if required
    if placeYaxisOnRight:
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        
    #Plotting
    #define X limits if required
    if (xlim[0] is None) and (xlim[1] is None):
        rang = None
    else:
        rang = (xlim[0], xlim[1])

#     print(data, bins, integralIsOne, weights, color, align)
        
    n, bns, ptchs = plt.hist(data, bins=bins, range=rang, density=integralIsOne, weights=weights, color=color,
                             align=align, histtype=histtype, label=label, zorder=zorder, alpha=alpha,
                             cumulative=cumulative, hatch=hatch, orientation=orientation, log=log, stacked=stacked)
    
    ax1.set_xlim(xlim)
    #set hatching pattern if there is one
    
    if showLegend:
        leg = plt.legend(loc=locLegend, prop={'size': legendTextSize}, shadow=shadow, fancybox=fancybox, edgecolor=legendEdgeColor, frameon=frameon,
                         framealpha=framealpha, ncol=legendNcols)
    else:
        leg = None
        
    if outputName is not None:
        #If we do not want to overwrite the file
        f = None
        if not overwrite:
            #Try to open it to check if it exists
            try:
                f = open(outputName, 'r')
            except:
                pass
            if f is not None:
                print('File %s already exists but overwritting was disabled. Thus exiting without writing.' %outputName)
                return ax1, n, bns
                
        f = open(outputName, 'w')
        
        bbox_inches = None
        if tightLayout:
            bbox_inches = 'tight'
            
        plt.savefig(outputName, bbox_inches=bbox_inches)
        
    return ax1, n, bns, ptchs, leg

                
def asManyPlots(*args, **kwargs):
    '''Alias for asManyPlots2. See asManyPlots2 docstring for more details.'''
    
    return asManyPlots2(*args, **kwargs)


def asManyPlots2(numPlot, datax, datay, 
                 dataProperties={}, generalProperties={}, axesProperties={}, titleProperties={}, 
                 colorbarProperties={}, legendProperties={}, outputProperties={}):
    
    """
    Function which plots on a highly configurable subplot grid either with pyplot.plot or pyplot.scatter. A list of X and Y arrays can be given to have multiple plots on the same subplot.
    This function has been developed to be used with numpy arrays or list of numpy arrays (structured or not). Working with astropy tables or any other kind of data structure might or might not work depending on its complexity and behaviour. 
    
    Mandatory inputs
    ----------------
        numPlot : int (3 digits), list of 3 int or GridSpec array element
            subplot within which the data will be plotted. There are three possible methods to define a subplot:
                - the easiest way is to provide a three digits number such as 111 (1 line, 1 column, first subplot) or 212 (2 lines, 1 column, second subplot from left to right and top to bottom). The main issue with using three digits numbers is that it is limited to 9 subplots maximum
                - to overcome this issue, one can provide a list of three values (up to 99) instead such as [1, 1, 1] (similar to 111) or [15, 24, 30]
                - finally, if one wants to generate a grid with subplots having different x and y sizes (say a main subplot plot and a smaller residual subplot appended to the bottom of the main one), one can generate a highly tunable grid using matplotlib.gridspec.gridspec and then provide the desired subplot with the correct array element.
                  For instance, if one writes grid = gridspec(2, 1, height_ratios=[1, 0.2]) this should generate a grid with the second subplot below the first one, with the same width and a height 1/5th of the figure. Then one can provide grid[0] (grid[1]) for numPlot if one wants to plot on the first (second) subplot.
            
        datax: list of numpy arrays/lists
            list of x-axis data which should be plotted. Each data belongs to a single array. A LIST MUST BE PROVIDED, so even if just one data is plotted one should write [yourDataArray] and not directly yourDataArray.
            
            One may want to plot their data differently on the same plot (for instance one usual line plot and one scatter plot). This can be set in dataProperties dictionary using the key 'type' (see below for more information).
            asManyPlots2 provide three kinds of plots:
                - usual plot (line or points) with a potential global color set in the dictionary dataProperties with the key 'color'
                - scatter plot (only points, no line), where the color (set with 'color' key as well) and size of points (set with 'size' key in dataProperties dict) can vary from point to point to show any variation with a third dimension (refered as the z component in the following)
                - 'mix' type of plot where lines only are plotted and colour coded according to a third value (also set with 'color' key in plotProperties dict). For this kind of plot, each array within the list will correspond to a single line.
            
        datay : list of numpy arrays/lists
            list of y-axis data which should be plotted. See datax description for more details.
            
    Optional inputs
    ---------------
    
    Most optional inputs have been gathered within dictionary structures. This was done in order to reduce the number of optional parameters in the function declaration. Unless provided, default values will be used.
    
    For those who are not familiar with dictionnaries, here we provide an explanation on how to use them for this purpose. Say, one wants to put labels on the x and y axes. This can be done by providing values to 'xlabel' and 'ylabel' keys of axesProperties dictionary.
    Thus, one would write 
    
        >>> mydict = {'xlabel':'this is my x label', 'ylabel':'this is another text for the y label'}
        >>> asManyPlots2(111, datax, datay, axesProperties=mydict)
        
    which will plot datay as a function of datax on a single subplot with x and y labels attached to the axes.
    
        dataProperties : dict
            dictionary gathering all tunable properties related to the data. See the list below for the complete list of dicionnary keys.
            
            Data points properties
            ----------------------
                'color' : list of str and/or arrays
                    list of colors for each plot. Each given color is mapped to the corresponding plot assuming 'color' and datax and datay are sorted in the same order. Default is 'black' for any kind of plot.
                        - for simple and 'mix' plots, color names must be provided
                        - for scatter plots, a list/array of values must be provided. This is because scatter plots actually map values to a color range and will apply a color to each point separately using this range. 
                
                'fillstyle' : list of str
                    marker fillstyle to use on the plot. Default is 'full' for 'plot' and 'scatter' plots and 'none' for 'mix' plots. Possible values are 'full', 'left', 'right', 'bottom', 'top', 'none'.
                
                'marker' : list of str/matplotlib markers
                    list of markers used for the data. Default is 'o'.
                        - if one does not want to have a marker (for instance for line plots), one can provide None for the given plot
                        - by default, markers will not appear on 'mix' plots
                        
                'markerEdgeColor' : list of str
                    color of the marker edges. Default is None for each plot so that marker edges have the same color as the data color. If the marker is unfilled, this value will be overidden and the marker will recover the data color instead.
                    
                'markerSize' : list of int
                    list of marker sizes for each plot
                        - for 'plot' plots a single value, a single value must be provided
                        - for 'scatter' plots, either a single value or a list/array of values (one per data point) can be provided (the latter will scale data points size accordingly to their value)
                        - for 'mix' plots, a single value can be provided if you want to have markers on your colored line plots
                    
                'unfillMarker' : list of bool
                    whether to unfill the markers or not. This keyword is different from 'fillstyle'. 'unfillMarker' will remove the facecolor from the markers, whereas 'fillstyle' will modify the shape of the markers (by removing the top part for instance). Default is False for every plot.
                    
                'transparency' : float, list of floats between 0 and 1
                    transparency of the data points (1 is plain, 0 is invisible). Default is 1 for any kind of plot.
                
            Line properties
            ---------------
                'linewidth' : list of int
                    list of line widths in the plots. For 'scatter' plots, any value can be given as there will be no line. Default is given by 'linewidth' key in generalProperties dict.
                    
                'linestyle' : list of str
                    list of line styles. Default for 'plot' and 'mix' types of plots is given by 'linestyle' key in generalProperties dict.
            
            Plot properties
            ---------------
                'type' : list of 'plot', 'scatter' and/or 'mix'
                    list of plot type for each array in datax and datay. Default is 'plot' for every data. One value per array must be provided. Three values are possible:
                        - 'plot' to plot a simple plot (line or points) with a single color applied to all the data points
                        - 'scatter' to plot a scatter plot (only points, no line) where the points are colour coded according to some array of the same dimension as datax and datay provided in the 'color' key
                        - 'mix' to plot a line plot only where each line will have a different colour
        
                'order' : list of int
                     order in which the data will be plotted. Data with the lowest order will be plotted first. Default is order of appearance in the datax and datay lists.
    
        generalProperties : dict
            dictionary gathering all tunable general properties. See the list below for the complete list of dictionary keys.
            
            Ticks properties
            ----------------
                'hideTicksLabels' : bool
                    whether to hide ticks labels. Default is False.
                'ticksDirection' : 'in', 'out' or 'inout'
                    direction of the ticks on the plot. Default is 'inout'.
                'ticksLabelsSize' : int
                    size of the ticks labels on the plot and the colorbar. Default is 2 points below 'textsize' key value.
                'ticksSize' : int
                    size of the ticks on the plot and on the colorbar if there is one. Default is 7.
                    
            Text properties
            ---------------
                'rcParams' : dict
                    advanced properties to pass to matplotlib rcParams
                'textsize' : int
                    overall text size in points. Default is 24.
                    
            Scale properties
            ----------------
                'scale' : str
                    scale of both axes
                    
            Plot properties
            --------------------
                'linewidth' : int
                    overall line width used in plots. For 'scatter' plots, any value can be given as there will be no line. Default is 2.    
                'linestyle' : str
                    overall line style used in plots. Default for 'plot' and 'mix' types of plots is plain style ('-').
                'markersize' : int
                    overall size of the markers for 'plot' and 'scatter' plots. Default is 16.
                    
            Miscellanous properties
            -----------------------
                'hideGrid' : bool
                    whether to hide the grid or not. Default is False.
                'gridZorder' : int
                    z order of the grid. Default is 1000 (generally enough to be plotted in last).
    
    
        axesProperties : dict
            dictionary gathering all tunable axes properties. See the list below for the complete list of dictionary keys.
            
            Ticks related keys
            ------------------
                'hideXticksLabels' : bool
                    whether to hide the x-axis ticks labels or not. Default is False.
                'hideYticksLabels' : bool
                    whether to hide the y-axis ticks labels or not. Default is False.
                'xTickDirection' : 'in', 'out' or 'inout'
                    where the x-axis ticks will appear. If 'in', they will be plotted within the subplots, if 'out', they will be plotted outside of the subplot and if 'inout', they will appear both within and outside. Default is 'tickDirection' value.
                'xTickSize' : int
                    size of the ticks on the x-axis. Default is 'tickSize' value. If a value is given, the 'tickSize' key value will be overriden. 
                'yTickDirection' : 'in', 'out' or 'inout'
                    where the y-axis ticks will appear. If 'in', they will be plotted within the subplots, if 'out', they will be plotted outside of the subplot and if 'inout', they will appear both within and outside. Default is 'tickDirection' value.
                'yTickSize' : int
                    size of the ticks on the y-axis. Default is 'tickSize' value. If a value is given, the 'tickSize' key value will be overriden. 
                
            Label related keys
            ------------------
                labelTextSize : int
                    size of the labels for every axis. Default is 24.
                'xlabel' :  str
                    label of the x-axis. Default is ''.
                xLabelTextSize : int
                    size of the x-axis label. Default is None. If a value is given, the 'labelTextSize' key value will be overriden. 
                'ylabel' : str
                    label of the y-axis. Default is ''.
                yLabelTextSize : int
                    size of the y-axis label. Default is None. If a value is given, the 'labelTextSize' key value will be overriden. 
                
            Scale and limits related keys
            -----------------------------
                'xscale' : str
                    scale for the y-axis (the most used are "linear", "log", "symlog"). Default is "linear".
                'yscale' : str
                    scale for the x-axis (the most used are "linear", "log", "symlog"). Default is "linear".
                    
                'xmax' : None, float or str
                    maximum value of the x-axis. Three different type of values can be given (default is None):
                        - None so that the maximum value of x data is used as upper limit
                        - float so that the given value is used as upper limit
                        - string with one of the following forms: 'num+offset' or '*+offset'
                    
                    in the latter case:
                        - num is an int corresponding to the position, in the x data list, of the data used to compute the minimum x value
                        - * corresponds to the full list of x data, so that the minimum value is that of the complete set of data
                        - offset is the offset applied to the x minimum value 
                    For e.g., '0+-1' will compute the minimum value in the Oth data element in the x data list (assume it is equal to -3). The x-axis minmum value will therefore be equal to this value minus 1 (-3-1=-4)
                    For e.g., '*+2' will compute the minimum value of all the x data in the x data list and set the x-axis minmum value to this value plus 2.
                    
                'xmin' : None, float or str
                    minimum value of the x-axis. Three different type of values can be given (default is None), see 'xmax' description for more information.
                'ymax' : None, float or str
                    maximum value of the y-axis. Three different type of values can be given (default is None), see 'xmax' description for more information.
                'ymin' : None, float or str
                    minimum value of the y-axis. Three different type of values can be given (default is None), see 'xmax' description for more information.
                
            Position related keys
            ---------------------
                'xAxisPos' : 'bottom' or 'top'
                    where to place the main x-axis. Default is 'bottom'.
                'yAxisPos' : 'left' or 'right'
                    where to place the main y-axis. Default is 'left'.   
                    
        titleProperties : dict
            dictionary gathering all tunable main title properties. See the list below for the complete list of dictionary keys.
            
            Text related keys
            -----------------
                'color' : str
                    color of the title. Default is 'black'.
                'font' : str
                    title font. Default is 'sans-serif'.
                'label' : str
                    title label. Default is ''.
                'size' : int
                    title size in points. Default is 26.
                'style' : 'normal', 'italic' or 'oblique'
                    title text style. Default is 'normal'.
                'weight' : int within range [0, 1000] or 'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'
                    'boldness' of the title font. Default is 'regular'.
                    
            Position related keys
            ---------------------
                'position' : 'center', 'left' or 'right'
                    where the title will be placed. Default is 'center'
                'verticalOffset' : float
                    title vertical offset from the regular position. Default is None, so that no offset is applied.
                    
        colorbarProperties : dict
            dictionary gathering all tunable colorbar properties. See the list below for the complete list of dictionary keys.
            A colorbar will only be plotted if at least one scatter plot is provided.
            
            General colobar properties
            --------------------------
                'hide' : bool
                    whether to hide the colorbar or not. Default is False.
                'orientation' : 'vertical' or 'horizontal'
                    orientation of the colorbar. Default is 'vertical'.
            
            Color map related keys
            ----------------------
                'cmap' : str or matplotlib colormap
                    colormap used by the scatter plot and in the colorbar. Default is 'Greys'.
                'min' : float
                    minmum value for the colormap. Default is None, so that the minimum value in the z component arrays of the scatter data will be used.
                'max' : float
                    maximum value for the colormap. Default is None, so that the maximum value in the z component arrays of the scatter data will be used.
                'offsetCenter' : float
                    colormap center of the dynamic color range. The given value will split the colormap mapping in two at the given position. This can be used to give more or less dynamic range for a certain range of values. Default is None, so that the regular center is used.
                'powerlaw' : float
                    powerlaw coefficient used in the 'powerlaw' scale. Default is 2.0.
                'scale' : matplotlib.colors.Normalize instance or str
                    norm of the colormap. Default is 'linear'. There are different ways one can change the scale of the colormap using an str argument
                        - for 'linear' and 'log' scales, just provide 'linear' or 'log' and the minimum and maximum values will be mapped from 'min' and 'max' keys
                        - for a 'symlog' scale, one can provide additional values using 'symLogLinThresh' and 'symLogLinScale' keys to change the range of the linear scale part.
                        - for a 'powerlaw' scale (z = y^n), one can provide an additional value using 'powerlaw' key to change the powerlaw coefficient.
                        
                    Note that it is also possible for a linear scale, to provide the 'offsetCenter' key in order to move the colormap center further up or down the colorbar values. This can be used to create two different mapping on either side of the center and modify the dynamic color range of each.
                    For instance, if one maps values between -10 and 10, but would like to have more details in the range [-5, 10], one can write colorbarProperties={'offsetCenter':-5} to split the colormap in two at position -5.
                    
                    It is also possible to declare a colormap normalize instance before calling matplotlib and directly give the colormap as a value for 'scale'.
                
                'symLogLinThresh' : float
                    linear threshold positive value. For a symetric log scale ('symlog'), this will correspond to the value such that the range [-linthresh, +linthresh] will be linearly maped around zero to avoid the colormap instance to diverge when reaching low values. Default is 0.1.
                'symLogLinScale' : float
                    linear scale factor in the colormap. This is used to strech or compress the range of the linear scale part in the colormap. Default is 1.0 and will correspond to 1 decade in the negative and in the positive part of the colormap. Default is 1.0.
                    
            Label related keys
            ------------------
                'label' : str
                    colorbar label placed next to it. Default is ''.
                'labelSize' : int
                    size in points of the colorbar label. Default is 24.
                
            Ticks related keys
            ------------------
                'ticks' : list
                    list of ticks plotted on the colorbar. If 'ticksLabels' key values are given, these values will be overriden and replaced by those given by 'ticksLabels'. Default if None, so that matplotlib automatically adjust the ticks values.
                'ticksColor' : str
                    color of the ticks on the colorbar. Default is 'black'.
                'ticksSize' : int
                    size of the ticks in points. Default is given by 'ticksSize' key in generalProperties dict.
                    
            Ticks labels properties
            -----------------------
                'ticksLabels' : list
                    list of ticks labels (str, int, float, etc.) for the colorbar. Default value is None, so that the ticks values are plotted on the colorbar.
                    This key can be used to change the colorbar ticks printed on the figure, without changing the scatter plot z values.
                    
                'ticksLabelsColor' : str
                    color of the ticks labels on the colorbar. Default is 'black'
                'ticksLabelsRotation' : float
                    counter-clockwise rotation of the ticks labels in degrees. Default is 0°.
                'ticksLabelsSize' : int
                    size of the labels associated to the ticks. Default is given by 'ticksLabelsSize' in generalProperties dict.
                    
        legendProperties : dict
            dictionary gathering all tunable legend properties. See the list below for the complete list of dictionary keys. By default, the legend is hidden.
            
            General legend properties keys
            ------------------------------
                'loc' : str/int
                    legend location on the plot. Default is 'best'.
                'ncols' : int
                    number of columns in the legend. Default is 1 so that every label will appear on a new line. 
                    For instance legendProperties={'numberColumns':3} will split labels into three columns.
                'order' : int
                    order of appearance of the legend (zorder). Default is 1000.
                    
            Text related keys
            -----------------
                'labels' : list of str
                    list of labels for the data. Labels must be given in the same order as the plots. Default is None so that no labels are shown for any kind of plot.
                'labelSize' : int
                    size of the text in the legend. Default is given by 'textsize' key in generalProperties dict.
            
            Lines and markers related keys
            ------------------------------
                'lineColor' : list of str
                    list of colors of the lines appearing in the legend before the text. Default is None so that the line color in the plot is used for plots and 'black' is used for scatter plots.
                    If provided, this list must contain as many colors as there are data plotted. For instance, if one plots a plot and a scatter plot, one may write legendProperties{'lineColor':['red', 'blue']} to draw the plot line in red and the scatter plot marker in blue within the legend.

                'markerEdgeColor' : list of str
                    list of marker edge colors shown in the legend. Default is None so that the line color in the plot is used for plots marker edges and black is used for scatter plots marker edges.
                    If provided, this list must contain as many colors as there are data plotted. For instance, if one plots a plot and a scatter plot, one may write legendProperties={'markerEdgeColor':['red', 'blue']} to draw the plot marker edge color in red and the scatter plot marker edge color in blue within the legend.
            
                'markerFaceColor' : list of str
                    list of marker face colors (main area) shown in the legend. Default is None so that the line color in the plot is used for plots markers and black is used for scatter plots markers.
                    If provided, this list must contain as many colors as there are data plotted. For instance, if one plots a plot and a scatter plot, one may write legendProperties={'markerFaceColor':['red', 'blue']} to draw the plot marker face color in red and the scatter plot marker face color in blue within the legend.
    
                'markerPosition' : 'left' or 'right'
                    whether to place the marker in the legend before the text ('left') or after it ('right'). Default is 'left'.
                    
                'markerScale' : int/float
                    scale factor to apply to the markers in the legend to increase or decrease their size relative to the plot (size_in_legend = markerScale*size_in_plot). Default is 1.0 so that they have the same size as in the plot.
    
            Style related keys
            ------------------
                'transparency' : float
                    transparency of the legend background. 1 is plain, 0 is fully transparent. Default is 1.
                'background' : str
                    color of the legend background. Default is None so that the default value in your rcParams file will be used (usually white).
                'fancy' : bool
                    whether to have a fancy legend box (round edges) or not. Default is True.
                'shadow' : bool
                    whether to draw a shadow around the legend or not. Default is True.
                'title' : str
                    legend title. Default is empty string.
                'titleSize' : float
                    legend title font size. Default is given by 'textsize' key in generalProperties dict.
    
        outputProperties : dict
            dictionary gathering all tunable ouput properties. See the list below for the comple list of dictionary keys.
            
            Properties
            ----------
                'outputName' : str
                    name of the file to save the graph into. If None, the plot will not be saved into a file. Default is None.
                'overwrite' : bool
                    whether to overwrite the ouput file or not. Default is False.
                'tightLayout' : bool
                    whether to set a tight_layout in the ouput image (no blank space on the sides) or not. Default is True.
                'transparent' : bool
                    whether to have a transparent background or not. Default is False.
        
    Return in this order: the current subplot, a list of all the different plots, the legend (None if no legend) and the colorbar (None if no colorbar).
    """
    
    ############################################################################################
    #                    Fonctions used to speed things up in the code                         #
    ############################################################################################
    
    def checkTypeAndChangeValueToList(data, typeToCheck, length=1):
        """
        Check data type and if not correct transform it into a list of a certain length
        
        Mandatory inputs
        ----------------
            data : any type
                data we want to check the type
            typeToCheck : any type
                type data should have
        
        Optional inputs
        ---------------
            length : int
                length of the output list
                
        Return given data if type is correct or a list of a given length with data value inside if not.
        """
        
        if not isinstance(data, typeToCheck):
            if isinstance(data, str):
                data = [data]*length
            else:
                try:
                    data = list(data)
                except TypeError:
                    data = [data]*length
        return data
            
    def completeList(data, length, default):
        """
        Complete a list with a default value if it is too short.
        
        Mandatory inputs
        ----------------
            data : list
                data to append default values to
            default : any type
                default value to append at the end of the data to complete the list
            length : int
                length the data list should have 
        
        Append default values to the list and return the new list.
        """
        
        ll = len(data)
        if ll != length:
            for num in range(length):
                if num >= ll:
                    data.append(default)

        return data
    
    def isType(data, typeToCheck, varName='NOT PROVIDED'):
        """
        Check data type.
        
        Mandatory inputs
        ----------------
            data : any type
                data we want to check the type
            typeToCheck : any type
                type data should have
                
        Optional inputs
        ---------------
            varName : str
                the name of the variable we want to test. It is used if an error is raised.
        
        Raise a TypeError if the data type is not correct or return True if it is.
        """
        
        typ = type(data)
        if typ is not typeToCheck:
            raise TypeError("TypeError: the given variable '%s' has type %s but the program requires it to have type %s. In order for the program to work, please provide the correct type. Cheers !" %(varName, typ, typeToCheck, ))
        return True
    
    def setDefault(data, default=None):
        """
        Set a default value in data list when given elements are None
        
        Mandatory inputs
        ----------------
            data : list
                data with any values whose None values will be replaced by a default value instead
                
        Optional inputs
        ---------------
            default : any type/list of any type
                default value which will replace None values
                
        Return a new list with None values replaced by a default one.
        """
        
        if type(default) is not list:
            default = [default]*len(data)
        
        for num, i  in enumerate(data):
            if i is None:
                data[num] = default[num]
        return data
    
    def setListFromDict(dictionary, keys=None, default=None):
        """
        Fill a list with values from a dictionary or from default ones if the key is not in the dictionary.
        
        Mandatory inputs
        ----------------
            dictionary : dict
                dictionary to get the keys values from
        
        Optional inputs
        ---------------
            default : list
                list of default values if given key is not in dictionary
            keys : list of str
                list of key names whose values will be appended into the list
                
        Return a list with values retrived from a dictionary keys or from a list of default values.
        """
        
        out = []
        for k, df in zip(keys, default):
            if k in dictionary:
                out.append(dictionary[k])
            else:
                out.append(df)
        return out
    
    def checkBoundStr(value, name):
        
        try:
            datanum, value = value.split('+')
            
            if datanum != '*':
                datanum    = int(datanum)
            value          = float(value)
        except:
            raise ValueError('Incorrect format (%s) for %s. If a string is provided it must have a format similar to 1+-2 or *+4.' %(value, name))
            
        if isinstance(datanum, str) and datanum != '*':
            raise ValueError('Mapping %s format into int and float failed. Check again that you provided the correct format. Cheers !' %name)
        if datanum != '*' and datanum < 0:
            raise ValueError('Given plot number for %s is negative. Only non-negative numbers are accepted.' %name)
        if datanum != '*' and datanum > data.nplots-1:
            raise ValueError('Given plot number (%d) for %s is too large. Expected a number below %d' %(datanum, name, data.plots-1))
            
        
        return datanum, value
        
    class gatherThingsUp:
        """
        An empty class used to store data within the same object.
        """
        info = "A simple container to gather things up."
    
    
    ##################################################
    #                  General layout                #
    ##################################################
    
    layout, layout.ticks, layout.line, layout.grid = gatherThingsUp(), gatherThingsUp(), gatherThingsUp(), gatherThingsUp()
    if isType(generalProperties, dict, 'generalProperties'):
        
        layout.textsize, layout.rcParams = setListFromDict(generalProperties, keys=['textsize', 'rcParams'], default=[24, None])
        
        # general ticks properties
        layout.ticks.labelSize, layout.ticks.size, layout.ticks.direction = setListFromDict(generalProperties, keys=['ticksLabelsSize', 'ticksSize', 'tickDirection'], default=[layout.textsize-2, 7, 'in'])
        
        # General properties
        layout.markersize, layout.ticks.hideLabels, layout.scale = setListFromDict(generalProperties, keys=['markersize', 'hideTicksLabels', 'scale'], default=[16, False, 'linear'])

        # General grid properties
        layout.grid.hide, layout.grid.zorder = setListFromDict(generalProperties, keys=['hideGrid', 'gridZorder'], default=[False, 1000])

        # General line properties
        layout.line.width, layout.line.style = setListFromDict(generalProperties, keys=['linewidth', 'linestyle'], default=[2, '-'])
    
    if layout.rcParams is not None:
        plt.rcParams.update(**layout.rcParams)
        
        
    ################################################################################################
    #                  Checking datax and datay are lists with a similar shape                     #
    ################################################################################################
    
    isType(datax, list, "datax")
    isType(datay, list, "datay")
    
    # Checking shape consistency between datax and datay
    shpX = np.shape(datax)
    shpY = np.shape(datay)
    if shpX != shpY:
        raise ValueError("Shape inconsistency: datax has shape %s but datay has shape %s. Please provide data with similar shape. Cheers !" %(str(shpX), str(shpY)))
    
    
    ################################################################################################
    #           Checking input type and setting values from optional dictionnaries                 #
    #              Gathering data properties into a single object for simplicity                   #
    ################################################################################################   
    
    # If everything went fine, we can define a few general properties useful later on
    data                   = gatherThingsUp()
    data.marker, data.line = gatherThingsUp(), gatherThingsUp()
    data.x, data.y         = gatherThingsUp(), gatherThingsUp()
    data.x.data            = datax
    data.y.data            = datay
    data.x.min             = np.min([np.min(i) for i in data.x.data])
    data.x.max             = np.max([np.max(i) for i in data.x.data])
    data.y.min             = np.min([np.min(i) for i in data.y.data])
    data.y.max             = np.max([np.max(i) for i in data.y.data])
    data.nplots            = len(data.x.data)

    if isType(dataProperties, dict, 'dataProperties'):
        
        # Set data properties for the plots
        data.zorder, data.color, data.type, data.transparency = setListFromDict(dataProperties, keys=['order', 'color', 'type', 'transparency'], default=[range(data.nplots), None, ['plot']*data.nplots, [1]*data.nplots])
        
        # Set properties of the markers on the plots
        data.marker.type, data.marker.size, data.marker.fillstyle, data.marker.unfill, data.marker.edgeColor = setListFromDict(dataProperties, keys=['marker', 'markerSize', 'markerFillstyle', 'unfillMarker', 'markerEdgeColor'], default=[['o']*data.nplots, None, None, [False]*data.nplots, [None]*data.nplots])
        
        # Set properties of the lines on the plots
        data.line.width, data.line.style = setListFromDict(dataProperties, keys=['linewidth', 'linestyle'], default=[None, None])
        
        # If the user provides a single value, we change it to a list
        data.type          = checkTypeAndChangeValueToList(data.type,          list, data.nplots)
        data.transparency  = checkTypeAndChangeValueToList(data.transparency,  list, data.nplots)
        data.marker.type   = checkTypeAndChangeValueToList(data.marker.type,   list, data.nplots)
        data.marker.unfill = checkTypeAndChangeValueToList(data.marker.unfill, list, data.nplots)
        data.line.style    = [str(i) for i in checkTypeAndChangeValueToList(data.line.style,    list, data.nplots)]
        
        # If the user provides an incomplete list, we complete it
        completeList(data.transparency, data.nplots, 1)
        completeList(data.marker.type, data.nplots, 'o')
        completeList(data.marker.unfill, data.nplots, False)
        
        if data.marker.size is None:      
            data.marker.size      = []
            
        if data.marker.fillstyle is None:
            data.marker.fillstyle = []
            
        if data.line.width is None:
            data.line.width       = []
            
        # If data.color is not provided, we set plot colors to 'black' and scatter plots points to the same value
        if data.color is None:
            data.color = ['black']*data.nplots
        else:
            setDefault(data.color, default='black')
        
        # Complete the size list with either 0 of the default value is the list is not complete
        lms = len(data.marker.size)  
        lmf = len(data.marker.fillstyle)
        llw = len(data.line.width)
        lls = len(data.line.style)
        lc  = len(data.color)
        for pos, typ in enumerate(data.type):
            if typ == 'mix':
                if pos >= lms:                
                    data.marker.size.append(0)
                if pos >= lmf:
                    data.marker.fillstyle.append('none')
                if pos >= lc:
                    data.color.append('black')
            else:
                if pos >= lms:
                    data.marker.size.append(layout.markersize)
                if pos >= lmf:
                    data.marker.fillstyle.append('full')
                if pos >= lc:
                    data.color.append('black')
            
            if typ == 'scatter':
                if pos >= llw:    
                    data.line.width.append(0)
                if pos >= lls:
                    data.line.style.append("None")
                if pos >= lc:
                    data.color.append('black')
            else:
                if pos >= llw:    
                    data.line.width.append(layout.line.width)
                if pos >= lls:
                    data.line.style.append(layout.line.style)
            
        # Set marker edge colors to face corlor (with 'face') if data points are supposed to be unfilled only for scatter plots
        for pos, nfllMrkr, typ in zip(range(data.nplots), data.marker.unfill, data.type):
            if typ == 'scatter' and (nfllMrkr or data.marker.edgeColor[pos] is None):
                data.marker.edgeColor[pos] = 'face'

    ########################################
    #            Axes properties           #
    ########################################
    
    xaxis, xaxis.label, xaxis.ticks = gatherThingsUp(), gatherThingsUp(), gatherThingsUp()
    yaxis, yaxis.label, yaxis.ticks = gatherThingsUp(), gatherThingsUp(), gatherThingsUp()
    if isType(axesProperties, dict, 'axesProperties'):
        
        # Label properties
        # X axis
        xaxis.label.text, xaxis.label.size, xaxis.scale = setListFromDict(axesProperties, keys=["xlabel", "xLabelTextSize", "xscale"], default=['', layout.ticks.labelSize, layout.scale])
        # y-axis
        yaxis.label.text, yaxis.label.size, yaxis.scale = setListFromDict(axesProperties, keys=["ylabel", "yLabelTextSize", "yscale"], default=['', layout.ticks.labelSize, layout.scale])

        # Ticks properties
        # x-axis
        xaxis.ticks.hideLabels, xaxis.ticks.size, xaxis.ticks.direction = setListFromDict(axesProperties, keys=["hideXticksLabels", "xTickSize", "xTickDirection"], default=[False, layout.ticks.size, layout.ticks.direction])

        # y-axis
        yaxis.ticks.hideLabels, yaxis.ticks.size, yaxis.ticks.direction = setListFromDict(axesProperties, keys=["hideYticksLabels", "yTickSize", "yTickDirection"], default=[False, layout.ticks.size, layout.ticks.direction])
        
        # Other properties
        xaxis.pos, yaxis.pos, xaxis.min, xaxis.max, yaxis.min, yaxis.max = setListFromDict(axesProperties, keys=["xAxisPos", "yAxisPos", "xmin", "xmax", "ymin", "ymax"], default=["bottom", "left", None, None, None, None])
      
    # Check minimum and maximum axes values first
    xminPlot                 = '*' 
    xminOffset               = 0
    if isinstance(xaxis.min, str):
        xminPlot, xminOffset = checkBoundStr(xaxis.min, 'xaxis.min')
        xaxis.min            = None
    
    xmaxPlot                 = '*' 
    xmaxOffset               = 0
    if isinstance(xaxis.max, str):
        xmaxPlot, xmaxOffset = checkBoundStr(xaxis.max, 'xaxis.max')
        xaxis.max            = None
        
    yminPlot                 = '*' 
    yminOffset               = 0
    if isinstance(yaxis.min, str):
        yminPlot, yminOffset = checkBoundStr(yaxis.min, 'yaxis.min')
        yaxis.min            = None
    
    ymaxPlot                 = '*' 
    ymaxOffset               = 0
    if isinstance(yaxis.max, str):
        ymaxPlot, ymaxOffset = checkBoundStr(yaxis.max, 'yaxis.max')
        yaxis.max            = None
        
        
    # Set x and y bounds if not provided
    if xaxis.min is None:
        if xminPlot == '*':
            xaxis.min = np.nanmin([np.nanmin(i) for i in data.x.data]) + xminOffset
        else:
            xaxis.min = np.nanmin(data.x.data[xminPlot]) + xminOffset
            
    if xaxis.max is None:
        if xmaxPlot == '*':
            xaxis.max = np.nanmax([np.nanmax(i) for i in data.x.data]) + xmaxOffset
        else:
            xaxis.max = np.nanmax(data.x.data[xmaxPlot]) + xmaxOffset
            
    if yaxis.min is None:
        if yminPlot == '*':
            yaxis.min = np.nanmin([np.nanmin(i) for i in data.y.data]) + yminOffset
        else:
            yaxis.min = np.nanmin(data.y.data[yminPlot]) + yminOffset
            
    if yaxis.max is None:
        if ymaxPlot == '*':
            yaxis.max = np.nanmax([np.nanmax(i) for i in data.y.data]) + ymaxOffset
        else:
            yaxis.max = np.nanmax(data.y.data[ymaxPlot]) + ymaxOffset
        
    #########################################
    #             Title properties          #
    #########################################
    
    title = gatherThingsUp()
    if isType(titleProperties, dict, 'titleProperties'):
        title.color, title.font, title.label, title.size, title.style, title.weight, title.position, title.verticalOffset = setListFromDict(titleProperties, keys=["color", "font", "label", "size", "style", "weight", "position", "verticalOffset"], default=['black', 'sans-serif', '', 26, 'normal', 'regular', 'center', None])
    
    
    ########################################
    #          Colormap properties         #
    ########################################
    
    colorbar, colorbar.ticks, colorbar.ticks.label, colorbar.label, colorbar.cmap = gatherThingsUp(), gatherThingsUp(), gatherThingsUp(), gatherThingsUp(), gatherThingsUp()
    if isType(colorbarProperties, dict, 'colorbarProperties'):
        
        # General properties
        colorbar.hide, colorbar.orientation = setListFromDict(colorbarProperties, keys=["hide", "orientation"], default=[False, 'vertical'])
        
        # Cmap properties
        colorbar.cmap.name, colorbar.cmap.min, colorbar.cmap.max = setListFromDict(colorbarProperties, keys=["cmap", "min", "max"], default=['Greys', None, None])
        
        # Ticks properties
        colorbar.ticks.values, colorbar.ticks.size, colorbar.ticks.color, colorbar.ticks.label.text, colorbar.ticks.label.size, colorbar.ticks.label.color, colorbar.ticks.label.rotation = setListFromDict(colorbarProperties, keys=["ticks", "ticksSize", "ticksColor", "ticksLabels", "ticksLabelsSize", "ticksLabelsColor", "ticksLabelsRotation"], default=[None,  layout.ticks.size, 'black', None, layout.ticks.labelSize, 'black', 0])
        
        # Scale properties
        colorbar.offsetCenter, colorbar.powerlaw, colorbar.scale, colorbar.symLogLinThresh, colorbar.symLogLinScale = setListFromDict(colorbarProperties, keys=["offsetCenter", "powerlaw", "scale", "symLogLinThresh", "symLogLinScale"], default=[0, 2, 'linear', 0.1, 1])
        
        # Label properties
        colorbar.label.text, colorbar.label.size = setListFromDict(colorbarProperties, keys=["label", "labelSize"], default=['', layout.textsize])
        
        if colorbar.ticks.label.text is not None and colorbar.ticks.values is not None:
            try:
                if len(colorbar.ticks.label.text) != len(colorbar.ticks.values):
                    raise ValueError("Keys 'ticksLabels' and 'ticks' in colorbarProperties dict have different lengths. Please provide both keys with the same number of elements. Cheers !")
            except TypeError:
                raise TypeError("Keys 'ticksLabels' and 'ticks' should be lists or None. Please provide a list of values for 'ticks' at the very least if you want to change the colorbar ticks. Cheers !")
        
        
        ###################################################
        #        Compute cmap minimum and maximum         #
        ###################################################
        
        # If colormap normalisation is not a string (i.e. it is a matplotlib.norm instance, we do not update the min and max)
        if type(colorbar.scale) is str:
            if colorbar.cmap.min is None or colorbar.cmap.max is None:
                tmp = []
                for col, typ in zip(data.color, data.type):
                    if typ == 'scatter':
                        tmp.append(col)
                if len(tmp) > 0:
                    if colorbar.cmap.min is None:
                        colorbar.cmap.min = np.min([np.min(i) for i in tmp])
                    if colorbar.cmap.max is None:
                        colorbar.cmap.max = np.max([np.max(i) for i in tmp])
            else:
                if colorbar.cmap.min > colorbar.cmap.max:
                    raise ValueError("Given minimum cmap value with key 'min' in colorbarProperties dict is larger than given maximum cmap value with key 'max'. Please provide value such that min <= max. Cheers !")
            
        #####################################################
        #           Defining colorbar normalisation         #
        #####################################################
        
        if colorbar.offsetCenter != 0 and colorbar.scale == "linear":
            colorbar.scale = 'div'
        
        # Creating the normalize instance for the colorbar
        colorbarDict = {'linear':   {'function':Normalize,    'params': {'vmin':colorbar.cmap.min, 'vmax':colorbar.cmap.max}},
                        'div':      {'function':TwoSlopeNorm, 'params': {'vmin':colorbar.cmap.min, 'vmax':colorbar.cmap.max, 'vcenter':colorbar.offsetCenter}},
                        'log':      {'function':LogNorm,      'params': {'vmin':colorbar.cmap.min, 'vmax':colorbar.cmap.max}},
                        'symlog':   {'function':SymLogNorm,   'params': {'vmin':colorbar.cmap.min, 'vmax':colorbar.cmap.max, 'linthresh':colorbar.symLogLinThresh, 'linscale':colorbar.symLogLinScale}},
                        'powerlaw': {'function':PowerNorm,    'params': {'gamma':colorbar.powerlaw}},
                       }
        
        if type(colorbar.scale) is str:
            colorbar.norm = colorbarDict[colorbar.scale]['function'](**colorbarDict[colorbar.scale]['params'])
        else:
            colorbar.norm = colorbar.scale
    
    
    ###########################################
    #            Legend properties            #
    ###########################################
    
    legend, legend.marker, legend.line, legend.labels, legend.style = gatherThingsUp(), gatherThingsUp(), gatherThingsUp(), gatherThingsUp(), gatherThingsUp()
    if isType(legendProperties, dict, 'legendProperties'):
        
        # Hide legend if no key is provided
        if legendProperties == {}:
            legend.hide = True
        else:
            legend.hide = False
     
        legend.title, legend.titleSize, legend.order = setListFromDict(legendProperties, keys=['title', 'titleSize', 'order'], default=['', layout.textsize, 1000])
        legend.style.shadow, legend.style.fancy, legend.style.bg, legend.style.alpha = setListFromDict(legendProperties, keys=['shadow', 'fancy', 'background', 'transparency'], default=[True, True, None, 1])
        legend.loc, legend.ncols, legend.labels.size, legend.labels.text = setListFromDict(legendProperties, keys=['loc', 'ncols', 'labelSize', 'labels'], default=['best', 1, layout.textsize, ['']*data.nplots]) 
        legend.line.color, legend.marker.edgecolor, legend.marker.facecolor, legend.marker.position , legend.marker.scale = setListFromDict(legendProperties, keys=['lineColor', 'markerEdgeColor', 'markerFaceColor', 'markerPosition', 'markerScale'], default=[None, None, None, 'left', 1.0])
        
        # Checking that given parameters have the correct type
        legend.labels.text               = checkTypeAndChangeValueToList(legend.labels.text,      list, data.nplots)
        legend.line.color                = checkTypeAndChangeValueToList(legend.line.color,       list, data.nplots)
        legend.marker.edgecolor          = checkTypeAndChangeValueToList(legend.marker.edgecolor, list, data.nplots)
        legend.marker.facecolor          = checkTypeAndChangeValueToList(legend.marker.facecolor, list, data.nplots)
        
        # Set default color to black for scatter plots in legend 
        for pos, col, typ, mkfclr in zip(range(data.nplots), data.color, data.type, legend.marker.facecolor):
            if typ == 'scatter':
                col = 'black'
            legend.marker.edgecolor[pos] = col
            legend.line.color[pos]       = col
            
            if mkfclr is None:
                legend.marker.facecolor[pos] = col
            
        # If some text in legend is missing we add empty strings
        llt = len(legend.labels.text)
        if llt < data.nplots:
            legend.labels.text += ['']*(data.nplots-llt)
            
    
    
    ########################################
    #           Output properties          #
    ########################################
    output = gatherThingsUp()
    if isType(outputProperties, dict, 'outputProperties'):
        output.name, output.overwrite, output.tight, output.transparent = setListFromDict(outputProperties, keys=['outputName', 'overwrite', 'tightLayout', 'transparent'], default=[None, False, True, False])
    
    ############################################################################################
    #                         Set subplot and its overall properties                           #
    ############################################################################################
    
    # Generate subplot
    typ = type(numPlot)
    if (typ == list or typ == np.ndarray) and len(numPlot)>=3:
        ax1 = plt.subplot(numPlot[0], numPlot[1], numPlot[2])
    else:
        ax1 = plt.subplot(numPlot)
        
    # Set overall properties
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.set_title(title.label, loc=title.position, pad=title.verticalOffset, fontsize=title.size, color=title.color, fontfamily=title.font, fontweight=title.weight, fontstyle=title.style)
        
    # Set x and y labels
    plt.xlabel(xaxis.label.text, size=xaxis.label.size) 
    plt.ylabel(yaxis.label.text, size=yaxis.label.size)
    
    # Place axes to the correct position
    labelleft       = False
    labelright      = False
    
    if yaxis.pos.lower() == "right":
        ax1.yaxis.set_label_position("right")
        labelright  = True
    elif yaxis.pos.lower() == "left":
        ax1.yaxis.set_label_position("left")
        labelleft   = True
    else:
        raise ValueError("ValueError: given key 'yAxisPos' from dictionary axesProperties is neither 'right' nor 'left'. Please provide one of these values or nothing. Cheers !")
    
    # Place axes to the correct position
    labelbottom     = False
    labeltop        = False
    
    if xaxis.pos.lower() == "bottom":
        ax1.xaxis.set_label_position("bottom")
        labelbottom = True
    elif xaxis.pos.lower() == "top":
        ax1.xaxis.set_label_position("top")
        labeltop    = True
    else:
        raise ValueError("ValueError: given key 'xAxisPos' from dictionary axesProperties is neither 'right' nor 'left'. Please provide one of these values or nothing. Cheers !")

    # Set ticks properties for x and y axes
    ax1.tick_params(axis='x', which='both', direction=xaxis.ticks.direction, labelsize=xaxis.label.size, length=xaxis.ticks.size, labelbottom=labelbottom, labeltop=labeltop)
    ax1.tick_params(axis='y', which='both', direction=yaxis.ticks.direction, labelsize=yaxis.label.size, length=xaxis.ticks.size, labelleft=labelleft, labelright=labelright)

    if not layout.grid.hide:
        plt.grid(zorder=layout.grid.zorder)

    # Setting a list of plots to provide to the user in the end
    listPlots = []
    sct       = None
    
    # List of handles for the legend
    handles = []
    
    for dtx, dty, typ, clr, zrdr, mrkr, mrkrSz, mrkrDgClr, fllstl, trnsprnc, nfll, lnstl, lnwdth, lbl in zip(data.x.data, data.y.data, data.type, data.color, data.zorder, data.marker.type, data.marker.size, data.marker.edgeColor, data.marker.fillstyle, data.transparency, data.marker.unfill, data.line.style, data.line.width, legend.labels.text):
        
        # Set marker facecelor to "none" if marker must be unfilled
        if nfll:
            facecolor = "none"
        else:
            facecolor = clr
        
        #######################################################
        #            Deal with 'plot' type first              #
        #######################################################
        
        if typ == 'plot':
            
            if mrkrSz==0 and lnstl=='None':
                lnstl = '-'
            
            listPlots.append( plt.plot(dtx, dty, label=lbl, marker=mrkr, color=clr, zorder=zrdr, alpha=trnsprnc,
                                       linestyle=lnstl, markerfacecolor=facecolor, markeredgecolor=mrkrDgClr,
                                       markersize=mrkrSz, linewidth=lnwdth)
                            )
            handles.append(copy(listPlots[-1][0]))
            
        #####################################################
        #           Deal with 'scatter' type then           #
        #####################################################
        
        elif typ == 'scatter':
            
            # Create a customisable marker object
            markerObject = MarkerStyle(marker=mrkr, fillstyle=fllstl)
            
            sct          = plt.scatter(dtx, dty, label=lbl, marker=markerObject, zorder=zrdr, 
                                       cmap=colorbar.cmap.name, norm=colorbar.norm, 
                                       alpha=trnsprnc, c=clr, s=mrkrSz, edgecolors=mrkrDgClr)
            
            if nfll:
                sct.set_facecolor('none')
                
            listPlots.append(sct)
        
    #################################################
    #             Deal with the colorbar            #
    #################################################
        
    # If there is at least one 'scatter' or 'mix' type of plot and if the color bar should not be hidden
    if np.any(np.array(data.type) != 'plot') and not colorbar.hide:
        
        # Make colorbar, set orientation, color, rotation and size
        col = plt.colorbar(sct, orientation=colorbar.orientation)
        col.ax.tick_params(labelsize=colorbar.ticks.label.size, length=colorbar.ticks.size,
                           labelcolor=colorbar.ticks.label.color, labelrotation=colorbar.ticks.label.rotation,
                           color=colorbar.ticks.color)
        
        # Set colorbar label
        col.set_label(colorbar.label.text, size=colorbar.label.size)
        
        # Set colorbar ticks if provided
        if colorbar.ticks.values is not None:
            col.set_ticks(colorbar.ticks.values)
            
        # Set colorbar ticks labels if provided
        if colorbar.ticks.label.text is not None:
            if colorbar.orientation == 'vertical':
                col.ax.set_yticklabels(colorbar.ticks.label.text, size=colorbar.ticks.label.size)
            elif colorbar.orientation == 'horizontal':
                col.ax.set_xticklabels(colorbar.ticks.label.text, size=colorbar.ticks.label.size)
    else:
        col = None
        
    #######################################
    #           Deal with legend          #
    #######################################
    
    if not legend.hide:
        
        # Define a markerfirst argument to position the markers in the legend accordingly
        if legend.marker.position == 'right':
            markerfirst = False
        else:
            markerfirst = True
        
        # Plot legend before making changes and get legend handles
        leg = plt.legend(loc=legend.loc, prop={'size': legend.labels.size}, shadow=legend.style.shadow, fancybox=legend.style.fancy, ncol=legend.ncols, markerfirst=markerfirst, 
                                               markerscale=legend.marker.scale, framealpha=legend.style.alpha, 
                                               title=legend.title, title_fontsize=legend.titleSize)
        
        # Map handles to the correct list shape
        legend.handles = []
        cnt            = 0
        for text in legend.labels.text:
            if text is not None:
                legend.handles.append(leg.legendHandles[cnt])
                cnt   += 1
            else:
                legend.handles.append(None)
        
        for h, mkfclr, mkeclr, lc, typ in zip(legend.handles, legend.marker.facecolor, legend.marker.edgecolor, legend.line.color, data.type):
            
            if h is not None:
                if typ in ['plot', 'mix']:
                    try:
                        h.set_color(lc)
                        h.set_markerfacecolor(mkfclr)
                        h.set_markeredgecolor(mkeclr)
                    except:
                        pass
                elif typ == 'scatter':
                    h.set_color(mkfclr)
        
        # Set zorder
        leg.set_zorder(legend.order)
    else:
        leg = None
                
    # Set x and y scales        
    plt.yscale(yaxis.scale)
    plt.xscale(xaxis.scale)
        
    # Define x and y axes limits
    if xaxis.min is not None:
        ax1.set_xlim(left=xaxis.min)
    if xaxis.max is not None:
        ax1.set_xlim(right=xaxis.max)
    if yaxis.min is not None:
        ax1.set_ylim(bottom=yaxis.min)
    if yaxis.max is not None:
        ax1.set_ylim(top=yaxis.max)
        
    # Hiding ticks labels if required
    if xaxis.ticks.hideLabels:
        ax1.axes.get_xaxis().set_ticklabels([])
    if yaxis.ticks.hideLabels:
        ax1.axes.get_yaxis().set_ticklabels([])


    #################################
    #       Deal with output        #
    #################################
    
    if output.name is not None:
        
        if not output.overwrite and isfile(output.name):
            print("File %s already exists but overwritting was disabled, thus writing was stopped. Please either provide a new file name, or set 'overwriting' key in outputProperties dictionary as True to overwrite the already existing file. Cheers !" %output.name)
        else:
            if output.tight:
                bbox_inches = 'tight'
            else:
                bbox_inches = None
            
            plt.savefig(output.name, bbox_inches=bbox_inches, transparent=output.transparent)
    
    #plt.show()
    return ax1, listPlots, leg, col
