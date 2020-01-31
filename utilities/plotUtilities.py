#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:43:25 2019

@author: Wilfried Mercier - IRAP

Acknowledgments to Issa Lina - ENS who spotted a quite tough bug in asManyPlots when it computes the minimum of the cmap when one wants to jointly plots a scatter plot with any other kind of plot.

Functions to automatise as much as possible plotting of data of any kind.
"""

#numpy imports
import numpy as np

#matplotlib imports
#import matplotlib.colors
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, PowerNorm, DivergingNorm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

import astropy.io.fits as fits

from copy import copy
from os.path import isfile

################################################################################################
#                                   Plots functions                                            #
################################################################################################


def display_hst_models(file1, fileout='test.pdf', title='title', cmap='spectral', log=False, show=False):
    '''This function enables to display an image, the associated GALFIT model and residuals.
    
    Authors:
    ----------
    Main contributor  : Epinat Benoit - LAM
    Sec. conttributor : Mercier Wilfried - IRAP
    
    Parameters
    ----------
    log : booelan
        whether to have a log scale or not
    cmap : string
        name of the colormap
    file1: string
        name of GALFIT file that contains the model
    fileout: string
        name of the output file
    show : boolean
        whether to show the image or not
    title: string
        title of the output image
    '''
    
    hdul     = fits.open(file1)
    data     = hdul[1].data
    model    = hdul[2].data
    res      = hdul[3].data
    
    fig      = plt.figure(figsize=(12, 3))
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
    Generates a pdf file with all the galfit images (data, model and residual side by side) found in the given list.
    
    Author:
        Main contributor : Mercier Wilfried - IRAP
    
    Mandatory inputs
    ----------------
        fnamesList : list or string
            list of all the file names (paths included) with the galfit images to be appended inside the tex file. If readFromFile is set True, give the name of a file containing all the different names (one per line) instead.
        pdfOut : string
            name of the output pdf file
        
    Optional inputs
    ---------------
        diverging : bool
            whether to use a diverging norm or not. If true, a linear norm will be used (overriding any log norm)
        cmap : string
            color map to use when plotting
        groupNumbers : list of strings
            the list of groups the galaxies belong to
        log : boolean
            whether to show images as log or not
        readFromFile : boolean
            whether to read the file names from a file or not. If True, the names must be listed as one per line only.
        zeroPoint : float
            value at which the diverging norm will split in two
    """
    
    #get file names from a file if necessary
    if readFromFile:
        fnamesList, groupNumbers = np.genfromtxt(fnamesList, dtype=str, unpack=True)
    
    #computing the number of necessary subfigures
    nbfig    = len(fnamesList)
    
    #gathering the file names only (without path)
    names = np.copy(fnamesList)
    for num, name in enumerate(fnamesList):
        names[num] = name.split('/')[-1]
    
    #creating default values if group numbers are not given
    if (groupNumbers is None) or (len(groupNumbers) != len(fnamesList)):
        groupNumbers = [None]*(nbfig)
    
    #going through the opening and plotting phases
    #first deifning the grid
    fig = plt.figure(figsize=(17, 6*nbfig))
    gs  = gridspec.GridSpec(nbfig, 3, figure=fig)
    
    previousNumber = None
    for num, file, name, gr in zip(range(0, 3*nbfig, 3), fnamesList, names, groupNumbers):
        #fetching fits file extensions with data, model and residual maps
        hdul       = fits.open(file)
        data       = hdul[1].data
        sz         = int(hdul[1].header['NAXIS2'])
        model      = hdul[2].data
        res        = hdul[3].data
        
        #defining a maximum and minimum which are symetrical
        maxi       = np.max([np.max([data, model]), np.abs(np.min([data, model]))])
        mini       = -maxi
        
        #defining norm based on input parameter value
        norm       = None
        if log:
            norm   = SymLogNorm(linthresh=maxi/1e3, vmin=mini, vmax=maxi)
        if diverging:
            mini   = np.min([data, model])
            maxi   = np.max([data, model])
            norm   = DivergingNorm(vmin=mini, vcenter=zeroPoint, vmax=maxi)
        
        #Plotting the three plots side by side
        ax1        = plt.subplot(gs[num])
        plt.grid()
        ax1.title.set_text(name)
        plt.imshow(data, origin='lower', cmap=cmap, interpolation='nearest', vmin=mini, vmax=maxi, norm=norm)
        plt.colorbar(fraction=0.05, shrink=1.)
        
        #adding group info on the plots if a new group is encountered
        if previousNumber != gr:
            print("group", gr)
            previousNumber = gr
            plt.text(0, sz+30, "Group: %s" %(str(gr)), fontsize=20, fontweight='bold')
        
        ax2 = plt.subplot(gs[num+1])
        plt.grid()
        ax2.title.set_text('model')
        plt.imshow(model, origin='lower', cmap=cmap, interpolation='nearest', vmin=mini, vmax=maxi, norm=norm)
        plt.colorbar(fraction=0.05, shrink=1.)
        
        ax3 = plt.subplot(gs[num+2])
        ax3.title.set_text('residual')
        
        maxi = np.max(res)
        mini = np.min(res)
        norm = DivergingNorm(vmin=mini, vmax=maxi, vcenter=zeroPoint)
        plt.imshow(res, origin='lower', cmap=cmap, interpolation='nearest', vmin=mini, vmax=maxi, norm=norm)
        plt.colorbar(fraction=0.05, shrink=1.)
    
    plt.savefig(pdfOut, bbox_inches='tight')
    plt.close()


def asManyHists(numPlot, data, bins=None, weights=None, hideXlabel=False, hideYlabel=False, hideYticks=False, hideXticks=False,
                placeYaxisOnRight=False, xlabel="", ylabel='', color='black',
                label='', zorder=0, textsize=24, showLegend=False, legendTextSize=24,
                xlim=[None, None], locLegend='best', tickSize=24, title='', titlesize=24,
                outputName=None, overwrite=False, tightLayout=True, integralIsOne=None,
                align='mid', histtype='stepfilled', alpha=1.0, cumulative=False, legendNcols=1, hatch=None, orientation='vertical', log=False, stacked=False):

    """
    Function which plots on a highly configurable subplot grid 1D histograms. A list of data can be given to have multiple histograms on the same subplot.

    Input
    -----
    align : 'left', 'mid' or 'right'
        how to align bars respective to their value
    alpha : float
        how transparent the bars are
    bins : int or list of int
        if an integer, the number of bins. If it is a list, edges of the bins must be given.
    color : list of strings/chars/RGBs
        color for the data. It can either be a string, char or RGB value.
    cumulative : boolean
        whether to plot the cumulative distribution (where each bin equals the sum of the values in the previous bins up to this one) or the histogram
    data: numpy array, list of numpy arrays
        the data
    hatch : char
        the hatching pattern
    hideXlabel : boolean
        whether to hide the x label or not
    hideXticks : boolean
        whether to hide the x ticks or not
    hideYlabel : boolean
        whether to hide the y label or not
    hideYticks : boolean
        whether to hide the y ticks or not
    histtype : 'bar', 'barstacked', 'step', 'stepfilled'
        how the histogram is plotted. Bar puts histograms next to each other. Barstacked stacks them. Step plots unfilled histograms. Stepfilled generates a filled histogram by default.
    integralIsOne : boolean or list of boolean
        whether to normalize the integral of the histogram
    label : string
        legend label for the data
    legendNcols : int
        number of columns in the legend
    legendTextSize : int
        size for the legend
    locLegend : string, int
        position where to place the legend
    numPlot : int (3 digits)
        the subplot number
    orientation : str
        orientation of the bars
    outputName : str
        name of the file to save the graph into. If None, the plot is not saved into a file
    overwrite : boolean
        whether to overwrite the ouput file or not
    placeYaxisOnRight : boolean
        whether to place the y axis of the plot on the right or not
    textsize : int
        size for the labels
    showLegend : boolean
        whether to show the legend or not
    tickSize : int
        size of the ticks on both axes
    tightLayout : boolean
        whether to use bbox_inches='tight' if tightLayout is True or bbox_inches=None otherwise
    weights : numpy array of floats or list of numpy arrays
        the weights to apply to each value in data
    xlabel : string
        the x label
    xlim : list of floats/None
        the x-axis limits to use. If None is specified as lower/upper/both limit(s), the minimum/maximum/both values are used
    ylabel : string
        the y label
    ylim : list of floats/None
        the y-axis limits to use. If None is specified as lower/upper/both limit(s), the minimum/maximum/both values are used
    zorder : int, list of ints for many plots
        whether the data will be plot in first position or in last. The lower the value, the earlier it will be plotted
        
    Return current axis, hist values and bins.
    """
    
    ax1 = plt.subplot(numPlot)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.set_title(title, size=titlesize)
    ax1.tick_params(which='both', direction='in', labelsize=tickSize)
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
    
    #set hatching pattern if there is one
    
    if showLegend:
        plt.legend(loc=locLegend, prop={'size': legendTextSize}, shadow=True, fancybox=True, ncol=legendNcols)
        
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
        
    return ax1, n, bns, ptchs

                
def asManyPlots(numPlot, datax, datay, hideXlabel=False, hideYlabel=False, hideYticks=False,
                placeYaxisOnRight=False, xlabel="", ylabel='', marker='o', color='black', plotFlag=True,
                label='', zorder=None, textsize=24, showLegend=False, legendTextSize=24, linestyle='None',
                ylim=[None, None], xlim=[None, None], cmap='Greys', cmapMin=None, cmapMax=None,
                showColorbar=False, locLegend='best', tickSize=24, title='', titlesize=24, 
                colorbarOrientation='vertical', colorbarLabel=None, colorbarTicks=None, colorbarTicksLabels=None,
                colorbarLabelSize=24, colorbarTicksSize=24, colorbarTicksLabelsSize=24,
                outputName=None, overwrite=False, tightLayout=True, linewidth=3,
                fillstyle='full', unfilledFlag=False, alpha=1.0,
                noCheck=False, legendNcols=1, removeGrid=False, markerSize=16, 
                legendMarkerFaceColor=None, legendMarkerEdgeColor=None, legendLineColor=None,
                norm=None, xscale=None, yscale=None):
    """
    Function which plots on a highly configurable subplot grid either with pyplot.plot or pyplot.scatter. A list of X and Y arrays can be given to have multiple plots on the same subplot.
    This function has been developed to be used with numpy arrays or list of numpy arrays (structured or not). Working with astropy tables or any other kind of data structure might or might not work depending on its complexity and behaviour. 
    
    Input
    -----
    alpha : float, list of floats
        indicates the transparency of the data points (1 is plain, 0 is invisible)
    cmap : matplotlib colormap
        the colormap to use for the scatter plot only
    cmapMin: float
        the minmum value for the colormap
    cmapMax: float
        the maximum value for the colormap
    color : list of strings/chars/RGBs/lists of values
        color for the data. For scatter plots, the values must be in numpy array format. For plots, it can either be a string, char or RGB value.
        WARNING: it is highly recommanded to give the color as a list. For instance, if plotting only one plot of black color, you should preferentially use ['black'] rather than 'black'. For, say one plot and one scatter plot, you have to use ['black', yourNumpyArray].
    colorbarLabel : string
        the name to be put next to the colorbar
    colorbarLabelSize : int
        size of the label next to the colorbar
    colorbarOrientation : 'vertical' or 'horizontal'
        specifies if the colorbar must be place on the right or on the bottom of the graph
    colorbarTicks : list of int/float
        specifies the values taken by the ticks which will be printed next to the colorbar
    colorbarTicksLabels : list of string
        specifies the labels associated to the chosen ticks values
    colorbarTicksLabelsSize : int
        size of the labels associated to the chosen ticks
    colorbarTicksSize : int
        size of the chosen ticks
    datax: numpy array, list of numpy arrays
        the x data
    datay : numpy array, list of numpy arrays 
        the y data
    fillstyle : string, list of strings
        which fillstyle use for the markers (see matplotlib fillstyles for more information)
    hideXlabel : boolean
        whether to hide the x label or not
    hideYlabel : boolean
        whether to hide the y label or not
    hideYticks : boolean
        whether to hide the y ticks or not
    label : string
        legend label for the data
    legendLineColor : list of strings/chars/RGBs
        the line color in the legend. If None, uses the plot color (for plots) and black (for scatter plots) as default.
    legendMarkerEdgeColor : list of strings/chars/RGBs
        the color of the edges of each marker in the legend. If None, uses the plot color (for plots) and black (for scatter plots) as default.
    legendMarkerFaceColor : list of strings/chars/RGBs
        the face color (color of the main area) of each marker in the legend. If None, uses the plot color (for plots) and black (for scatter plots) as default.
    legendNcols : int
        number of columns in the legend
    legendTextSize : int
        size for the legend
    linestyle : string, list of strings for many plots
        which line style to use
    linewidth : float
        the width of the line
    locLegend : string, int
        position where to place the legend
    marker : string, char, list of both for many plots
        the marker to use for the data
    markerSize : float or list of floats for scatter plots
        the size of the marker
    noCheck : boolean
        whether to check the given parameters all have the relevant shape or not
    norm : Matplotlib Normalize instance
        the norm of the colormap (for log scale colormaps for instance)
    numPlot : int (3 digits)
        the subplot number
    outputName : str
        name of the file to save the graph into. If None, the plot is not saved into a file
    overwrite : boolean
        whether to overwrite the ouput file or not
    placeYaxisOnRight : boolean
        whether to place the y axis of the plot on the right or not
    plotFlag : boolean, list of booleans for many plots
        if True, plots with pyplot.plot function. If False, use pyplot.scatter
    removeGrid : boolean, list of booleans for many plots
        whether to remove the grid or not
    textsize : int
        size for the labels
    showColorbar : boolean
        whether to show the colorbar for a scatter plot or not
    showLegend : boolean
        whether to show the legend or not
    tickSize : int
        size of the ticks on both axes
    tightLayout : boolean
        whether to use bbox_inches='tight' if tightLayout is True or bbox_inches=None otherwise
    unfilledFlag : boolean, list of booleans
        whether to unfill the points' markers or not
    xlabel : string
        the x label
    xlim : list of floats/None
        the x-axis limits to use. If None is specified as lower/upper/both limit(s), the minimum/maximum/both values are used
    xscale : string
        the scale to use (most used are "linear", "log", "symlog") for the x axis
    ylabel : string
        the y label
    ylim : list of floats/None
        the y-axis limits to use. If None is specified as lower/upper/both limit(s), the minimum/maximum/both values are used
    yscale : string
        the scale to use (most used are "linear", "log", "symlog") for the y axis
    zorder : int, list of ints for many plots
        whether the data will be plot in first position or in last. The lower the value, the earlier it will be plotted
        
    Return current axis and last plot.
    """
    
    try:
        len(numPlot)
        ax1 = plt.subplot(numPlot[0], numPlot[1], numPlot[2])
    except:
        ax1 = plt.subplot(numPlot)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.set_title(title, size=titlesize)
    ax1.tick_params(which='both', direction='in', labelsize=tickSize)
    
    if not removeGrid:
        plt.grid(zorder=1000)
    
    #Checking shape consistency between datax and datay
    shpX = np.shape(datax)
    shpY = np.shape(datay)
    if shpX != shpY:
        exit("X data was found to have shape", shpX, "but Y data seems to have shape", shpY, ".Exiting.")
        
    #If we have an array instead of a list of arrays, transform it to the latter
    try:
        np.shape(datax[0])[0]
    except:
        datax = [datax]
        datay = [datay]
        
    lx = len(datax)
        
    #If we have only one marker/color/zorder/linestyle/label/plotFlag, transform them to a list of the relevant length
    if not noCheck:
        try:
            np.shape(linestyle)[0]
        except:
            linestyle = [linestyle]*lx
    try:
        np.shape(marker)[0]
    except:
        marker = [marker]*lx
    try:
        np.shape(markerSize)[0]
    except:
        markerSize = [markerSize]*lx
        
    try: 
        np.shape(legendMarkerFaceColor)[0]
    except:
        legendMarkerFaceColor = [legendMarkerFaceColor]*lx
        
    try: 
        np.shape(legendMarkerEdgeColor)[0]
    except:
        legendMarkerEdgeColor = [legendMarkerEdgeColor]*lx
        
    try: 
        np.shape(legendLineColor)[0]
    except:
        legendLineColor = [legendLineColor]*lx
    
    if len(color) ==  1:
        color = [color]*lx
     
    if zorder is None:
        zorder = range(1, lx+1)
        
    try:
        np.shape(plotFlag)[0]
    except:
        plotFlag = [plotFlag]*lx
    try:
        np.shape(fillstyle)[0]
    except:
        fillstyle = [fillstyle]*lx
    try:
        np.shape(unfilledFlag)[0]
    except:
        unfilledFlag = [unfilledFlag]*lx
    try:
        np.shape(alpha)[0]
    except:
        alpha = [alpha]*lx
    try:
        np.shape(label)[0]
    except:
        if lx>1:
            if showLegend:
                print("Not enough labels were given compared to data dimension. Printing empty strings instead.")
            label = ''
        label = [label]*lx
    
#     print(color, marker, zorder, linestyle, plotFlag, label)
        
    #hiding labels if required
    if hideXlabel:
        ax1.axes.get_xaxis().set_ticklabels([])
    else:
        plt.xlabel(xlabel, size=textsize)    
    if hideYticks:
        ax1.axes.get_yaxis().set_ticklabels([])
    if not hideYlabel:    
        plt.ylabel(ylabel, size=textsize)
    
    #Place Y axis on the right if required
    if placeYaxisOnRight:
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")

    #Plotting
    tmp     = []
    sct     = None
    
    #list of handels for the legend
    handles = []
    
    # Compute cmap minimum and maximum if there are any scatter plots (it is ineficient because color list contains lists and strings, so we cannot convert it easily to an array to use vectorized function)
    tmp = []
    for col, flag in zip(color, np.array(plotFlag)==False):
        if flag:
            tmp.append(col)
    if len(tmp) > 0:
        cmapMin = np.min(tmp)
        cmapMax = np.max(tmp)
    
    for dtx, dty, mrkr, mrkrSz, clr, zrdr, lnstl, lbl, pltFlg, fllstl, lph, nflldFlg in zip(datax, datay, marker, markerSize, color, zorder, linestyle, label, plotFlag, fillstyle, alpha, unfilledFlag):
        edgecolor = clr
        if nflldFlg:
            facecolor = "none"
        else:
            facecolor=clr
        
        if pltFlg:
            tmp.append(plt.plot(dtx, dty, label=lbl, marker=mrkr, color=clr, zorder=zrdr, alpha=lph,
                           linestyle=lnstl, markerfacecolor=facecolor, markeredgecolor=edgecolor,
                           markersize=mrkrSz, linewidth=linewidth))
            handles.append(copy(tmp[-1][0]))
        else:
            print(clr, cmap)
            markerObject = MarkerStyle(marker=mrkr, fillstyle=fllstl)
            sct = plt.scatter(dtx, dty, label=lbl, marker=markerObject, zorder=zrdr, 
                              cmap=cmap, norm=norm, vmin=cmapMin, vmax=cmapMax, alpha=lph, c=clr, s=mrkrSz)
            print(sct)
            tmp.append(sct)
            
            if nflldFlg:
                sct.set_facecolor('none')
        
    if np.any(np.logical_not(plotFlag)) and showColorbar:
        col = plt.colorbar(sct, orientation=colorbarOrientation)
        col.ax.tick_params(labelsize=colorbarTicksLabelsSize)
        
        if colorbarLabel is not None:
            col.set_label(colorbarLabel, size=colorbarLabelSize)
        if colorbarTicks is not None:
            col.set_ticks(colorbarTicks)
        if colorbarTicksLabels is not None:
            if colorbarOrientation == 'vertical':
                col.ax.set_yticklabels(colorbarTicksLabels, size=colorbarTicksLabelsSize)
            elif colorbarOrientation == 'horizontal':
                col.ax.set_xticklabels(colorbarTicksLabels, size=colorbarTicksLabelsSize)
            
    if showLegend:
        
        def setDefault(data, default):
            for num, i in enumerate(data):
                if i is None:
                    data[num] = default
            return data
        
        if pltFlg:
            for h, mkfclr, mkeclr, lc, c in zip(handles, legendMarkerFaceColor, legendMarkerEdgeColor, legendLineColor, color):
                mkfclr, mkeclr, lc = setDefault([mkfclr, mkeclr, lc], c)
                
                h.set_color(lc)
                h.set_markerfacecolor(mkfclr)
                h.set_markeredgecolor(mkeclr)
            leg = plt.legend(loc=locLegend, prop={'size': legendTextSize}, shadow=True, fancybox=True, ncol=legendNcols, handles=handles)
            
        if not pltFlg:
            leg = plt.legend(loc=locLegend, prop={'size': legendTextSize}, shadow=True, fancybox=True, ncol=legendNcols)
            
            for marker, mkfclr, mkeclr, lc in zip(leg.legendHandles, legendMarkerFaceColor, legendMarkerEdgeColor, legendLineColor):
                mkfclr = setDefault([mkfclr], 'black')
                
                marker.set_color(mkfclr)
                
    if yscale is not None:
        plt.yscale(yscale)
    if xscale is not None:
        plt.xscale(xscale)
        
    #Define Y limits if required
    if ylim[0] is not None:
        ax1.set_ylim(bottom=ylim[0])
#     else:
#         ax1.set_ylim(bottom=ax.get_ylim()[0])
    if ylim[1] is not None:
        ax1.set_ylim(top=ylim[1])
#    else:
#         ax1.set_ylim(top=ax1.get_ylim()[1])
        
    #define X limits if required
    if xlim[0] is not None:
        ax1.set_xlim(left=xlim[0])
#     else:
#         ax1.set_xlim(left=ax.get_xlim()[0])
    if xlim[1] is not None:
        ax1.set_xlim(right=xlim[1])
#     else:
#         ax1.set_xlim(right=ax.get_xlim()[1])

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
                return ax1, tmp
                
        f = open(outputName, 'w')
        
        bbox_inches = None
        if tightLayout:
            bbox_inches = 'tight'
            
        plt.savefig(outputName, bbox_inches=bbox_inches)
    
    return ax1, tmp





def asManyPlots2(numPlot, datax, datay, 
                 dataProperties={}, generalProperties={}, axesProperties={}, titleProperties={}, 
                 colorbarProperties={}, legendProperties={}, outputProperties={},
                 outputName=None, overwrite=False, tightLayout=True):
    
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
                    size of the ticks on the plot and on the colorbar if ther is one. Default is 7.
                    
            Text properties
            ---------------
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
                    
                'xmax' : float
                    maximum value of the x-axis. Default is None, so that matplotlib automatically scales the axes.
                'xmin' : float
                    minimum value of the x-axis. Default is None, so that matplotlib automatically scales the axes.
                'ymax' : float
                    maximum value of the y-axis. Default is None, so that matplotlib automatically scales the axes.
                'ymin' : float
                    minimum value of the y-axis. Default is None, so that matplotlib automatically scales the axes.
                
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
                'background' : str
                    color of the legend background. Default is None so that the default value in your rcParams file will be used (usually white).
                'fancy' : bool
                    whether to have a fancy legend box (round edges) or not. Default is True.
                'shadow' : bool
                    whether to draw a shadow around the legend or not. Default is True.
    
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
        
    Ouputs
    ------
        Return the current subplot as well as a list of all the different plot outputs.
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
        
        if data is not typeToCheck:
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
            defaultVals : list
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
    
    class gatherThingsUp:
        """
        An empty class used to store data within the same object.
        """
        info = "A simple container to gather things up."
    
    
    ##################################################
    #                  General layout                #
    ##################################################
    
    layout, layout.ticks, layout.line = gatherThingsUp(), gatherThingsUp(), gatherThingsUp()
    if isType(generalProperties, dict, 'generalProperties'):
        
        layout.textsize, = setListFromDict(generalProperties, keys=['textsize'], default=[24])
        
        # general ticks properties
        layout.ticks.labelSize, layout.ticks.size, layout.ticks.direction = setListFromDict(generalProperties, keys=['ticksLabelsSize', 'ticksSize', 'tickDirection'], default=[layout.textsize-2, 7, 'in'])
        
        # General properties
        layout.markersize, layout.ticks.hideLabels, layout.scale, layout.grid = setListFromDict(generalProperties, keys=['markersize', 'hideTicksLabels', 'scale', 'hideGrid'], default=[16, False, 'linear', False])

        # General line properties
        layout.line.width, layout.line.style = setListFromDict(generalProperties, keys=['linewidth', 'linestyle'], default=[2, '-'])
    
        
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
        data.line.width, data.line.style = setListFromDict(dataProperties, keys=['linewdith', 'linestyle'], default=[None, None])
        
        # If the user provides a single value, we change it to a list
        data.type          = checkTypeAndChangeValueToList(data.type, list, data.nplots)
        data.transparency  = checkTypeAndChangeValueToList(data.transparency, list, data.nplots)
        data.marker.type   = checkTypeAndChangeValueToList(data.marker.type, list, data.nplots)
        data.marker.unfill = checkTypeAndChangeValueToList(data.marker.unfill, list, data.nplots)
        
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
            
        if data.line.style is None:
            data.line.style       = []
        
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
                else:
                    data.line.style[pos] = str(data.line.style[pos])
            
        # If data.color is not provided, we set plot colors to 'black' and scatter plots points to the same value
        if data.color is None:
            data.color = ['black']*data.nplots
        else:
            setDefault(data.color, default='black')
            
        # Check that marker edge colors are not set if the data points are supposed to be unfilled
        for pos, nfllMrkr in enumerate(data.marker.unfill):
            if nfllMrkr:
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
        colorbar.cmap.name, colorbar.cmap.min, colorbar.cmap.max = setListFromDict(colorbarProperties, keys=["cmap", "min", "max"], default=['Greys', 0, None, None])
        
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
        colorbarDict = {'linear':   {'function':Normalize,     'params': {'vmin':colorbar.cmap.min, 'vmax':colorbar.cmap.max}},
                        'div':      {'function':DivergingNorm, 'params': {'vmin':colorbar.cmap.min, 'vmax':colorbar.cmap.max, 'vcenter':colorbar.offsetCenter}},
                        'log':      {'function':LogNorm,       'params': {'vmin':colorbar.cmap.min, 'vmax':colorbar.cmap.max}},
                        'symlog':   {'function':SymLogNorm,    'params': {'vmin':colorbar.cmap.min, 'vmax':colorbar.cmap.max, 'linthresh':colorbar.symLogLinThresh, 'linscale':colorbar.symLogLinScale}},
                        'powerlaw': {'function':PowerNorm,     'params': {'gamma':colorbar.powerlaw}},
                       }
        
        colorbar.norm = colorbarDict[colorbar.scale]['function'](**colorbarDict[colorbar.scale]['params'])
    
    
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
     
        legend.style.shadow, legend.style.fancy, legend.style.bg = setListFromDict(legendProperties, keys=['shadow', 'fancy', 'background'], default=[True, True, None])
        legend.loc, legend.ncols, legend.labels.size, legend.labels.text = setListFromDict(legendProperties, keys=['loc', 'ncols', 'labelSize', 'labels'], default=['best', 1, layout.textsize, ['']*data.nplots]) 
        legend.line.color, legend.marker.edgecolor, legend.marker.facecolor, legend.marker.position , legend.marker.scale = setListFromDict(legendProperties, keys=['lineColor', 'markerEdgeColor', 'markerFaceColor', 'markerPosition', 'markerScale'], default=[None, None, None, 'left', 1.0])
     
        # Checking that given parameters have the correct type
        legend.labels.text               = checkTypeAndChangeValueToList(legend.labels.text, list, data.nplots)
        legend.line.color                = checkTypeAndChangeValueToList(legend.line.color, list, data.nplots)
        legend.marker.edgecolor          = checkTypeAndChangeValueToList(legend.marker.edgecolor, list, data.nplots)
        legend.marker.facecolor          = checkTypeAndChangeValueToList(legend.marker.facecolor, list, data.nplots)
        
        # Set default color to black for scatter plots in legend 
        for pos, col, typ in zip(range(data.nplots), data.color, data.type):
            if typ == 'scatter':
                col = 'black'
            legend.marker.edgecolor[pos] = col
            legend.marker.facecolor[pos] = col
            legend.line.color[pos]       = col
            
        # If some text in legend is missing we add empty strings
        llt = len(legend.labels.text)
        if llt < data.nplots:
            legend.labels.text += ['']*(data.nplots-llt)
            
    
    
    ########################################
    #           Output properties          #
    ########################################
    output = gatherThingsUp()
    if isType(outputProperties, dict, 'outputProperties'):
        output.name, output.overwrite, output.tight = setListFromDict(outputProperties, keys=['outputName', 'overwrite', 'tightLayout'], default=[None, False, True])
    
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
    
    # Set ticks properties for x and y axes
    ax1.tick_params(axis='x', which='both', direction=xaxis.ticks.direction, labelsize=xaxis.label.size, length=xaxis.ticks.size)
    ax1.tick_params(axis='y', which='both', direction=yaxis.ticks.direction, labelsize=yaxis.label.size, length=xaxis.ticks.size)
        
    # Set x and y labels
    plt.xlabel(xaxis.label.text, size=xaxis.label.size) 
    plt.ylabel(yaxis.label.text, size=yaxis.label.size)
    
    # Hiding ticks labels if required
    if xaxis.ticks.hideLabels:
        ax1.axes.get_xaxis().set_ticklabels([])
    if yaxis.ticks.hideLabels:
        ax1.axes.get_yaxis().set_ticklabels([])   
    
    # Place axes to the correct position
    if yaxis.pos.lower() == "right":
        ax1.yaxis.set_label_position("right")
    elif yaxis.pos.lower() == "left":
        ax1.yaxis.set_label_position("left")
    else:
        raise ValueError("ValueError: given key 'yAxisPos' from dictionary axesProperties is neither 'right' nor 'left'. Please provide one of these values or nothing. Cheers !")
    
    if xaxis.pos.lower() == "bottom":
        ax1.xaxis.set_label_position("bottom")
    elif xaxis.pos.lower() == "top":
        ax1.xaxis.set_label_position("top")
    else:
        raise ValueError("ValueError: given key 'xAxisPos' from dictionary axesProperties is neither 'right' nor 'left'. Please provide one of these values or nothing. Cheers !")

    if not layout.grid:
        plt.grid(zorder=1000)

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
            print('coucou', clr,mrkrSz, lnwdth)
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
                                       cmap=colorbar.cmap.name, norm=colorbar.norm, vmin=colorbar.cmap.min, 
                                       vmax=colorbar.cmap.max, alpha=trnsprnc, c=clr, s=mrkrSz, edgecolors=mrkrDgClr)
            
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
        leg = plt.legend(loc=legend.loc, prop={'size': legend.labels.size}, shadow=True, fancybox=True, ncol=legend.ncols, markerfirst=markerfirst, markerscale=legend.marker.scale)
        
        for h, mkfclr, mkeclr, lc, typ in zip(leg.legendHandles, legend.marker.facecolor, legend.marker.edgecolor, legend.line.color, data.type):
            if typ in ['plot', 'mix']:
                h.set_color(lc)
                h.set_markerfacecolor(mkfclr)
                h.set_markeredgecolor(mkeclr)
            elif typ == 'scatter':
                h.set_color(mkfclr)
        
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
            
            plt.savefig(output.name, bbox_inches=bbox_inches)
    
    #plt.show()
    return ax1, listPlots