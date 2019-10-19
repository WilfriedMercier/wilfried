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
from matplotlib.colors import SymLogNorm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

#astropy imports
import astropy.io.fits as fits

#copy imports
from copy import copy


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


def genMeThatPDF(fnamesList, pdfOut, readFromFile=False, groupNumbers=None, log=True, cmap='bwr'):
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
    cmap : string
        color map to use when plotting
    groupNumbers : list of strings
        the list of groups the galaxies belong to
    log : boolean
        whether to show images as log or not
    readFromFile : boolean
        whether to read the file names from a file or not. If True, the names must be listed as one per line only.
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
            norm   = SymLogNorm(linthresh=1, vmin=mini, vmax=maxi)
        
        #Plotting the three plots side by side
        ax1        = plt.subplot(gs[num])
        ax1.title.set_text(name)
        plt.imshow(data, origin='lower', cmap=cmap, interpolation='nearest', vmin=mini, vmax=maxi, norm=norm)
        plt.colorbar(fraction=0.05, shrink=1.)
        
        #adding group info on the plots if a new group is encountered
        if previousNumber != gr:
            print("group", gr)
            previousNumber = gr
            plt.text(0, sz+30, "Group: %s" %(str(gr)), fontsize=20, fontweight='bold')
        
        ax2 = plt.subplot(gs[num+1])
        ax2.title.set_text('model')
        plt.imshow(model, origin='lower', cmap=cmap, interpolation='nearest', vmin=mini, vmax=maxi, norm=norm)
        plt.colorbar(fraction=0.05, shrink=1.)
        
        ax3 = plt.subplot(gs[num+2])
        ax3.title.set_text('residual')
        maxi = np.max([np.max(res), np.abs(np.max(res))])
        mini = -maxi
        plt.imshow(res, origin='lower', cmap=cmap, interpolation='nearest', vmin=mini, vmax=maxi)
        plt.colorbar(fraction=0.05, shrink=1.)
        
    plt.savefig(pdfOut, bbox_inches='tight')
    plt.close()


def asManyHists(numPlot, data, bins=None, weights=None, hideXlabel=False, hideYlabel=False, hideYticks=False, hideXticks=False,
                placeYaxisOnRight=False, xlabel="", ylabel='', color='black',
                label='', zorder=0, textsize=24, showLegend=False, legendTextSize=24,
                xlim=[None, None], locLegend='best', tickSize=24, title='', titlesize=24,
                outputName=None, overwrite=False, tightLayout=True, integralIsOne=None,
                align='mid', histtype='stepfilled', alpha=1.0, cumulative=False, legendNcols=1, hatch=None, orientation='vertical'):

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
                             cumulative=cumulative, hatch=hatch, orientation=orientation)
    
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
        
        def isOrisNotNone(data, default):
            for num, i in enumerate(data):
                if i is None:
                    data[num] = default
            return data
        
        if pltFlg:
            for h, mkfclr, mkeclr, lc, c in zip(handles, legendMarkerFaceColor, legendMarkerEdgeColor, legendLineColor, color):
                mkfclr, mkeclr, lc = isOrisNotNone([mkfclr, mkeclr, lc], c)
                
                h.set_color(lc)
                h.set_markerfacecolor(mkfclr)
                h.set_markeredgecolor(mkeclr)
            leg = plt.legend(loc=locLegend, prop={'size': legendTextSize}, shadow=True, fancybox=True, 
                             ncol=legendNcols, handles=handles)
            
        if not pltFlg:
            leg = plt.legend(loc=locLegend, prop={'size': legendTextSize}, shadow=True, fancybox=True, 
                             ncol=legendNcols)
            
            for marker, mkfclr, mkeclr, lc in zip(leg.legendHandles, legendMarkerFaceColor, legendMarkerEdgeColor, legendLineColor):
                mkfclr = isOrisNotNone([mkfclr], 'black')
                
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