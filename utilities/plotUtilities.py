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
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, PowerNorm, DivergingNorm
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
            leg = plt.legend(loc=locLegend, prop={'size': legendTextSize}, shadow=True, fancybox=True, 
                             ncol=legendNcols, handles=handles)
            
        if not pltFlg:
            leg = plt.legend(loc=locLegend, prop={'size': legendTextSize}, shadow=True, fancybox=True, 
                             ncol=legendNcols)
            
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
                 dataProperties, generalProperties={}, axesProperties={}, titleProperties={}, colorbarProperties={}, legendProperties={},
                marker='o',
                zorder=None, linestyle='None',
                outputName=None, overwrite=False, tightLayout=True, linewidth=3,
                fillstyle='full', unfilledFlag=False,
                removeGrid=False, markerSize=16):
    
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
            
            One may want to plot their data differently on the same plot (for instance one usual line plot and one scatter plot). This can be set in dataProperties dictionnary using the key 'type' (see below for more information).
            asManyPlots2 provide three kinds of plots:
                - usual plot (line or points) with a potential global color set in the dictionnary dataProperties with the key 'color'
                - scatter plot (only points, no line), where the color (set with 'color' key as well) and size of points (set with 'size' key in dataProperties dict) can vary from point to point to show any variation with a third dimension (refered as the z component in the following)
                - 'mix' type of plot where lines only are plotted and colour coded according to a third value (also set with 'color' key in plotProperties dict). For this kind of plot, each array within the list will correspond to a single line.
            
        datay : list of numpy arrays/lists
            list of y-axis data which should be plotted. See datax description for more details.
            
    Optional inputs
    ---------------
    
    Most optional inputs have been gathered within dictionnary structures. This was done in order to reduce the number of optional parameters in the function declaration. Unless provided, default values will be used.
    
    For those who are not familiar with dictionnaries, here we provide an example. Say, one wants to put labels on the x and y axes. This can be done by providing values to 'xlabel' and 'ylabel' keys of axesProperties dictionnary.
    Thus, one would write axesProperties={'xlabel':'this is my x label', 'ylabel':'this is another text for the y label'}.
    
        dataProperties : dict
            dictionnary gathering all tunable properties related to the data. See the list below for the complete list of dicionnary keys.
            
            Data points related properties
            ------------------------------
            'color' : list of str and/or arrays
                list of colors for each plot. Each given color is mapped to the corresponding plot assuming 'color' and datax and datay are sorted in the same order. Default is 'black' for any kind of plot.
                    - for simple aand 'mix' plots, color names must be provided
                    - for scatter plots, a list/array of values must be provided. This is because scatter plots actually map values to a color range and will apply a color to each point separately using this range. 
            
            'transparency' : float, list of floats between 0 and 1
                transparency of the data points (1 is plain, 0 is invisible). Default is 1 for any kind of plot.
                
            Plot related properties
            -----------------------
            'type' : list of 'plot', 'scatter' and/or 'mix'
                list of plot type for each array in datax and datay. Default is 'plot' for every data. One value per array must be provided. Three values are possible:
                    - 'plot' to plot a simple plot (line or points) with a single color applied to all the data points
                    - 'scatter' to plot a scatter plot (only points, no line) where the points are colour coded according to some array of the same dimension as datax and datay provided in the 'color' key
                    - 'mix' to plot a line plot only where each line will have a different colour
    
    
        generalProperties : dict
            dictionnary gathering all tunable general properties. See the list below for the complete list of dictionnary keys.
            
            Ticks related keys
            ------------------
                'hideTicksLabels' : bool
                    whether to hide ticks labels. Default is False.
                'tickDirection' : 'in', 'out' or 'inout'
                    direction of the ticks on the plot. Default is 'inout'.
                    
            Text related keys
            -----------------
                'textsize' : int
                    overall text size in points. Default is 24.
                    
            Scale related keys
            ------------------
                'scale' : str
                    scale of both axes
    
    
        axesProperties : dict
            dictionnary gathering all tunable axes properties. See the list below for the complete list of dictionnary keys.
            
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
                    maximum value of the x-axis. Default is None, so that the maximum value in the x data is used.
                'xmin' : float
                    minimum value of the x-axis. Default is None, so that the minimum value in the x data is used.
                'ymax' : float
                    maximum value of the y-axis. Default is None, so that the maximum value in the y data is used.
                'ymin' : float
                    minimum value of the y-axis. Default is None, so that the minimum value in the y data is used.
                
            Position related keys
            ---------------------
                'xAxisPos' : 'bottom' or 'top'
                    where to place the main x-axis. Default is 'bottom'.
                'yAxisPos' : 'left' or 'right'
                    where to place the main y-axis. Default is 'left'.   
                    
        titleProperties : dict
            dictionnary gathering all tunable main title properties. See the list below for the complete list of dictionnary keys.
            
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
            dictionnary gathering all tunable colorbar properties. See the list below for the complete list of dictionnary keys.
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
                'ticksLabels' : list
                    list of ticks labels (str, int, float, etc.) for the colorbar. Default value is None, so that the ticks values are plotted on the colorbar.
                    This key can be used to change the colorbar ticks printed on the figure, without changing the scatter plot z values.
                'ticksLabelsSize' : int
                    size of the labels associated to the ticks. Default is 24.
                'ticksSize' : int
                    size of the ticks in points. Default is 24.
                    
        legendProperties : dict
            dictionnary gathering all tunable legend properties. See the list below for the complete list of dictionnary keys.
            By default, the legend will not appear. To activate it, set 'hideLegend' key value to False.
            
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
                'size' : int
                    size of the text in the legend. Default is 24.
            
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
    
                
    

    
    fillstyle : string, list of strings
        which fillstyle use for the markers (see matplotlib fillstyles for more information)
    linestyle : string, list of strings for many plots
        which line style to usereturn TypeError
    linewidth : float
        the width of the line
    marker : string, char, list of both for many plots
        the marker to use for the data
    markerSize : float or list of floats for scatter plots
        the size of the marker
    
    outputName : str
        name of the file to save the graph into. If None, the plot is not saved into a file
    overwrite : boolean
        whether to overwrite the ouput file or not
    removeGrid : boolean, list of booleans for many plots
        whether to remove the grid or not
    showLegend : boolean
        whether to show the legend or not
    tightLayout : boolean
        whether to use bbox_inches='tight' if tightLayout is True or bbox_inches=None otherwise
    unfilledFlag : boolean, list of booleans
        whether to unfill the points' markers or not
    zorder : int, list of ints for many plots
        whether the data will be plot in first position or in last. The lower the value, the earlier it will be plotted
        
    Return current axis and last plot.
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
            data = [data]*length
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
        for num, i in enumerate(data):
            if i is None:
                data[num] = default
        return data
    
    def setListFromDict(dictionnary, keys=None, defaultVals=None):
        out = []
        for k, df in zip(keys, defaultVals):
            if k in dictionnary:
                out.append(dictionnary[k])
            else:
                out.append(df)
        return out
    
    class gatherThingsUp:
        """
        An empty class used to store data within the same object.
        """
        info = "A simple container to gather things up."
        
        
    ################################################################################################
    #                  Checking datax and datay are lists with a similar shape                     #
    ################################################################################################
    
    isType(datax, list, "datax")
    isType(datay, list, "datay")
    
    # Checking shape consistency between datax and datay
    shpX = np.shape(datax)
    shpY = np.shape(datay)
    if shpX != shpY:
        raise ValueError("Shape inconsistency: datax has shape %s but datay has shape %s. Both datax and datay should be lists with the same shape. Please provide data with similar shape. Cheers !" %(str(shpX), str(shpY)))
    
    
    ############################################################################################
    #           Checking input type and setting values from optional dictionnaries             #
    ############################################################################################
    
    ################################################################################################
    #                Gathering data properties into a single object for simplicity                 #
    ################################################################################################   
    
    # If everything went fine, we can define a few general properties useful later on
    data, data.x, data.y = [gatherThingsUp()]*3
    data.x.data          = datax
    data.y.data          = datay
    data.x.min           = np.min(data.x.data)
    data.x.max           = np.max(data.x.data)
    data.y.min           = np.min(data.y.data)
    data.y.max           = np.max(data.y.data)
    data.nplots          = len(data.x.data)

    if isType(dataProperties, dict, 'dataProperties'):
        data.color, data.type, data.transparency = setListFromDict(dataProperties, keys=['color', 'type', 'transparency'], default=[None, ['plot']*data.nplots, [1]*data.nplots])
        
        # If the user provides a single value, we change it to a list
        checkTypeAndChangeValueToList(data.type, list, data.nplots)
        checkTypeAndChangeValueToList(data.transparency, list, data.nplots)
            
        # If data.color is not provided, we set plot colors to 'black' and scatter plots points to the same value
        if data.color is None:
            data.color = [[0]*len(data.x.data[num]) if data.color[num]=='scatter' else 'black' for num in range(data.nplots)]
            # We override given cmap since we want scatter plot points to be black as well
            colorbarProperties['cmap'] = 'Greys_r'
    
    # General layout properties
    layout = gatherThingsUp()
    if isType(generalProperties, dict, 'generalProperties'):
        layout.textsize, layout.hideTicksLabels, layout.scale, layout.tickDirection = setListFromDict(generalProperties, keys=['textsize', 'hideTicksLabels', 'scale', 'tickDirection'], default=[24, False, 'linear', 'in'])
    
    # Axes properties dict
    xaxis = gatherThingsUp()
    yaxis = gatherThingsUp()
    if isType(axesProperties, dict, 'axesProperties'):
        xaxis.label, yaxis.label, xaxis.hideTicksLabels, yaxis.hideTicksLabels, xaxis.tickSize, yaxis.tickSize, xaxis.size, yaxis.size, xaxis.scale, yaxis.scale, xaxis.pos, yaxis.pos, xaxis.tickDirection, yaxis.tickDirection, xaxis.min, xaxis.max, yaxis.min, yaxis.max = setListFromDict(axesProperties, keys=["xlabel", "ylabel", "hideXticks", "hideYticksLabels", "xTickSize", "yTickSize", "xLabelTextSize", "yLabelTextSize", "xscale", "yscale", "xAxisPos", "yAxisPos", "xTickDirection", "yTickDirection", "xmin", "xmax", "ymin", "ymax"], default=['', '', False, False, layout.size, layout.size, layout.size, layout.size, layout.scale, layout.scale, "bottom", "left", layout.tickDirection, layout.tickDirection, data.x.min, data.x.max, data.y.min, data.y.max])
        
    # Title properties dict
    title = gatherThingsUp()
    if isType(titleProperties, dict, 'titleProperties'):
        title.color, title.font, title.label, title.size, title.style, title.weight, title.position, title.verticalOffset = setListFromDict(titleProperties, keys=["color", "font", "label", "size", "style", "weight", "position", "verticalOffset"], default=['black', 'sans-serif', '', 26, 'normal', 'regular', 'center', None])
    
    # Colormap properties dict
    colorbar = gatherThingsUp()
    if isType(colorbarProperties, dict, 'colorbarProperties'):
        colorbar.hide, colorbar.orientation, colorbar.cmap, colorbar.offsetCenter, colorbar.min, colorbar.max, colorbar.ticksLabels, colorbar.ticksLabelsSize, colorbar.powerlaw, colorbar.scale, colorbar.symLogLinThresh, colorbar.symLogLinScale, colorbar.label, colorbar.labelSize, colorbar.ticksSize, colorbar.ticks = setListFromDict(colorbarProperties, keys=["hide", "orientation", "cmap", "min", "max", "offsetCenter", "ticks", "ticksLabels", "powerlaw", "scale", "symLogLinThresh", "symLogLinScale", "label", "labelSize", "ticksLabelsSize"], default=[False, 'vertical', 'Greys', 0] + [None]*4 + ['linear', 0.1, 1, ''] + [24]*2)
        
        # TICKS AND TICKSLABELS ARE NONE, CHECK IF THIS IS OKAY
        
        ###################################################
        #        Compute cmap minimum and maximum         #
        ###################################################
        
        tmp = []
        for col, typ in zip(data.color, data.type):
            if typ == 'scatter':
                tmp.append(col)
        if len(tmp) > 0:
            colorbar.min = np.min(tmp)
            colorbar.max = np.max(tmp)
        
        #####################################################
        #           Defining colorbar normalisation         #
        #####################################################
        
        if colorbar.offsetCenter != 0 and colorbar.scale == "linear":
            colorbar.scale = 'div'
        
        # Creating the normalize instance for the colorbar
        colorbarDict = {'linear':   {'function':Normalize,     'params': {'vmin':colorbar.min, 'vmax':colorbar.max}},
                        'div':      {'function':DivergingNorm, 'params': {'vmin':colorbar.min, 'vmax':colorbar.max, 'vcenter':colorbar.offsetCenter}},
                        'log':      {'function':LogNorm,       'params': {'vmin':colorbar.min, 'vmax':colorbar.max}},
                        'symlog':   {'function':SymLogNorm,    'params': {'vmin':colorbar.min, 'vmax':colorbar.max, 'linthresh':colorbar.symLogLinThresh, 'linscale':colorbar.symLogLinScale}},
                        'powerlaw': {'function':PowerNorm,     'params': {'gamma':colorbar.gamma}},
                       }
        
        colorbar.norm = colorbarDict[colorbar.scale]['function'](**colorbarDict[colorbar.scale]['params'])
        
            
    
    # Legend properties dict
    legend = gatherThingsUp()
    if isType(legendProperties, dict, 'legendProperties'):
        legend.loc, legend.ncols, legend.size, legend.labels, legend.lineColor, legend.markerEdgeColor, legend.markerFaceColor = setListFromDict(legendProperties, keys=['loc', 'ncols', 'size',' labels', 'lineColor', 'markerEdgeColor', 'markerFaceColor'], default=['best', 1, 24, ['']*data.nplots] + [None]*3) 
    
    
    ############################################################################################
    #                         Set subplot and its overall properties                           #
    ############################################################################################
    
    # Generate subplot
    typ = type(numPlot)
    if typ == list or typ == np.ndarray:
        ax1 = plt.subplot(numPlot[0], numPlot[1], numPlot[2])
    else:
        ax1 = plt.subplot(numPlot)
        
    # Set overall properties
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.set_title(title.label, loc=title.position, pad=title.verticalOffset, fontsize=title.size, color=title.color, fontfamily=title.font, fontweight=title.weight, fontstyle=title.style)
    
    # Set ticks properties for x and y axes
    ax1.tick_params(axis='x', which='both', direction=xaxis.tickDirection, labelsize=xaxis.tickSize)
    ax1.tick_params(axis='y', which='both', direction=yaxis.tickDirection, labelsize=yaxis.tickSize)
    
    if not removeGrid:
        plt.grid(zorder=1000)
        
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
        
    # Set x and y labels
    plt.xlabel(xaxis.label, size=xaxis.size) 
    plt.ylabel(yaxis.label, size=yaxis.size)
    
    # Hiding ticks labels if required
    if xaxis.hideTicks:
        ax1.axes.get_xaxis().set_ticklabels([])
    if yaxis.hideTicks:
        ax1.axes.get_yaxis().set_ticklabels([])   
    
    # Place axes to the correct position
    if yaxis.pos.lower() == "right":
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
    elif yaxis.pos.lower() == "left":
        ax1.yaxis.tick_left()
        ax1.yaxis.set_label_position("left")
    else:
        print("ValueError: given key 'yAxisPos' from dictionnary 'axesProperties' is neither 'right' nor 'left'. Please provide one of these values or nothing. Cheers !")
        return ValueError
    
    if xaxis.pos.lower() == "bottom":
        ax1.xaxis.tick_bottom()
        ax1.xaxis.set_label_position("bottom")
    elif xaxis.pos.lower() == "top":
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position("top")
    else:
        print("ValueError: given key 'xAxisPos' from dictionnary 'axesProperties' is neither 'right' nor 'left'. Please provide one of these values or nothing. Cheers !")
        return ValueError

    #Plotting
    tmp     = []
    sct     = None
    
    #list of handels for the legend
    handles = []
    
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
        if pltFlg:
            for h, mkfclr, mkeclr, lc, c in zip(handles, legendMarkerFaceColor, legendMarkerEdgeColor, legendLineColor, color):
                mkfclr, mkeclr, lc = setDefault([mkfclr, mkeclr, lc], c)
                
                h.set_color(lc)
                h.set_markerfacecolor(mkfclr)
                h.set_markeredgecolor(mkeclr)
            leg = plt.legend(loc=locLegend, prop={'size': legendTextSize}, shadow=True, fancybox=True, 
                             ncol=legendNcols, handles=handles)
            
        if not pltFlg:
            leg = plt.legend(loc=locLegend, prop={'size': legendTextSize}, shadow=True, fancybox=True, 
                             ncol=legendNcols)
            
            for marker, mkfclr, mkeclr, lc in zip(leg.legendHandles, legendMarkerFaceColor, legendMarkerEdgeColor, legendLineColor):
                mkfclr = setDefault([mkfclr], 'black')
                
                marker.set_color(mkfclr)
        
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