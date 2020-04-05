#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:18:55 2019

@author: Wilfried - IRAP
"""

import tkinter            as     tk
from   tkinter            import messagebox
from   tkinter.filedialog import askopenfilenames
from   tkinter            import ttk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure, Axes
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, PowerNorm, DivergingNorm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import astropy.io.fits as fits
import numpy as np
import copy


class container:
    '''Similar to a simple C struct'''
    
    def __init__(self):
        info = 'A simple container'

# Global variables
DICT_MODELS = {'deVaucouleur' : 'de Vaucouleur', 
               'edgeOnDisk'   : 'Edge on disk', 
               'expDisk'      : 'Exponential disk',
               'ferrer'       : 'Ferrer', 
               'gaussian'     : 'Gaussian', 
               'king'         : 'King', 
               'moffat'       : 'Moffat', 
               'nuker'        : 'Nuker', 
               'psf'          : 'PSF', 
               'sersic'       : 'Sersic', 
               'sky'          : 'Sky'}

FONT        = 'Arial'

CIRCLE      = """
#define circle_width 13
#define circle_height 13
static unsigned char circle_bits[] = {
   0xf0, 0x01, 0x08, 0x02, 0x04, 0x04, 0x02, 0x08, 0x01, 0x10, 0x01, 0x10,
   0x01, 0x10, 0x01, 0x10, 0x01, 0x10, 0x02, 0x08, 0x04, 0x04, 0x08, 0x02,
   0xf0, 0x01};"""

CIRCLE_INV  = """
#define circle_inv_width 13
#define circle_inv_height 13
static unsigned char circle_inv_bits[] = {
   0xf0, 0x01, 0xf8, 0x03, 0xfc, 0x07, 0xfe, 0x0f, 0xff, 0x1f, 0xff, 0x1f,
   0xff, 0x1f, 0xff, 0x1f, 0xff, 0x1f, 0xfe, 0x0f, 0xfc, 0x07, 0xf8, 0x03,
   0xf0, 0x01};"""





class singlePlotFrame:
    '''A frame for a single plot, with all its properties and methods.'''
    
    
    def __init__(self, parent, root, data=None, name='', title=None, figsize=(3.4, 3.4), bgColor='beige'):
        
        # General properties
        self.bdOn           = 'black'
        self.bdOff          = bgColor
        
        self.selected       = False
        self.data           = data
        self.shape          = (0, 0)
        self.name           = name
        
        self.parent         = parent
        self.root           = root
        
        # PA line properties
        self.paLine         = container()
        self.paLine.posx    = [None, None]
        self.paLine.posy    = [None, None]
        self.paLine.drawing = False
        self.paAngle        = None
        
        # Making a figure
        self.bgColor        = bgColor
        self.figure         = Figure(figsize=figsize, tight_layout=True, facecolor=self.bgColor)
            
        # Creating an axis
        self.invertAx       = {'x':False, 'y':False}
        self.ax             = self.figure.add_subplot(111)
        self.ax.yaxis.set_ticks_position('both')
        self.ax.xaxis.set_ticks_position('both')
        self.ax.tick_params(which='both', direction='in', labelsize=12)
        
        # Adding a title
        if title is not None and type(title) is str:
            self.title = title
            self.ax.set_title(self.title)
            
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.draw()
        
        # Linking to events
        self.canvas.mpl_connect('button_press_event',  self.onClick)
        self.canvas.mpl_connect('motion_notify_event', self.onMove)
        self.canvas.mpl_connect('figure_enter_event',  self.onFigure)
        
        
    def borderOnOff(self, on=True):
        
        if not on :
            self.parent.config({'bg':self.bdOff})
            self.selected                     = False
            self.root.bottomPane.numSelected -= 1
        else:
            self.parent.config({'bg':self.bdOn})
            self.selected                     = True
            self.root.bottomPane.numSelected += 1
        return


    def changeBorder(self):
        '''Adds or remove a border around the figure canvas.'''
        
        # Case 1: the plot is selected, we change the border color back to the frame color
        if self.parent['bg'] == self.bdOn:
            self.borderOnOff(on=False)
        # Case 2: the plot is unselected and we change the border color to the 'On' color (default is black)
        else:
           self.borderOnOff(on=True )
        return
    
    
    def invertAxis(self, axis='x'):
        '''
        Invert one of the axes.

        Optional inputs
        ---------------
            axis : 'x' or 'y'
                which axis to invert. Default is x-axis.
        '''
        
        if self.selected:
            self.invertAx[axis] = not self.invertAx[axis]
            
            if axis == 'x':
                self.ax.set_xlim(self.ax.get_xlim()[::-1])
            elif axis == 'y':
                self.ax.set_ylim(self.ax.get_ylim()[::-1])
                
            self.canvas.draw()
        return
        
        
    def onClick(self, event):
        '''Actions taken when the user clicks on one of the figures.'''
        
        # If state is default, we draw a border (technically we change the color of the border) around a plot frame to indicate that it has been selected
        if self.root.state == 'default':
            self.selectIt()
        else:
            if not self.paLine.drawing:
                self.paLine.posx[0] = event.xdata
                self.paLine.posy[0] = event.ydata
                self.paLine.drawing = True
            else:
                self.paLine.drawing = False
                self.paLine.line.set_data([self.paLine.posx, self.paLine.posy])
                
                # Computing the PA angle (between -90° and +90°, counting from the vertical axis counter clockwise)
                self.paAngle        = np.arctan((self.paLine.posy[1]-self.paLine.posy[0])/(self.paLine.posx[1]-self.paLine.posx[0]))
                if self.paAngle>=0:
                    self.paAngle    = np.pi/2 - self.paAngle
                else:
                    self.paAngle   = -(np.pi/2 + self.paAngle)
                self.paAngle       *= 180/np.pi # convert to degrees
                
                # Update the file manager with this new data
                self.updateFileManagerValues(valNames=['PA'], values=[np.round(self.paAngle, 2)])
                
                self.canvas.draw()
                
        # if at least one graph is selected, we enable the x and y invert widgets
        if self.root.bottomPane.numSelected == 1:
            self.root.topPane.updateInvertWidgets(state='normal')
        elif self.root.bottomPane.numSelected == 0:
            self.root.topPane.updateInvertWidgets(state='disabled')
        return
                
                
    def onFigure(self, event):
        '''Set focus onto the main window when the cursor enters it.'''
        
        self.root.parent.focus()
        return
            
            
    def onMove(self, event):
        '''Update the image name, x and y positions of the mouse cursor printed on the top pane.'''
        
        # If image title in top Frame is different, set new name
        if self.root.topPane.hover.var.get() != self.title:
            self.root.topPane.hover.var.set( 'Current image: %s' %self.title)
            
        # Update x and y pos in top Frame
        if event.xdata is not None:
            self.root.topPane.hover.xvar.set('x: %.2f' %event.xdata)
            
        if event.ydata is not None:
            self.root.topPane.hover.yvar.set('y: %.2f' %event.ydata)
            
        # Draw PA line
        if self.paLine.drawing:
            self.paLine.posx[1] = event.xdata
            self.paLine.posy[1] = event.ydata
            self.paLine.line.set_data([self.paLine.posx, self.paLine.posy])
            self.canvas.draw()
            
        # Set active singlePlot frame for key press events which are not canvas dependent
        if self.root.bottomPane.activeSingleFrame is not self:
            self.root.bottomPane.activeSingleFrame = self
            
        # Update the magnifier on the top right corner
        self.root.topPane.updateMagnifier(self.data, shape=self.shape, xpos=event.xdata, ypos=event.ydata, invertAxes=self.invertAx)
        return
    
    
    def updateFileManagerValues(self, valNames=[], values=[]):
        '''Update the file manager with some new values'''
        
        # Find position in the heading list of all the values to change
        posList     = []
        for name in valNames:
            posList.append(self.root.topPane.fWindow.window.treeview['columns'].index(name))
        
        # Get current values
        currentVal  = self.root.topPane.fWindow.window.treeview.item(self.root.topPane.fWindow.window.itemDict[self.name])['values']
        
        # Update each value
        for pos, val in zip(posList, values):
            currentVal[pos] = val
            
        # Update the corresponding item once everything has been updated
        self.root.topPane.fWindow.window.treeview.item(self.root.topPane.fWindow.window.itemDict[self.name], values=currentVal)
        return
    
    
    def updateImage(self, newData, mini=None, maxi=None, cmap='bwr'):
        '''
        Update the image.
        
        Inputs
        ------
            newData : list/numpy array
                new data to plot
            mini : float
                minimum value for the color mapping
            maxi : float
                maximum value for the color mapping
            cmap : str
                name of the colormap to use
        '''
        
        norm             = DivergingNorm(vcenter=0, vmin=mini, vmax=maxi)
        self.data        = newData
        self.shape       = newData.shape
        self.ax.clear()
        self.ax.set_title(self.title)
        self.im          = self.ax.imshow(newData, cmap=cmap, norm=norm, origin='lower')
        self.paLine.line = self.ax.plot([], [], 'k-')[0]
        self.canvas.draw()
        return
    
    
    def selectIt(self):
        ''''(Un)select the plot, that is change the border and either select or (un)select it in the file window.'''
        
        # Changing border first
        self.changeBorder()
        
        # (Un)selecting the corresponding line in the file list window
        if self.selected:
            self.root.topPane.fWindow.window.selectLine(self.name, select=True)
        else:
            self.root.topPane.fWindow.window.selectLine(self.name, select=False)
            
        # Changing the selection value
        self.selected = not self.selected
        
        # Updating number of selected frames
        if self.selected:
            self.root.bottomPane.numSelected += 1
        else:
            self.root.bottomPane.numSelected -= 1
        return
    


class graphFrame:
    def __init__(self, parent, root, bgColor='beige'):
        
        ##########################################################
        #                 Setting instance variables             #
        ##########################################################
        
        self.parent            = parent
        self.root              = root
        self.bdSize            = 5
        self.bgColor           = bgColor
        
        self.nbFrames          = 0
        self.nbFramesLine      = 0
        self.scaleFactor       = 3/1084
        
        # Updated number of frames per line when the window is resized
        self.newNbFramesLine  = self.nbFramesLine
        
        # Because matplotlib key press event is not Tk canvas dependent (i.e it links it to the last drawn Tk canvas), we define an active singleFrame instance
        # which we use to update graphs when the focus is on them (CQFD ;))
        self.activeSingleFrame = None
        self.numSelected       = 0
        
        self.frameList         = []
        self.plotList          = []
        self.linkNameToPlot    = {}
        self.originalFigSize   = (3.4, 3.4)
        
        #############################################
        #       Generating frames and canvas        #
        #############################################
        
        # Container objects
        self.frame        = container()
        
        # We need to draw a canvas to place widgets within
        self.canvas       = tk.Canvas(self.parent, bd=0, bg=self.bgColor)
        self.scrollbar    = tk.Scrollbar(self.parent, orient="vertical", command=self.canvas.yview, width=5, bg='black')
        
        # Configure scrollbar on the right
        self.canvas.configure(scrollregion=self.canvas.bbox('all'), yscrollcommand=self.scrollbar.set, bg=self.bgColor)
        
        # Define a frame within the canvas to hold widgets
        self.frame.obj    = tk.Frame(self.canvas, bg=self.bgColor)
        self.frame.id     = self.canvas.create_window(0, 0, anchor='nw', window=self.frame.obj)
        self.canvas.bind("<Configure>", self.resizeFrame)
  
        # Draw widgets
        self.scrollbar.pack( fill='both', side='left')
        self.canvas.pack(    fill='both', expand='yes')
        
        
    def makeFrames(self, nb=0, titles=[], names=[]):
        ''''Create the frames when uploading images'''
        
        self.nbFrames = nb
        self.updateFramesPerLine(self.canvas.size)
        
        if nb != len(titles):
            raise Exception('Given number of frame titles does not match number of loaded frames.')
            
        if nb != len(names):
            raise Exception('Given number of frame names does not match number of loaded frames.')

        for i, title, name in zip(range(nb), titles, names):
            self.frameList.append(tk.Frame(self.frame.obj, bd=self.bdSize, bg=self.bgColor))
            self.plotList.append(singlePlotFrame(self.frameList[i], self.root, title=title, name=name, figsize=self.originalFigSize, bgColor=self.bgColor))
        return
    
    
    def placeFrames(self, fill='x', expand=True, side=tk.LEFT, padx=5, pady=5, anchor=tk.N, sticky=tk.N+tk.E+tk.W+tk.S):
        '''Place the created frames into the main frame'''
        
        # If the new number of frames per line is different from the old one we update it and (re)place the frames
        if self.newNbFramesLine != self.nbFramesLine:
            
            self.nbFramesLine = self.newNbFramesLine
            for pos, frame in enumerate(self.frameList):
                colPos  = pos%self.nbFramesLine
                linePos = pos//self.nbFramesLine
                frame.grid(row=linePos, column=colPos, padx=padx, pady=pady, sticky=sticky)
                
            # Always update canvas scrollregions otherwise it does weird stuff
            self.canvas.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        return
           
    
    def updateFramesPerLine(self, winSize):
        '''Update the number of allowed frames per each line'''
        
        # Compute the new number of frames per line
        self.newNbFramesLine     = int(self.scaleFactor*winSize)
        
        # If it falls bellow 1, we force it to be 1
        if self.newNbFramesLine < 1:
            self.newNbFramesLine = 1
        return
    
    
    def resetFrames(self):
        '''Destroy plots and frames, and reset their corresponding lists'''
        
        self.nbFrames         = 0
        self.nbFramesLine     = 0
        
        for frame in self.frameList:
            frame.grid_forget()
            frame.destroy()
        
        self.frameList        = []
        self.plotList         = []
        self.linkNameToPlot   = {}
        return
    
    
    def resizeFrame(self, event):
        '''Resize the main frame when the canvas is resised'''
        
        self.updateFramesPerLine(event.width)
        self.canvas.size = event.width
        self.placeFrames()
        self.canvas.itemconfig(self.frame.id, width=event.width)
        
        return
            


class topFrame:
    '''
    Top frame with widgets used to import data and get additional information.
    '''
    def __init__(self, parent, root, bgColor='grey'):
        '''
        Inputs
        ------
            parent : tk object
                parent object to put the widget within
            root : tk.Tk instance
                main application object
        '''
        
        self.parent      = parent
        self.root        = root
        self.bgColor     = bgColor
        
        # Visible (in input) and list of opened file names
        self.initDir     = tk.StringVar(value='/home/wilfried/Thesis/hst/all_stamps/')
        self.fnames      = []
        
        # Data stored in a dictionnary with file names as keys
        self.data        = {}
        
        padx             = 10
        pady             = 10
        
        self.loadButton  = ttk.Button(self.parent, command=self.openFile, text='Browse')
        
        # Making the cmap list widget
        cmapNames        = list(matplotlib.cm.cmap_d.keys())
        cmapNames.sort()
        
        self.cmap        = container()
        self.cmap.var    = tk.StringVar(value='bwr')
        self.cmap.list   = ttk.Combobox(self.parent, textvariable=self.cmap.var, values=cmapNames, state='readonly')
        self.cmap.list.bind("<<ComboboxSelected>>", self.updateCmap)
        
        self.cmap.label  = tk.Label(self.parent, text='Colormap', bg=self.bgColor)
        
        # Making the position and value label when moving through the graphs
        self.hover       = container()
        self.hover.frame = tk.Frame(self.parent, bg=self.bgColor)
        self.hover.var   = tk.StringVar(value='Current image:')
        self.hover.xvar  = tk.StringVar(value='x:')
        self.hover.yvar  = tk.StringVar(value='y:')
        
        self.hover.label = tk.Label(self.hover.frame, textvariable=self.hover.var,  bg=self.bgColor)
        self.hover.xpos  = tk.Label(self.hover.frame, textvariable=self.hover.xvar, bg=self.bgColor)
        self.hover.ypos  = tk.Label(self.hover.frame, textvariable=self.hover.yvar, bg=self.bgColor)
        
        # Adding checkboxes to invert axes
        self.invert      = container()
        self.invert.x    = tk.Checkbutton(self.parent, text='Invert x', bg=self.bgColor, command=self.invertxAxes, state=tk.DISABLED)
        self.invert.y    = tk.Checkbutton(self.parent, text='Invert y', bg=self.bgColor, command=self.invertyAxes, state=tk.DISABLED)
        
        
        ######################################################
        #               File window properties               #
        ######################################################
        self.fWindow        = container()
        self.fWindow.window = fileWindow(parent, root, bgColor=self.bgColor)
        
        
        ###################################################
        #                    Zoom widget                  #
        ###################################################
        
        # Variables
        self.zoom         = container()
        self.zoom.name    = ''
        self.zoom.xpos    = None
        self.zoom.ypos    = None
        self.zoom.im      = None
        self.zoom.xline   = None
        self.zoom.yline   = None
        self.zoom.list    = [2, 3, 4, 6, 8, 10, 20]
        self.zoom.zoom    = self.zoom.list[-2]
        self.zoom.title   = tk.StringVar(value='Zoom (x%i)' %self.zoom.zoom)
        
        # Frame, figure, axis and canvas
        self.zoom.frame   = tk.Frame(self.parent, bg=self.bgColor)
        self.zoom.axFrame = tk.Frame(self.zoom.frame, bg=self.bgColor, bd=2)
        self.zoom.fig     = Figure(figsize=(1.2, 1.2), facecolor=self.bgColor)
        self.zoom.ax      = Axes(self.zoom.fig, [0., 0., 1., 1.])
        self.zoom.ax.set_axis_off()
        self.zoom.fig.add_axes(self.zoom.ax)
        self.zoom.canvas  = FigureCanvasTkAgg(self.zoom.fig, master=self.zoom.axFrame)
        
        self.zoom.buttonP = tk.Button(self.zoom.frame, command=lambda step=1: self.changeZoom(step), text='+', activebackground='black', activeforeground='white', 
                                      bg=self.bgColor, relief=tk.FLAT, bd=0, highlightthickness=0, font=(FONT, '9', 'bold'))
        self.zoom.buttonM = tk.Button(self.zoom.frame, command=lambda step=-1: self.changeZoom(step), text='-', activebackground='black', activeforeground='white', 
                                      bg=self.bgColor, relief=tk.FLAT, bd=0, highlightthickness=0, font=(FONT, '9', 'bold'))
        self.zoom.text    = tk.Label(self.zoom.frame, textvariable=self.zoom.title, bg=self.bgColor, font=(FONT, '9', 'bold'))
        
        # Drawing elements
        self.loadButton.grid(  row=0, column=0, padx=padx,   pady=pady, sticky=tk.W+tk.N)
        
        self.cmap.label.grid(  row=0, column=3, padx=2*padx, pady=pady, sticky=tk.W+tk.N)
        self.cmap.list.grid(   row=0, column=4, padx=0,      pady=pady, sticky=tk.W+tk.N)
        
        self.hover.label.grid( row=0, column=0, sticky=tk.W+tk.N)
        self.hover.xpos.grid(  row=1, column=0, sticky=tk.W+tk.N)
        self.hover.ypos.grid(  row=2, column=0, sticky=tk.W+tk.N)
        self.hover.frame.grid( row=1, column=0, padx=padx,   pady=pady, sticky=tk.W+tk.N+tk.E, columnspan=7)
        
        self.invert.x.grid(    row=0, column=5, padx=2*padx, pady=pady, sticky=tk.W+tk.N)
        self.invert.y.grid(    row=0, column=6, padx=0,      pady=pady, sticky=tk.W+tk.N)
        
        self.zoom.frame.grid(  row=0, column=7, sticky=tk.E, padx=2*padx, pady=pady, rowspan=4)
        self.zoom.axFrame.grid(row=1, column=0, columnspan=2)
        self.zoom.canvas._tkcanvas.pack(fill=tk.BOTH)
        self.zoom.canvas.draw()
        
        self.zoom.buttonP.grid(row=2, column=1, sticky=tk.W+tk.N, padx=10, pady=2)
        self.zoom.buttonM.grid(row=2, column=0, sticky=tk.E+tk.N, padx=10, pady=2)
        self.zoom.text.grid(   row=0, column=0, sticky=tk.N,     columnspan=2)
        self.zoom.frame.grid_rowconfigure(0, weight=0)
        self.zoom.frame.grid_rowconfigure(1, weight=1)
        self.zoom.frame.grid_rowconfigure(2, weight=0)
        
        self.parent.grid_rowconfigure(1, weight=1)
        self.parent.grid_columnconfigure(7, weight=1)
        
        # Change border color to black
        self.zoom.axFrame.config({'bg':'black'})
        
        
    def changeZoom(self, step=1):
        '''
        Change the value of the zoom for the magnifier.

        Optional inputs
        ---------------.
            step : int
                How much to move the zoom value in the zoom list.
        '''
        
        newPos         = self.zoom.list.index(self.zoom.zoom)+step
        ll             = len(self.zoom.list)
        
        if newPos >= ll:
            newPos     = ll-1
        elif newPos < 0:
            newPos     = 0
            
        # If w reach the ends of the list we gray out the corresponding button
        if newPos == 0:
            self.zoom.buttonM['state'] = 'disabled'
        elif newPos == ll-1:
            self.zoom.buttonP['state'] = 'disabled'
        else:
            for butt in [self.zoom.buttonM, self.zoom.buttonP]:
                if butt['state'] == 'disabled':
                    butt['state'] = 'normal'
            
        self.zoom.zoom = self.zoom.list[newPos]
        self.zoom.title.set('Zoom (x%i)' %self.zoom.zoom)
        return


    def genFigures(self):
        '''Generate figures which have been loaded'''
        
        # First we generate the frames
        titles = [i.split('/')[-1] for i in self.data.keys()]
        self.root.bottomPane.makeFrames(nb=len(self.data), titles=titles, names=[i.rstrip('.fits') for i in titles])
        
        # Second we place them in our widget
        self.root.bottomPane.placeFrames()
            
        # Then we update each frame with the corresponding data
        for name, galaxy in self.data.items():
            self.root.bottomPane.linkNameToPlot[name.split('/')[-1].rstrip('.fits')] = self.root.bottomPane.plotList[galaxy['loc']]
            self.root.bottomPane.plotList[galaxy['loc']].updateImage(galaxy['image'], maxi=galaxy['max'], mini=galaxy['min'], cmap=self.cmap.var.get())
        return
    

    def invertxAxes(self):
        '''Invert the x axis of the selected graphs'''
        
        for plot in self.root.bottomPane.plotList:
            plot.invertAxis(axis='x')
        return
            
             
    def invertyAxes(self):
        for plot in self.root.bottomPane.plotList:
            plot.invertAxis(axis='y')
        return
    
    
    def loadFitsFile(self, file, num):
        '''
        Loads a .fits file and data into self.data dict.
        
        Mandatory inputs
        ----------------
            file : str
                file name to open
            num : int
                position number of the plot (counted from left to right, top to bottom)
        '''
        
        try:
            hdul            = fits.open(file)
            self.data[file] = {'loc':num, 'image':hdul[0].data, 'min':hdul[0].data.min(), 'max':hdul[0].data.max()}
        except IOError:
            messagebox.showerror('Invalid input', 'The given file name %s is not a FITS file' %self.fname.get())
        return
    
        
    def openFile(self, *args):
        '''Open a file using self.fname.get() value.'''
        
        self.fnames = list(askopenfilenames(initialdir=self.initDir.get().rsplit('/', 1)[0], title='Select file(s)...', filetypes=(('Fits files', '.fits'), )))
        
        if self.fnames not in ['', []]:
            
            # Empty data dict first
            self.data = {}
            
            for pos, name in enumerate(self.fnames):
                # Load data
                self.loadFitsFile(name, pos)
            
            # Reset frames before loading new figures
            self.root.bottomPane.resetFrames()
            
            # Generate frames and figures
            self.genFigures()
        
            # Set flag to True if data was succesfully loaded
            self.imLoaded = True
            
            # Generate the window manager dialog
            self.fWindow.window = fileWindow(self, self.root, bgColor=self.bgColor, dataLoaded=self.data)
            
        return

        
    def updateCmap(self, event):
        '''Update the cmap of the already plotted images.'''
        
        for plot in self.root.bottomPane.plotList:
            plot.im.set_cmap(self.cmap.var.get())
            plot.canvas.draw()
            
        # Update zoom widget as well
        self.zoom.im.set_cmap(self.cmap.var.get())
        self.zoom.canvas.draw()
        return
    
    
    def updateInvertWidgets(self, state):
        '''Change the invert widgets into some state (normal, disabled, etc.)'''
        
        if self.invert.x['state'] != state:
            self.invert.x.config({'state':state})
        
        if self.invert.y['state'] != state:
            self.invert.y.config({'state':state})
        return
    
    
    def updateMagnifier(self, data, shape=(0, 0), xpos=None, ypos=None, cmap='bwr', invertAxes={'x':False, 'y':False}):
        '''
        Update the mangifier in the top right corner.
        
        Mandatory inputs
        ----------------
            data : 2D numpy array
                full array used to update the magnifier
        
        Optional inputs
        ---------------
            xpos : int
                x-axis mouse position (in the array). Default is None, so that the image centre is used.
            ypos : int
                y-axis mouse position (in the array). Default is None, so that the image centre is used.
        '''
        
        if xpos is not None and ypos is not None:
            if (xpos != self.zoom.xpos or ypos != self.zoom.ypos):
                xpos           = int(xpos)
                ypos           = int(ypos)
                
                # Derive new shape
                dimX, dimY     = np.shape(data)
                newDimX        = dimX//self.zoom.zoom
                newDimY        = dimY//self.zoom.zoom
                
                # We make the new shape odd so that we have the center (xpos, ypos) fall into a pixel
                if newDimX%2 == 0:
                    newDimX   += 1
                if newDimY%2 == 0:
                    newDimY   += 1
                    
                xLen           = (newDimX-1)//2
                yLen           = (newDimY-1)//2
                
                # We do not have to update anything if we are out of x-axis AND y-axis bounds
                if (xpos >= xLen and xpos < shape[0]) or (ypos >= yLen or ypos < shape[1]):
                    
                    # We define the coordinates of the cursor in the zoomed array (by default centred)
                    xNewPos       = xLen
                    yNewPos       = yLen
                    
                    # We set either x or y pos if we are getting near enough to the x or y edges
                    if xpos < xLen:
                        xNewPos   += xpos-xLen
                        xpos       = xLen
                    elif xpos >= shape[0]-xLen-1:
                        xNewPos   += xpos-(shape[0]-1-xLen)
                        xpos       = shape[0]-xLen-1
                        
                    if ypos < yLen:
                        yNewPos   += ypos-yLen
                        ypos       = yLen
                    elif ypos >= shape[1]-yLen-1:
                        yNewPos   += ypos-(shape[1]-1-yLen)
                        ypos       = shape[1]-yLen-1
                        
                    xmin           = xpos-xLen
                    xmax           = xpos+xLen
                    ymin           = ypos-yLen
                    ymax           = ypos+yLen
                    
                    # Update centre position
                    self.zoom.xpos = xpos
                    self.zoom.ypos = ypos
                    newData        = data[ymin:ymax+1, xmin:xmax+1]
                    norm           = DivergingNorm(vcenter=0, vmin=np.nanmin(data), vmax=np.nanmax(data))
                    
                    self.zoom.ax.clear()
                    self.zoom.ax.axes.set_axis_off()
                    #self.zoom.ax.set_title(self.zoom.title)
                    self.zoom.xline = self.zoom.ax.plot([-1, newDimX], [yNewPos, yNewPos], 'k--', linewidth=1)
                    self.zoom.yline = self.zoom.ax.plot([xNewPos, xNewPos], [-1, newDimY], 'k--', linewidth=1)
                    self.zoom.im    = self.zoom.ax.imshow(newData, cmap=self.cmap.var.get(), norm=norm, origin='lower')
                    
                    if invertAxes['x']:
                        self.zoom.ax.set_xlim(self.zoom.ax.get_xlim()[::-1])
                    if invertAxes['y']:
                        self.zoom.ax.set_ylim(self.zoom.ax.get_ylim()[::-1])
                    self.zoom.canvas.draw()
        return
    


class fileWindow:
    ''''Window with all the info concerning the opened files'''

    # Keeping track of the only instance allowed, _isLoaded 
    _instance      = None

    def __new__(cls, parent, root, bgColor='beige', dataLoaded=False):
        '''We want to init the window only when data is loaded for the first time, then only the widgets within will be updated if necessary so we use a custom __new__ method'''
        
        if cls._instance is None:
            instance       = super(fileWindow, cls).__new__(cls)
            cls._instance  = instance
            
            # First instance, window is not drawn yet
            cls._isDrawn   = False
            cls._isVisible = tk.BooleanVar()
            cls._isVisible.set(False)
        else:
            instance       = cls._instance
        return instance
        
    def __init__(self, parent, root, bgColor='beige', dataLoaded=None):
        
        # If we have the first instance and if there is data to load
        if not fileWindow._isDrawn and dataLoaded is not None:
            self.parent   = parent
            self.root     = root
            
            self.bgColor  = bgColor
            self.window   = tk.Toplevel(bg=self.bgColor)
            self.window.wm_attributes('-type', 'utility')
            self.window.title('File(s)')
            self.window.protocol("WM_DELETE_WINDOW", self.destroy)
            self.window.resizable(width=True, height=False)
            
            # Configure escape key
            self.window.bind('<Escape>',    lambda forceUnselect=True, forceSelect=False: self.selectAll(forceUnselect, forceSelect))
            self.window.bind('<Control-a>', lambda forceUnselect=False, forceSelect=True: self.selectAll(forceUnselect, forceSelect))
            
            # Create style
            self.style    = ttk.Style()
            self.style.configure("mystyle.Treeview", highlightthickness=10, background=self.bgColor, font=(FONT, 9)) # Modify the font of the body
            self.style.configure("mystyle.Treeview.Heading", font=(FONT, 9, 'bold')) # Modify the font of the headings
            
            # Dictionnary of items (identifiers) and dictionnary of singlePlotFrames
            self.itemDict = {}
            self.plotDict = {}
            
            # Generate treeview
            self.createTreeview()
            
            # Fill treeview
            for name, data in dataLoaded.items():
                self.insertData(name.split('/')[-1].rstrip('.fits'), values=('%dx%d' %data['image'].shape, ''))
            
            self.__class__._isDrawn   = True
            self.__class__._isVisible.set(True)
            
        # If window is already drawn but new data is provided, we update widgets only
        elif fileWindow._isDrawn and dataLoaded is not None:
            self.empty()
            self.createTreeview()
            
            # Fill treeview
            for name, data in dataLoaded.items():
                self.insertData(name.split('/')[-1].rstrip('.fits'), values=('%dx%d' %data['image'].shape, ''))
        
    
    def createTreeview(self):
        '''Create the treeview widgets (generally called everytime new data is loaded)'''
        
        # Create treeview
        self.columns  = ('Dimensions', 'PA')
        self.treeview = ttk.Treeview(self.window, columns=self.columns, style='mystyle.Treeview', height=0)
        self.treeview.heading('#0', text='Name',            anchor=tk.W, command=lambda forceUnselect=False, forceSelect=False: self.selectAll(forceUnselect, forceSelect))
        self.treeview.heading('#1', text='Dimensions (px)', anchor=tk.W, command=lambda forceUnselect=False, forceSelect=False: self.selectAll(forceUnselect, forceSelect))
        self.treeview.heading('#2', text='PA (°)',          anchor=tk.W, command=lambda forceUnselect=False, forceSelect=False: self.selectAll(forceUnselect, forceSelect))
        self.treeview.tag_configure('all', background=self.bgColor)
        self.treeview.column('#0', anchor=tk.W)
        for i in range(len(self.columns)):
            self.treeview.column('#%d' %(i+1), anchor=tk.CENTER)
            
        # Bind event(s)
        self.treeview.bind('<<TreeviewSelect>>', self.onClick)
        
        # Pack things up
        self.treeview.pack(fill='x', side=tk.TOP, expand=True)
        
        return
        
    
    def destroy(self):
        '''Just hide the window instead of really destroying it'''
        
        self.setState(state='withdrawn')
        return
        
    
    def empty(self):
        '''Empty the window of any widgets and reset lists'''
        
        self.treeview.destroy()
        #for item in self.itemDict.values():
            #self.treeview.delete(item)
        self.itemDict = {}
        self.plotDict = {}
        return
    
    
    def insertData(self, text, values, pos='end'):
        '''
        Insert a new line into the treeview

        Mandatory inputs
        ----------------
            text : str
                Main text associated to the line
            values : tuple
                values to be printed in the treeview
        
        Optional inputs
        ---------------
            pos : int or str
                where to place the new line in the treeview list. 'end' to place it at the end, otherwise a number
        '''
        
        self.itemDict[text]                = self.treeview.insert("", pos, text=text, values=values, tags=('all'))
        self.plotDict[self.itemDict[text]] = self.root.bottomPane.linkNameToPlot[text]
        self.treeview.configure(height=self.treeview['height']+1)
        return
    
    
    def onClick(self, event):
        ''''Actions taken when clicking on a line.'''
        
        for iid in self.itemDict.values():
            if iid in self.treeview.selection():
                if not self.plotDict[iid].selected:
                    self.plotDict[iid].borderOnOff(on=True)
                    self.root.topPane.updateInvertWidgets(state='normal')
            else:
                if self.plotDict[iid].selected:
                    self.plotDict[iid].borderOnOff(on=False)
                    self.root.topPane.updateInvertWidgets(state='disabled')
        return
        
    
    def selectAll(self, forceUnselect=False, forceSelect=False):
        '''
        Either select or unselect all the lines.
        
        Optional inputs
        ---------------
            forceUnselect : bool
                whether to force it to unselect all the lines or not. Default is to not force anything.
            forceSelect : bool
                whether to force it to select all the lines or not. Default is to not force anything.
        '''
        
        # First try to force the (un)selection if one of the keywords is provided, otherwise do according to the number of selected items
        if forceSelect:
            self.treeview.selection_set(list(self.itemDict.values()))
        elif forceUnselect:
            self.treeview.selection_remove(list(self.itemDict.values()))
        else:                
            if len(self.treeview.selection()) != len(self.itemDict):
                self.treeview.selection_set(list(self.itemDict.values()))
            else:
                self.treeview.selection_remove(list(self.itemDict.values()))
            
        return
    
    
    def selectLine(self, name, select=True):
        '''
        Select or unselect a line given the keyword value.
        
        Mandatory parameters
        --------------------
            name : str
                Name of the desired item as listed in the self.itemDict dictionnary
        
        Optional parameters
        -------------------
            select : bool
                whether to select the line (True) or unselect it (False)
        '''
        
        if select:
            if self.itemDict[name] not in self.treeview.selection():
                self.treeview.selection_add(self.itemDict[name])
        else:
            if self.itemDict[name] in self.treeview.selection():
                self.treeview.selection_remove(self.itemDict[name])
        return
    
    
    def setState(self, state='normal'):
        '''
        Set the sate of the window.

        Optional parameters
        -------------------
            state : 'normal' or 'withdrawn'
                change the window state to the given value if its state is different
        '''
        
        state = state.lower()
        if state not in ['normal', 'withdrawn']:
            raise ValueError("Either provide 'normal' or 'withdrawn' as a state for the file manager dialog. Cheers !")
            
        if self.window.state() != state:
            if state == 'normal':
                self.__class__._isVisible.set(True)
                self.window.deiconify()
            elif state == 'withdrawn':
                self.__class__._isVisible.set(False)
                self.window.state(state)
        return
        
    
    
    def switchWindowState(self):
        '''Enable back the window if its state is such that it has been hidden, or the oposite if it is normal'''
        
        if self.window.state() != 'normal':
            self.setState(state='normal')
        else:
            self.setState(state='withdrawn')
        return
        
                
                
class modelFrame:
    '''Frame-like object drawn in canvas which holds widgets relative to configuring models'''
    
    def __init__(self, parent, canvas, root, num, posx=0, posy=0, bgColor='grey', padx=4, pady=4, width=100, height=100):
        global DICT_MODELS
        
        self.parent              = parent
        self.canvas              = canvas
        self.root                = root
        self.bgColor             = bgColor
        
        self.num                 = num
        
        self.width               = width-posx
        self.height              = height
        
        self.posx                = [posx, self.width]
        self.posy                = [posy, posy+self.height]
        
        self.padx                = padx
        self.pady                = pady
        
        # Container objects
        self.frame, self.modelLabel, self.modelList     = container(), container(), container()
        self.topLine, self.leftLine, self.plusMinButton = container(), container(), container()
        self.circleBitmap                               = container()

        # Draw an extensible line on top of each model sub part
        self.topLine.pad         = 3*self.padx
        self.topLine.id          = self.canvas.create_line(self.posx[0]+self.topLine.pad, self.posy[0], self.posx[1]-self.topLine.pad, self.posy[0])
        
        self.modelLabel.obj      = tk.Label(self.canvas, text='Model:', bg=self.bgColor)
        self.modelLabel.id       = self.canvas.create_window(self.padx+self.posx[0], self.pady+self.posy[0], window=self.modelLabel.obj, anchor='nw')
        
        # Making a combobox to select the model
        self.modelName           = tk.StringVar(value='Sersic')
        
        self.modelList.obj       = ttk.Combobox(self.canvas, textvariable=self.modelName, values=list(DICT_MODELS.values()), state='readonly')
        self.modelList.id        = self.canvas.create_window(self.padx+self.canvas.bbox(self.modelLabel.id)[2], self.pady+self.posy[0], window=self.modelList.obj, anchor='nw')
        self.modelList.bbox      = list(self.canvas.bbox(self.modelList.id))
        self.modelList.bbox[2]   = self.posx[1]-self.padx
        self.modelList.width     = 1 + ((self.modelList.bbox[2])-91)//7 # Why this equation ? no one knows... but it works    
        print(self.modelList.bbox, self.posx, width)
        self.modelList.obj.configure(width=self.modelList.width)        
        
        # Make widgets relative to Sersic model
        self.CanvasWidgets       = self.SersicWidgets(self.posx[0]+self.padx, self.modelList.bbox[3]+self.pady)
        
        # Update maximum y pos in bbox
        self.computeMaxY()
        
        # Draw a line on the left side of each model sub part
        self.leftLine.pad        = 2*self.pady-10
        self.leftLine.id         = self.canvas.create_line(self.posx[0], self.posy[0]+self.leftLine.pad, self.posx[0], self.posy[1], dash=3)
        
        # Add a button on top of the left line with a bitmap
        self.circleBitmap.normal = tk.BitmapImage(data=CIRCLE)
        self.circleBitmap.inv    = tk.BitmapImage(data=CIRCLE_INV)
        
        self.plusMinButton.obj   = tk.Button(self.canvas, image=self.circleBitmap.normal, text='-', compound='center', bd=0, highlightthickness=0, width=1, 
                                             bg=self.bgColor, activebackground=self.bgColor, activeforeground='white')
        self.plusMinButton.id    = self.canvas.create_window(self.posx[0]-12, self.posy[0]-10, window=self.plusMinButton.obj, anchor='nw')
        
        #Binding events to this object
        self.plusMinButton.obj.bind('<Enter>',    self.invertCircleBitmap)
        self.plusMinButton.obj.bind('<Leave>',    self.invertCircleBitmap)
        self.plusMinButton.obj.bind('<Button-1>', self.hideWidgets)
        
        # Complete the list of widgets handled by the canvas
        self.CanvasWidgets += [self.topLine, self.modelLabel, self.modelList, self.leftLine, self.plusMinButton]
        
        self.canvas.update_idletasks()
       
    
    def computeMaxY(self):
        '''Update the pos y value according to widgets' pos '''
        
        for widget in self.model.widgetList:
            if self.posy[1] != self.canvas.bbox(widget.id)[3]:
                self.posy[1] = self.canvas.bbox(widget.id)[1] + widget.obj['height']
        
        
    def updateDimension(self, newWidth=None, newHeight=None):
        '''Update the dimensions of the rectangle around'''
        
        # Update width, height and pos x and y of main rectangle
        if newWidth is not None:
            self.width               = newWidth
            self.posx[1]             = self.posx[0] + self.width
        
        if newHeight is not None:
            self.height              = newHeight           
            self.posy[1]             = self.posy[0] + self.height
            
        # Update top line coordinates
        self.canvas.coords(self.topLine.id, (self.posx[0]+self.topLine.pad, self.posy[0], self.posx[1]-self.topLine.pad, self.posy[0]))
        
        # Update combobox width
        self.modelList.bbox      = self.modelList.bbox[0:2] + [self.posx[1]-self.padx, self.modelList.bbox[3]]
        self.modelList.width     = 1 + ((self.modelList.bbox[2])-91)//7
        if self.modelList.width < 0:
            self.modelList.width = 0
            self.canvas.itemconfigure(self.modelList.id, state='hidden')
        else:
            if self.canvas.itemcget(self.modelList.id, 'state') == 'hidden':
                self.canvas.itemconfigure(self.modelList.id, state='normal')
            self.modelList.obj.configure(width=self.modelList.width)
            
            
    def invertCircleBitmap(self, event):
        '''Invert the plus-minus colors when getting/losing mouse focus'''
        
        if event.type == tk.EventType.Enter:
            self.plusMinButton.obj['image'] = self.circleBitmap.inv
        elif event.type == tk.EventType.Leave:
            self.plusMinButton.obj['image'] = self.circleBitmap.normal
            
    
    def hideWidgets(self, event):
        '''Hide the frame containing the model dependent widgets'''

        if self.plusMinButton.obj['text'] == '-':
            for widget in self.model.widgetList:
                self.canvas.itemconfigure(widget.id, state='hidden')
            self.plusMinButton.obj['text'] = '+'
            
            offset       = self.modelList.bbox[3]-self.posy[1]
            
            self.posy[1] = self.modelList.bbox[3]
            self.canvas.coords(self.leftLine.id, list(self.canvas.coords(self.leftLine.id)[0:3]) + [self.posy[1]])
            
            # Update position of all widgets below
            self.parent.verticalOffset(nb=self.num, where='below', offset=offset)
            
            
        elif self.plusMinButton.obj['text'] == '+':
            for widget in self.model.widgetList:
                self.canvas.itemconfigure(widget.id, state='normal')
            self.plusMinButton.obj['text'] = '-'
            
            offset       = self.model.lframe.obj['height'] + self.canvas.bbox(self.model.lframe.id)[1] - self.posy[1]
            
            self.posy[1] = self.model.lframe.obj['height'] + self.canvas.bbox(self.model.lframe.id)[1]
            self.canvas.coords(self.leftLine.id, list(self.canvas.coords(self.leftLine.id)[0:3]) + [self.posy[1]])
            
            # Update position of all widgets below
            self.parent.verticalOffset(nb=self.num, where='below', offset=offset)
            
            
    def SersicWidgets(self, xpos, ypos, pady=5, padx=5):
        '''Draw widgets when selecting a model profile'''
        
        self.model, self.model.lframe = container(), container()
        self.model.x, self.model.y    = container(), container()
        
        # Label frame around center widgets
        self.model.lframe.obj = tk.LabelFrame(self.canvas, text='Center position', bg=self.bgColor, padx=padx, pady=pady, bd=1, relief=tk.RIDGE)
        self.model.lframe.id  = self.canvas.create_window(xpos, ypos, window=self.model.lframe.obj, anchor='nw')
        
        # x pos widgets
        self.model.x.var      = tk.DoubleVar()
        self.model.x.var.set(70)
        self.model.x.label    = tk.Label(self.model.lframe.obj, text='x:', bg=self.bgColor)
        self.model.x.entry    = tk.Entry(self.model.lframe.obj, textvariable=self.model.x.var)
        self.model.x.padx     = padx
        
        # y pos widgets
        self.model.y.var      = tk.DoubleVar()
        self.model.y.var.set(70)
        self.model.y.label    = tk.Label(self.model.lframe.obj, text='y:', bg=self.bgColor)
        self.model.y.entry    = tk.Entry(self.model.lframe.obj, textvariable=self.model.y.var)
        self.model.y.padx     = padx
        
        self.model.x.label.grid(row=0, column=0, sticky=tk.W+tk.N)
        self.model.x.entry.grid(row=0, column=1, sticky=tk.W+tk.N, padx=self.model.x.padx)
        self.model.y.label.grid(row=1, column=0, sticky=tk.W+tk.N)
        self.model.y.entry.grid(row=1, column=1, sticky=tk.W+tk.N, padx=self.model.y.padx)
        
        #list of model widgets
        self.model.widgetList = [self.model.lframe]
        
        #Update label frame width and height
        self.model.lframe.obj['width']  = 2*padx + self.model.x.padx + self.model.x.label['width'] + self.model.x.entry['width']
        self.model.lframe.obj['height'] = 2*pady + 2*20 + pady #Temporary solution till I find a way to get the label frame true size on cavas
        
        return copy.copy((self.model.widgetList))
    
    
    def moveWidgetsVert(self, offset, direction='vertical'):
        '''Move all the widget either up/down or left/right by some offset value'''
        
        # Update vertical coordinates
        if direction == 'vertical':
            self.posy               = [i+offset for i in self.posy]
            self.modelList.bbox[1] += offset
            self.modelList.bbox[3] += offset
        elif direction == 'horizontal':
            self.posx = [i+offset for i in self.posx]
            self.modelList.bbox[0] += offset
            self.modelList.bbox[2] += offset
            
        for widget in self.CanvasWidgets:
            coords = list(self.canvas.coords(widget.id))
            if direction == 'vertical':
                coords[1]     += offset
                if len(coords) == 4:
                    coords[3] += offset
                self.canvas.coords(widget.id, coords)
                
            elif direction == 'horizontal':
                coords[0]     += offset
                if len(coords) == 4:
                    coords[2] += offset
                self.canvas.coords(widget.id, coords)
        
                          
        
class rightFrame:
    '''Right frame window with different options to trigger.'''
    
    def __init__(self, parent, root, bgColor='grey'):
        '''
        Inputs
        ------
            parent : tk object
                parent object to put the widget within
            root : tk.Tk instance
                main application object
        '''
        
        self.parent       = parent
        self.root         = root
        
        # Number of models used and drawn on screen
        self.nbModels     = 0
        self.modelsFrames = []
        self.scrollFrac   = 0.0
        
        self.bgColor      = bgColor
        self.bd           = 4
        self.padx         = 5
        self.pady         = 5
        self.padyModels   = 10
        
        # Container objects
        self.frame, self.addModel = container(), container()
        
        # Define label frame
        self.labelFrame   = tk.LabelFrame(self.parent, text='Configuration pane', bg=self.root.topFrame.color, relief=tk.RIDGE, bd=self.bd)
        
        # Define canvas within label frame to add a scrollbar
        self.canvas       = tk.Canvas(self.labelFrame, bd=0, bg=self.bgColor)
        self.scrollbar    = tk.Scrollbar(self.labelFrame, orient="vertical", command=self.canvas.yview, width=5, bg='black')
        
        # Define a frame within the canvas to hold widgets
        self.frame.obj    = tk.Frame(self.canvas, bg=self.bgColor)
        self.frame.id     = self.canvas.create_window(0, 0, anchor='nw', window=self.frame.obj)
        
        # Button used to add a new model
        self.addModel.obj = tk.Button(self.canvas, text='+ add new model', relief=tk.FLAT, bg=self.bgColor, bd=0, highlightthickness=0, 
                                      activebackground='black', activeforeground=self.bgColor, command=self.addNewModel, font=(FONT, '9', 'bold'))
        self.addModel.id  = self.canvas.create_window(4, 4, anchor='nw', window=self.addModel.obj)
            
        self.canvas.update_idletasks()
        
        # Configure scrollbar on the right
        self.canvas.configure(scrollregion=self.canvas.bbox('all'), yscrollcommand=self.scrollbar.set)
        
        # Bind resize event to updating the frame size
        self.canvas.bind('<Configure>', self.updateFrameSize) 
  
        # Draw widgets
        self.scrollbar.pack( fill='both', side='right')
        self.canvas.pack(    fill='both', expand='yes')
        self.labelFrame.pack(fill='both', expand='yes', padx=self.padx, pady=self.pady)
       
        
    def updateFrameSize(self, event):
        '''Update the frame size of every model frame'''
        
        # Update main window frame element first
        self.canvas.itemconfig(self.frame.id, width = event.width) 
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        
        self.width = event.width
        
        # Update each model frame
        for mframe in self.modelsFrames:
            mframe.updateDimension(newWidth=event.width-2*mframe.posx[0])
            
        
        
        self.canvas.update_idletasks()
        
    
    def addNewModel(self):
        '''Add a new model frame'''
        
        #Set initial y position and then place model frame objects below the latter one
        posy = self.padyModels + self.canvas.bbox('all')[3]
        
        if self.nbModels == 0:
            self.width = self.canvas.bbox('all')[2] - self.canvas.bbox('all')[0]
        
        self.nbModels += 1
        self.modelsFrames.append(modelFrame(self, self.canvas, self.root, num=self.nbModels, posx=10, posy=posy, width=self.width, height=100,
                                            pady=self.padyModels, padx=10, bgColor=self.bgColor))   
        
        # Always update canvas scrollregions otherwise it does weird stuff
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        
        
    def verticalOffset(self, nb=1, where='below', offset=0):
        '''Apply a vertical offset to all the widgets in model frame either above or below some widget'''
        
        if   where == 'below':
            mframes = self.modelsFrames[nb:]
        elif where == 'above':
            mframes = self.modelsFrames[0:nb]
            
        for mframe in mframes:
            mframe.moveWidgetsVert(offset)


        
class topMenu:
    '''Application top menu'''
    
    def __init__(self, parent, root, color):
        
        self.parent   = parent
        self.root     = root
        self.color    = color
        self.exists   = False
        
        # Top Menu widget
        self.topMenu  = tk.Menu(bg=self.color)
        
        # File menu within top menu
        self.fileMenu = tk.Menu(self.topMenu, tearoff=0)
        self.fileMenu.add_command(label='Open (Ctrl+O)',      command=self.parent.topPane.openFile)
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label='Close (Alt+F4)',     command=self.parent.exitProgram)
        
        # View menu within top menu
        self.viewMenu = tk.Menu(self.topMenu, tearoff=0)
        self.viewMenu.add_checkbutton(label=' File manager', variable=fileWindow._isVisible, command=self.parent.topPane.fWindow.window.switchWindowState)
        
        # Help menu within top menu
        self.helpMenu = tk.Menu(self.topMenu, tearoff=0)
        self.helpMenu.add_command(label='Shortcuts (Ctrl+H)', command=self.showShortcuts)
        self.helpMenu.add_command(label='Galfit website',     command=lambda:print('work in progress'))
        
        # Adding sections into top menu
        self.topMenu.add_cascade( label="File", menu=self.fileMenu)
        self.topMenu.add_cascade( label="View", menu=self.viewMenu)
        self.topMenu.add_cascade( label="Help", menu=self.helpMenu)
       
        
    def exitProgram(self):
        '''Exit top level window.'''
        
        self.window.destroy()
        self.exists    = False
        
        
    def makeLabelLine(self, tab, row, text, listButtonLabels, underline=-1):
        '''Used to create a line of labels within the help window'''
        
        labels = [tk.Label(tab, text=text, underline=underline, anchor='w', justify='left', width=25)]
        for i in listButtonLabels[:-1]:
            labels.append(tk.Label(tab, text=i, bg=self.keyColor, bd=self.bd, relief=self.keyRelief, padx=self.pad, pady=self.pad))
            labels.append(tk.Label(tab, text='+'))
        labels.append(tk.Label(tab, text=listButtonLabels[-1], bg=self.keyColor, bd=self.bd, relief=self.keyRelief, padx=self.pad, pady=self.pad))
        
        for pos, i in enumerate(labels):
            if pos==0:
                padx, pady = 10, 10
                sticky     = tk.W
            else:
                padx, pady = 0, 0
                sticky     = ''
                
            i.grid(row=row, column=pos, padx=padx, pady=pady, sticky=sticky)
        return labels
            
        
        
    def showShortcuts(self, *args):
        if not self.exists:
            self.window        = tk.Toplevel(height=300, width=300)
            self.window.maxsize(300, 300)
            self.window.minsize(300, 300)
            self.window.title('List of shortcuts')
            self.exists        = True
            
            self.keyColor      = 'white'
            self.keyRelief     = tk.RAISED
            self.bd            = 2
            self.pad           = 3
            
            self.window.protocol("WM_DELETE_WINDOW", self.exitProgram)
            
            # Create notebook
            self.notebook      = ttk.Notebook(self.window)
            self.notebook.enable_traversal()
            self.notebook.pack(expand=1, fill='both')
            
            # Make first tab (edit tab)
            self.tabEdit       = ttk.Frame(self.notebook)
            self.notebook.add(self.tabEdit, text='Edit figure', underline=0)
            self.editLines     = {'Select all'  :         ['Ctrl', 'A'], 
                                  'Draw PA line':         ['Ctrl', 'P'],
                                  'Back to default mode': ['ESC']}
            
            for pos, text in enumerate(self.editLines.keys()):
                self.makeLabelLine(self.tabEdit, pos, text, self.editLines[text])
                
            # Make second tab (file tab)
            self.tabFile       = ttk.Frame(self.notebook)
            self.notebook.add(self.tabFile, text='File', underline=0)
            self.fileLines     = {'Open file': ['Ctrl', 'O']}
            
            for pos, text in enumerate(self.fileLines.keys()):
                self.makeLabelLine(self.tabFile, pos, text, self.fileLines[text])


class mainApplication:
    '''
    Main application where all the different layouts are.
    '''
    
    def __init__(self, parent):
        '''
        Inputs
        ------
            parent : tk.Tk instance
                root propagated throughout the different Frames and Canvas
        '''
        
        self.parent            = parent
        
        # Set default cursor and state
        self.state             = 'default'
        self.parent.config(cursor='arrow')
        
        # Set active frame
        self.activeFrame       = None
        
        # Set containers
        self.topFrame, self.rightFrame, self.bottomFrame = container(), container(), container()
        self.topMenu                                     = container()
        
        # Set colors
        self.topFrame.color    = 'lavender'
        self.rightFrame.color  = 'beige'
        self.bottomFrame.color = 'beige'
        self.topMenu.color     = 'slate gray'
        
        # Making main three frames
        self.topFrame.frame    = tk.Frame(self.parent, bg=self.topFrame.color)
        self.rightFrame.frame  = tk.Frame(self.parent, bg=self.topFrame.color)
        self.bottomFrame.frame = tk.Frame(self.parent, bg=self.bottomFrame.color, bd=2, relief=tk.GROOVE)
        
        # Creating widgets within frames
        self.topPane           = topFrame(  self.topFrame.frame,    self, bgColor=self.topFrame.color)
        self.rightPane         = rightFrame(self.rightFrame.frame,  self, bgColor=self.rightFrame.color)
        self.bottomPane        = graphFrame(self.bottomFrame.frame, self, bgColor=self.bottomFrame.color)
        
        # Creating window top menu
        self.topMenu.window    = topMenu(self, self.parent, self.topMenu.color)
        self.parent.config(menu=self.topMenu.window.topMenu)
        
        # Binding key events
        self.parent.bind('<Control-p>',  self.lineTracingState)
        self.parent.bind('<Escape>',     self.defaultState)
        self.parent.bind('<Control-a>',  self.selectAll)
        self.parent.bind('<Control-z>',  self.cancel)
        self.parent.bind('<Control-o>',  self.topPane.openFile)
        self.parent.bind('<Control-h>',  self.topMenu.window.showShortcuts)
        
        # Bind enter and leave frames to know where the cursor lies
        self.rightFrame.frame.bind('<Enter>', lambda event, frame='right': self.setScrollable(event, frame))
        self.rightFrame.frame.bind('<Leave>', lambda event, frame='right': self.unsetScrollable(event, frame))
        
        self.bottomFrame.frame.bind('<Enter>', lambda event, frame='bottom': self.setScrollable(event, frame))
        self.bottomFrame.frame.bind('<Leave>', lambda event, frame='bottom': self.unsetScrollable(event, frame))
        
        # Drawing frames
        self.topFrame.frame.grid(   row=0, sticky=tk.N+tk.S+tk.W+tk.E, columnspan=3)
        self.bottomFrame.frame.grid(row=1, sticky=tk.N+tk.S+tk.W+tk.E, columnspan=2)
        self.rightFrame.frame.grid( row=1, sticky=tk.N+tk.S+tk.W+tk.E, column=2)
        
        # Setting grid geometry for main frames
        tk.Grid.rowconfigure(   self.parent, 0, weight=0, minsize=130)
        tk.Grid.rowconfigure(   self.parent, 1, weight=1)
        tk.Grid.columnconfigure(self.parent, 2, weight=0, minsize=100)
        tk.Grid.columnconfigure(self.parent, 1, weight=1)
        
        
    def setScrollable(self, event, frame='right'):
        self.parent.bind('<MouseWheel>', lambda event, frame=frame: self.setMouseWheel(event, frame))
        self.parent.bind("<Button-4>",   lambda event, frame=frame: self.setMouseWheel(event, frame))
        self.parent.bind("<Button-5>",   lambda event, frame=frame: self.setMouseWheel(event, frame))
        return
        
        
    def unsetScrollable(self, event, frame='right'):
        self.parent.unbind('<MouseWheel>')
        self.parent.unbind("<Button-4>")
        self.parent.unbind("<Button-5>")
        return
            
        
    def setMouseWheel(self, event, frame='right'):
        if frame == 'right':
            if self.rightPane.canvas.bbox('all')[3] > self.rightPane.labelFrame.winfo_height():
                if event.num==5 or event.delta<0:
                    step = -1
                else:
                    step = 1            
                self.rightPane.canvas.yview_scroll(step, 'units')
        elif frame == 'bottom':
            if event.num==5 or event.delta<0:
                step = -1
            else:
                step = 1            
            self.bottomPane.canvas.yview_scroll(step, 'units')
        return
        
    
    def defaultState(self, event):
        '''Change the graphFrame instance back to default state (where the user can select plots)'''
        
        if self.state != 'default':
            self.bottomFrame.frame.config(cursor='arrow')
            self.state = 'default'
        return
            
    def cancel(self, event):
        '''Cancel PA line drawing if Ctrl-z is pressed'''
        
        if self.bottomPane.activeSingleFrame.paLine.line is not None:
            self.bottomPane.activeSingleFrame.paLine.drawing = False
            self.bottomPane.activeSingleFrame.paLine.line.set_data([[], []])
            self.bottomPane.activeSingleFrame.canvas.draw()
            
            # Update the PA in the file manager
            self.bottomPane.activeSingleFrame.updateFileManagerValues(valNames=['PA'], values=[''])
        
        
    def lineTracingState(self, event):
        '''Change the graphFrame instance to lineTracing state to enable the tracing of PA line.'''
        
        if self.state != 'lineTracing':
            self.bottomFrame.frame.config(cursor='crosshair')
            self.state = 'lineTracing'
            
            
    def exitProgram(self):
        '''Destroys main window.'''
        
        self.parent.destroy()
            
    
    def selectAll(self, event):
        '''Select or unselect all the plots.'''
        
        # Case 1: not all the plots are selected, so we select those that are not yet
        if self.bottomPane.nbFrames > self.bottomPane.numSelected:
            for plot in self.bottomPane.plotList:
                if not plot.selected:
                    plot.selectIt()
                    
            self.topPane.updateInvertWidgets(state='normal')
            
        # Case 2: all the plots are selected, so we unselect them all
        elif self.bottomPane.nbFrames == self.bottomPane.numSelected:
            for plot in self.bottomPane.plotList:
                if plot.selected:
                    plot.selectIt()
            self.topPane.updateInvertWidgets(state='disabled')
        
        return
            

def main(): 
    
    root = tk.Tk()
    root.title('GalBit - Easily do stuff')
    root.geometry("1500x800")
    app  = mainApplication(root)
    
    root.protocol("WM_DELETE_WINDOW", app.exitProgram)
    
    root.mainloop()

if __name__ == '__main__':
    main()