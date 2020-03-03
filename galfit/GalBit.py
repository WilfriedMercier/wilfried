#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:18:55 2019

@author: Wilfried - IRAP
"""

import tkinter            as     tk
from   tkinter            import messagebox
from   tkinter.filedialog import askopenfilename
from   tkinter            import ttk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
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
               'model'       : 'model', 
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
    def __init__(self, parent, root, data=None, title=None, bgColor='beige'):
        
        # General properties
        self.bdOn           = 'black'
        self.bdOff          = bgColor
        
        self.selected       = False
        
        self.parent         = parent
        self.root           = root
        
        # PA line properties
        self.paLine         = container()
        self.paLine.posx    = [None, None]
        self.paLine.posy    = [None, None]
        self.paLine.drawing = False
        
        # Making a figure
        self.bgColor        = bgColor
        self.figure         = Figure(figsize=(4,4), tight_layout=True, facecolor=self.bgColor)
            
        # Creating an axis
        self.ax             = self.figure.add_subplot(111)
        self.ax.yaxis.set_ticks_position('both')
        self.ax.xaxis.set_ticks_position('both')
        self.ax.tick_params(which='both', direction='in', labelsize=12)
        self.ax.grid()
        
        # Plotting empty plots in order to update them later
        self.paLine.line    = self.ax.plot([], [], 'k-')[0]
        
        # Adding a title
        if title is not None and type(title) is str:
            self.title = title
            self.ax.set_title(self.title)
            
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Linking to events
        self.canvas.mpl_connect('button_press_event',  self.onClick)
        self.canvas.mpl_connect('motion_notify_event', self.onMove)
        self.canvas.mpl_connect('figure_enter_event',  self.onFigure)
        
    
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
        
        norm    = DivergingNorm(vcenter=0, vmin=mini, vmax=maxi)
        
        if not self.root.topPane.imLoaded:
            print('tada')
            self.im = self.ax.imshow(newData, cmap=cmap, norm=norm, origin='lower')
        else:
            print('coucou')
            self.im.set_data(newData)
            self.im.set_cmap(cmap)
            self.im.set_norm(norm)
            
        self.canvas.draw()
        
        
    def changeBorder(self):
        '''Adds or remove a border around the figure canvas.'''
        
        # Case 1: the plot is selected, we change the border color back to the frame color
        if self.parent['bg'] == self.bdOn:
            self.parent.config({'bg':self.bdOff})
            self.selected = False
            self.root.bottomPane.numSelected -= 1
        # Case 2: the plot is unselected and we change the border color to the 'On' color (default is black)
        else:
            self.parent.config({'bg':self.bdOn})
            self.selected = True
            self.root.bottomPane.numSelected += 1
        
        
    def onClick(self, event):
        '''Actions taken when the user clicks on one of the figures.'''
        
        # If state is default, we draw a border (technically we change the color of the border) around a plot frame to indicate that it has been selected
        if self.root.state == 'default':
            self.changeBorder()
        else:
            if not self.paLine.drawing:
                self.paLine.posx[0] = event.xdata
                self.paLine.posy[0] = event.ydata
                self.paLine.drawing = True
            else:
                self.paLine.drawing = False
                self.paLine.line.set_data([self.paLine.posx, self.paLine.posy])
                self.canvas.draw()
                
        # if at least one graph is selected, we enable the x and y invert widgets
        if self.root.bottomPane.numSelected == 1:
            self.root.topPane.invert.x.config({'state':'normal'})
            self.root.topPane.invert.y.config({'state':'normal'})
        elif self.root.bottomPane.numSelected == 0:
            self.root.topPane.invert.x.config({'state':'disabled'})
            self.root.topPane.invert.y.config({'state':'disabled'})
                
                
    def onFigure(self, event):
        '''Set focus onto the main window when the cursor enters it.'''
        
        self.root.parent.focus()
            
            
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
        


class graphFrame:
    def __init__(self, parent, root, bgColor='beige'):
        
        ##########################################################
        #                 Setting instance variables             #
        ##########################################################
        
        self.parent            = parent
        self.root              = root
        self.bdSize            = 5
        
        self.numSelected       = 0
        
        # Because matplotlib key press event is not Tk canvas dependent (i.e it links it to the last drawn Tk canvas), we define an active singleFrame instance
        # which we use to update graphs when the focus is on them
        self.activeSingleFrame = None
        
        
        ##################################################
        #              Making single plot frame          #
        ##################################################
        
        self.leftFrame         = tk.Frame(self.parent, bd=self.bdSize, bg=bgColor)
        self.midFrame          = tk.Frame(self.parent, bd=self.bdSize, bg=bgColor)
        self.rightFrame        = tk.Frame(self.parent, bd=self.bdSize, bg=bgColor)
        
        
        #################################################################
        #                Making singlePlotFrame instances               #
        #################################################################
        
        self.plot1             = singlePlotFrame(self.leftFrame,  self.root, title='data',     bgColor=bgColor)
        self.plot2             = singlePlotFrame(self.midFrame,   self.root, title='model',    bgColor=bgColor)
        self.plot3             = singlePlotFrame(self.rightFrame, self.root, title='residual', bgColor=bgColor)
        
        self.leftFrame.grid(row=0, column=0, padx=5, pady=5,  sticky=tk.N+tk.E+tk.W)
        self.midFrame.grid(row=0, column=1, padx=5, pady=5,   sticky=tk.N+tk.E+tk.W)
        self.rightFrame.grid(row=0, column=3, padx=5, pady=5, sticky=tk.N+tk.E+tk.W)
        """
        self.leftFrame.pack( fill='x', expand=True, side=tk.LEFT, padx=5, pady=5, anchor=tk.N)
        self.midFrame.pack(  fill='x', expand=True, side=tk.LEFT, padx=5, pady=5, anchor=tk.N)
        self.rightFrame.pack(fill='x', expand=True, side=tk.LEFT, padx=5, pady=5, anchor=tk.N)
        """
        


class topFrame:
    '''
    Top frame with widgets used to import data.
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
        
        self.parent     = parent
        self.root       = root
        self.bgColor    = bgColor
        self.imLoaded   = False
        self.fname      = tk.StringVar()
        self.fname.set('/home/wilfried/Thesis/galfit/modelling/outputs/')
        
        padx            = 10
        pady            = 10
        
        self.loadButton = ttk.Button(self.parent, command=self.openFile, text='Browse')
        self.loadInput  = ttk.Entry( self.parent, cursor='xterm', textvariable=self.fname, width=50)
        
        self.sendButton = ttk.Button(self.parent, command=self.updateFigures, text='Load')
        
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
        
        self.hover.var   = tk.StringVar(value='Current image:')
        self.hover.xvar  = tk.StringVar(value='x:')
        self.hover.yvar  = tk.StringVar(value='y:')
        
        self.hover.label = tk.Label(self.parent, textvariable=self.hover.var,  bg=self.bgColor)
        self.hover.xpos  = tk.Label(self.parent, textvariable=self.hover.xvar, bg=self.bgColor)
        self.hover.ypos  = tk.Label(self.parent, textvariable=self.hover.yvar, bg=self.bgColor)
        
        # Adding checkboxes to invert axes
        self.invert      = container()
        self.invert.x    = tk.Checkbutton(self.parent, text='Invert x', bg=self.bgColor, command=self.invertxAxes, state=tk.DISABLED)
        self.invert.y    = tk.Checkbutton(self.parent, text='Invert y', bg=self.bgColor, command=self.invertyAxes, state=tk.DISABLED)
        
        # Drawing elements
        self.loadInput.grid(   row=0, column=1, padx=0,      pady=pady)
        self.loadButton.grid(  row=0, column=0, padx=padx,   pady=pady, sticky=tk.W)
        self.sendButton.grid(  row=0, column=2, padx=padx,   pady=pady)
        
        self.cmap.label.grid(  row=0, column=3, padx=2*padx, pady=pady)
        self.cmap.list.grid(   row=0, column=4, padx=0,      pady=pady)
        
        self.hover.label.grid( row=2, column=0, padx=padx,   pady=pady, sticky=tk.W, columnspan=2)
        self.hover.xpos.grid(  row=3, column=0, padx=padx,   pady=0,    sticky=tk.W, columnspan=2)
        self.hover.ypos.grid(  row=4, column=0, padx=padx,   pady=0,    sticky=tk.W, columnspan=2)
        
        self.invert.x.grid(    row=0, column=5, padx=2*padx, pady=0)
        self.invert.y.grid(    row=0, column=6, padx=0,      pady=0)
        
        
    def openFile(self, *args):
        '''Opens a file using self.fname.get() value.'''
        
        tmp = askopenfilename(initialdir=self.fname.get().rsplit('/', 1)[0], title='Select file...', filetypes=(('Fits files', '.fits'), ))

        if tmp not in ['', ()]:
            # Update file name
            self.fname.set(tmp)
            
            # Load data
            self.loadFitsFiles()
            
            # Update figures accordingly
            self.updateFigures()
            
            # Set flag to True if data was succesfully loaded
            self.imLoaded      = True
            
            # Set number of plots
            self.root.numPlots = 3
            
            
    def loadFitsFiles(self):
        '''Loads a .fits file with 3 extensions and update the plots in the bottom window.'''
        
        try:
            hdul = fits.open(self.fname.get())
        except IOError:
            messagebox.showerror('Invalid input', 'The given file name %s is not a .fits file' %self.fname.get())
            return
        
        self.image, self.model, self.res = container(), container(), container()
        
        # Getting data
        self.image.data    = hdul[1].data
        self.model.data    = hdul[2].data
        
        self.image.cmapMax = np.max([self.model.data.max(), self.image.data.max()])
        self.model.cmapMax = self.image.cmapMax
        
        self.image.cmapMin = np.min([self.model.data.min(), self.image.data.min()])
        self.model.cmapMin = self.image.cmapMin
        
        self.res.data      = hdul[3].data
        
        
    def updateFigures(self):
        '''Updates the main three figures'''
        
        try:
            self.root.bottomPane.plot1.updateImage(self.image.data, maxi=self.image.cmapMax,  mini=self.image.cmapMin,  cmap=self.cmap.var.get())
            self.root.bottomPane.plot2.updateImage(self.model.data, maxi=self.model.cmapMax,  mini=self.model.cmapMin,  cmap=self.cmap.var.get())
            self.root.bottomPane.plot3.updateImage(self.res.data,   maxi=self.res.data.max(), mini=self.res.data.min(), cmap=self.cmap.var.get())
        except AttributeError:
            messagebox.showerror('Invalid input', 'The given file name %s is not a .fits file' %self.fname.get())

        
    def updateCmap(self, event):
        '''Update the cmap of the already plotted images.'''
        
        if self.imLoaded:
            for plot in [self.root.bottomPane.plot1, self.root.bottomPane.plot2, self.root.bottomPane.plot3]:
                plot.im.set_cmap(self.cmap.var.get())
                plot.canvas.draw()
            
    
    def invertxAxes(self):
        '''Invert the x axis of the selected graphs'''
        
        for plot in [self.root.bottomPane.plot1, self.root.bottomPane.plot2, self.root.bottomPane.plot3]:
            if plot.selected:
                plot.ax.set_xlim(plot.ax.get_xlim()[::-1])
                plot.canvas.draw()
            
             
    def invertyAxes(self):
        for plot in [self.root.bottomPane.plot1, self.root.bottomPane.plot2, self.root.bottomPane.plot3]:
            if plot.selected:
                plot.ax.set_ylim(plot.ax.get_ylim()[::-1])
                plot.canvas.draw()
                
                
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
        self.modelName           = tk.StringVar(value='Exponential disk')
        
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
        
        # Help menu within top menu
        self.helpMenu = tk.Menu(self.topMenu, tearoff=0)
        self.helpMenu.add_command(label='Shortcuts (Ctrl+H)', command=self.showShortcuts)
        self.helpMenu.add_command(label='Galfit website',     command=lambda:print('work in progress'))
        
        # Adding sections into top menu
        self.topMenu.add_cascade( label="File", menu=self.fileMenu)
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
        
        # Set number of plots in bottom frame to None till files are opened
        self.numPlots          = 3
        
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
        self.rightFrame.frame.bind('<Enter>', self.setScrollable)
        self.rightFrame.frame.bind('<Leave>', self.unsetScrollable)
        
        # Drawing frames
        self.topFrame.frame.grid(   row=0, sticky=tk.N+tk.S+tk.W+tk.E, columnspan=3)
        self.bottomFrame.frame.grid(row=1, sticky=tk.N+tk.S+tk.W+tk.E, columnspan=2)
        self.rightFrame.frame.grid( row=1, sticky=tk.N+tk.S+tk.W+tk.E, column=2)
        
        # Setting grid geometry for main frames
        tk.Grid.rowconfigure(   self.parent, 0, weight=0, minsize=130)
        tk.Grid.rowconfigure(   self.parent, 1, weight=1)
        tk.Grid.columnconfigure(self.parent, 2, weight=0, minsize=100)
        tk.Grid.columnconfigure(self.parent, 1, weight=1)
        
        print(self.bottomFrame.frame['width'], self.bottomFrame.frame['height'])
        
        
    def setScrollable(self, event):
        self.parent.bind('<MouseWheel>', self.setMouseWheel)
        self.parent.bind("<Button-4>",   self.setMouseWheel)
        self.parent.bind("<Button-5>",   self.setMouseWheel)
        
        
    def unsetScrollable(self, event):
        self.parent.unbind('<MouseWheel>')
        self.parent.unbind("<Button-4>")
        self.parent.unbind("<Button-5>")
            
        
    def setMouseWheel(self, event):
        print('oui')
        if self.rightPane.canvas.bbox('all')[3] > self.rightPane.labelFrame.winfo_height():
            if event.num==5 or event.delta<0:
                step = -1
            else:
                step = 1            
            self.rightPane.canvas.yview_scroll(step, 'units')
        
        
    def defaultState(self, event):
        '''Change the graphFrame instance back to default state (where the user can select plots)'''
        
        if self.state != 'default':
            self.bottomFrame.frame.config(cursor='arrow')
            self.state = 'default'
            
            
    def cancel(self, event):
        '''Cancel PA line drawing if Ctrl-z is pressed'''
        
        if self.bottomPane.activeSingleFrame.paLine.line is not None:
            self.bottomPane.activeSingleFrame.paLine.drawing = False
            self.bottomPane.activeSingleFrame.paLine.line.set_data([[], []])
            self.bottomPane.activeSingleFrame.canvas.draw()
        
        
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
        if self.numPlots > self.bottomPane.numSelected:
            for plot in [self.bottomPane.plot1, self.bottomPane.plot2, self.bottomPane.plot3]:
                if not plot.selected:
                    plot.changeBorder()
            self.bottomPane.numSelected = self.numPlots
        # Case 2: all the plots are selected, so we unselected them all
        elif self.numPlots == self.bottomPane.numSelected:
            for plot in [self.bottomPane.plot1, self.bottomPane.plot2, self.bottomPane.plot3]:
                plot.changeBorder()
            self.bottomPane.numSelected = 0
        

def main(): 
    
    root = tk.Tk()
    root.title('GalBit - Easily do stuff')
    root.geometry("1500x800")
    app  = mainApplication(root)
    
    root.protocol("WM_DELETE_WINDOW", app.exitProgram)
    
    root.mainloop()

if __name__ == '__main__':
    main()