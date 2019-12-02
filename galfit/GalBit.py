#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:18:55 2019

@author: wilfried
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


class container:
    def __init__(self):
        info = 'A simple container'


class singlePlotFrame:
    def __init__(self, parent, root, data=None, title=None, bgColor='beige'):
        
        # General properties
        self.bdOn           = 'black'
        self.bdOff          = bgColor
        
        self.clicked        = False
        
        self.parent         = parent
        self.root           = root
        
        # PA line properties
        self.paLine         = container()
        self.paLine.posx    = [None, None]
        self.paLine.posy    = [None, None]
        self.paLine.drawing = False
        self.paLine.line    = None
        
        # Making a figure
        self.bgColor        = bgColor
        self.figure         = Figure(figsize=(4,4), tight_layout=True, facecolor=self.bgColor)
            
        # Creating an axis
        self.ax             = self.figure.add_subplot(111)
        self.ax.yaxis.set_ticks_position('both')
        self.ax.xaxis.set_ticks_position('both')
        self.ax.tick_params(which='both', direction='in', labelsize=12)
        self.ax.grid()
        
        # Adding a title
        if title is not None and type(title) is str:
            self.title = title
            self.ax.set_title(self.title)
        
        # Plotting
        if data is not None:
            self.im = self.ax.imshow(data, origin='lower')
            
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Linking to events
        self.canvas.mpl_connect('button_press_event',  self.onClick)
        self.canvas.mpl_connect('motion_notify_event', self.onMove)
        self.canvas.mpl_connect('key_press_event',     self.keyPressed)
        
    
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
        self.im = self.ax.imshow(newData, cmap=cmap, norm=norm, origin='lower')
        self.canvas.draw()
        
        
    def onClick(self, event):
        '''Adds or remove a border around the figure canvas every time the figure is clicked on.'''
        
        if self.root.state == 'default':
            if self.parent['bg'] == self.bdOn:
                self.parent.config({'bg':self.bdOff})
                self.clicked = False
            else:
                self.parent.config({'bg':self.bdOn})
                self.clicked = True
        else:
            
            if not self.paLine.drawing:
                self.paLine.posx[0] = event.xdata
                self.paLine.posy[0] = event.ydata
                self.paLine.drawing = True
            else:
                self.paLine.drawing = False
                self.paLine.line    = self.ax.plot(self.paLine.posx, self.paLine.posy, 'k-')
                self.canvas.draw()
            
            
    def onMove(self, event):
        '''Update the image name, x and y positions of the mouse cursor printed on the top pane.'''
        
        if self.root.topPane.hover.var.get() != self.title:
            self.root.topPane.hover.var.set( 'Current image: %s' %self.title)
            
        if event.xdata is not None:
            self.root.topPane.hover.xvar.set('x: %.2f' %event.xdata)
            
        if event.ydata is not None:
            self.root.topPane.hover.yvar.set('y: %.2f' %event.ydata)
            
        if self.paLine.drawing:
            self.paLine.posx[1] = event.xdata
            self.paLine.posy[1] = event.ydata
        
        
    def keyPressed(self, event):
        print(self.ax.lines, self.paLine.line)
        '''if event.key == 'ctrl+z':
            if self.paLine.line is not None:
                self.ax.lines.remove(self.paLine.line)
                self.canvas.draw()'''
        


class graphFrame:
    def __init__(self, parent, root, bgColor='beige'):
        self.parent = parent
        self.root   = root
        self.bdSize = 5
        
        self.leftFrame  = tk.Frame(self.parent, bd=self.bdSize, bg=bgColor)
        self.midFrame   = tk.Frame(self.parent, bd=self.bdSize, bg=bgColor)
        self.rightFrame = tk.Frame(self.parent, bd=self.bdSize, bg=bgColor)
        
        self.plot1      = singlePlotFrame(self.leftFrame,  self.root, title='data',     bgColor=bgColor)
        self.plot2      = singlePlotFrame(self.midFrame,   self.root, title='model',    bgColor=bgColor)
        self.plot3      = singlePlotFrame(self.rightFrame, self.root, title='residual', bgColor=bgColor)
        
        self.leftFrame.grid(row=0, column=0, padx=5, pady=5)
        self.midFrame.grid(row=0, column=1, padx=5, pady=5)
        self.rightFrame.grid(row=0, column=3, padx=5, pady=5)
        


class topFrame:
    '''
    Top frame with options used to import data.
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
        
        self.loadButton = tk.Button(self.parent, command=self.openFile, text='Browse')
        self.loadInput  = tk.Entry( self.parent, cursor='xterm', textvariable=self.fname, width=50)
        
        self.sendButton = tk.Button(self.parent, command=self.updateFigures, text='Load')
        
        # Making the cmap list widget
        cmapNames        = list(matplotlib.cm.cmap_d.keys())
        cmapNames.sort()
        
        self.cmap        = container()
        self.cmap.var    = tk.StringVar(value='bwr')
        self.cmap.list   = ttk.Combobox(self.parent, textvariable=self.cmap.var, values=cmapNames)
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
        
    def openFile(self):
        '''Opens a file using self.fname.get() value.'''
        
        tmp = askopenfilename(initialdir=self.fname.get().rsplit('/', 1)[0], title='Select file...', filetypes=(('Fits files', '.fits'), ))
#        raise IOError(tmp)
        if tmp not in ['', ()]:
            # Update file name
            self.fname.set(tmp)
            
            # Load data
            self.loadFitsFiles()
            
            # Update figures accordingly
            self.updateFigures()
            
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
        
        # Set flag to True if data was succesfully loaded
        self.imLoaded      = True
        
        # Enable invert axes checkboxes
        self.invert.x.config({'state':'normal'})
        self.invert.y.config({'state':'normal'})
        
    def updateFigures(self):
        '''Updates the main three figures'''
        
        try:
            self.root.bottomPane.plot1.updateImage(self.image.data, maxi=self.image.cmapMax,  mini=self.image.cmapMin,  cmap=self.cmap.var.get())
            self.root.bottomPane.plot2.updateImage(self.model.data, maxi=self.model.cmapMax,  mini=self.model.cmapMin,  cmap=self.cmap.var.get())
            self.root.bottomPane.plot3.updateImage(self.res.data,   maxi=self.res.data.max(), mini=self.res.data.min(), cmap=self.cmap.var.get())
        except AttributeError:
            messagebox.showerror('Invalid input', 'The given file name %s is not a .fits file' %self.fname.get())

        
    def updateCmap(self, event):
        if self.imLoaded:
            self.updateFigures()
            
    
    def invertxAxes(self):
        for i in [self.root.bottomPane.plot1, self.root.bottomPane.plot2, self.root.bottomPane.plot3]:
            if i.clicked:
                i.ax.set_xlim(i.ax.get_xlim()[::-1])
                i.canvas.draw()
            
    def invertyAxes(self):
        for i in [self.root.bottomPane.plot1, self.root.bottomPane.plot2, self.root.bottomPane.plot3]:
            if i.clicked:
                i.ax.set_ylim(i.ax.get_ylim()[::-1])
                i.canvas.draw()
        
        
class rightFrame:
    '''
    Right frame window with different option to trigger.
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
        
        self.parent = parent
        self.root   = root


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
        
        # Set default cursor
        self.state             = 'default'
        self.parent.config(cursor='arrow')
        
        # Set containers
        self.topFrame, self.rightFrame, self.bottomFrame = container(), container(), container()
        
        # Set frame colors
        self.topFrame.color    = 'grey'
        self.rightFrame.color  = 'grey'
        self.bottomFrame.color = 'beige'
        
        # Making main three frames
        self.topFrame.frame    = tk.Frame(self.parent, bg=self.topFrame.color)
        self.rightFrame.frame  = tk.Frame(self.parent, bg=self.rightFrame.color)
        self.bottomFrame.frame = tk.Frame(self.parent, bg=self.bottomFrame.color)
        
        # Creating widgets within frames
        self.topPane           = topFrame(  self.topFrame.frame,    self, bgColor=self.topFrame.color)
        self.rightPane         = rightFrame(self.rightFrame.frame,  self, bgColor=self.rightFrame.color)
        self.bottomPane        = graphFrame(self.bottomFrame.frame, self, bgColor=self.bottomFrame.color)
        
        # Binding key events
        self.parent.bind('<Control-p>',  self.lineTracingState)
        self.parent.bind('<Escape>',     self.defaultState)
        
        # Setting grid geometry for main frames
        tk.Grid.rowconfigure(   self.parent, 0, weight=2)
        tk.Grid.columnconfigure(self.parent, 0, weight=7)
        tk.Grid.rowconfigure(   self.parent, 1, weight=7)
        tk.Grid.columnconfigure(self.parent, 1, weight=3)
        
        # Drawing frames
        self.topFrame.frame.grid(   row=0, sticky=tk.N+tk.S+tk.W+tk.E, columnspan=2)
        self.bottomFrame.frame.grid(row=1, sticky=tk.N+tk.S+tk.W+tk.E)
        self.rightFrame.frame.grid( row=1, sticky=tk.N+tk.S+tk.W+tk.E, column=1)
        
    def defaultState(self, event):
        if self.state != 'default':
            self.bottomFrame.frame.config(cursor='arrow')
            self.state = 'default'
        
        
    def lineTracingState(self, event):
        if self.state != 'lineTracing':
            self.bottomFrame.frame.config(cursor='crosshair')
            self.state = 'lineTracing'
        

def main(): 
    
    def exitProgram():
        root.destroy()
    
    root = tk.Tk()
    root.title('GalBite - Easily do stuff')
    root.geometry("1400x1000")
    mainApplication(root)
    
    root.protocol("WM_DELETE_WINDOW", exitProgram)
    
    root.mainloop()

if __name__ == '__main__':
    main()