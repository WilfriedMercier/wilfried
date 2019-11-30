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

class singlePlotFrame:
    def __init__(self, parent, root, data=None):
        self.parent   = parent
        self.root     = root
        
        # Making a figure
        self.figure   = Figure(figsize=(4,4))
        self.figure.patch.set_alpha(0.)
        self.figure.tight_layout()
        
        # Creating an axis
        self.ax       = self.figure.add_subplot(111)
        self.ax.yaxis.set_ticks_position('both')
        self.ax.xaxis.set_ticks_position('both')
        self.ax.tick_params(which='both', direction='in', labelsize=12)
        self.ax.grid()
        
        if data is not None:
            self.im = self.ax.imshow(data)
            
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
    def updateImage(self, newData, mini=None, maxi=None, cmap='bwr'):
        norm    = DivergingNorm(vcenter=0, vmin=mini, vmax=maxi)
        self.im = self.ax.imshow(newData, cmap=cmap, norm=norm)
        self.canvas.draw()
        


class graphFrame:
    def __init__(self, parent, root):
        self.parent = parent
        self.root   = root
        
        self.leftFrame  = tk.Frame(self.parent)
        self.midFrame   = tk.Frame(self.parent)
        self.rightFrame = tk.Frame(self.parent)
        
        self.plot1      = singlePlotFrame(self.leftFrame,  self.root)
        self.plot2      = singlePlotFrame(self.midFrame,   self.root)
        self.plot3      = singlePlotFrame(self.rightFrame, self.root)
        
        self.leftFrame.grid(row=0, column=0, padx=5, pady=5)
        self.midFrame.grid(row=0, column=1, padx=5, pady=5)
        self.rightFrame.grid(row=0, column=3, padx=5, pady=5)
        


class topFrame:
    '''
    Top frame with options used to import data.
    '''
    def __init__(self, parent, root):
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
        self.imLoaded   = False
        self.fname      = tk.StringVar()
        self.fname.set('/home/wilfried/Thesis/galfit/modelling/outputs/')
        
        padx            = 10
        pady            = 10
        
        self.loadButton = tk.Button(self.parent, command=self.openFile, text='Browse')
        self.loadInput  = tk.Entry(self.parent, cursor='xterm', textvariable=self.fname, width=50)
        
        self.sendButton = tk.Button(self.parent, command=self.openFile, text='Load')
        
        # Making the cmap list widget
        cmapNames       = list(matplotlib.cm.cmap_d.keys())
        self.cmap       = tk.StringVar(value='bwr')
        self.cmapList   = ttk.Combobox(self.parent, textvariable=self.cmap, values=cmapNames)
        self.cmapList.bind("<<ComboboxSelected>>", self.loadFitsFiles)
        
        
        # Drawing elements
        self.loadInput.grid( row=0,           padx=padx, pady=pady)
        self.loadButton.grid(row=0, column=1, padx=padx, pady=pady)
        self.sendButton.grid(row=1, column=0, padx=padx, pady=pady, columnspan=2)
        self.cmapList.grid(  row=0, column=2, padx=padx, pady=pady)
        
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
        
        # Getting data
        self.image       = hdul[1].data
        self.model       = hdul[2].data
        
        self.image.max   = np.max([self.model, self.image])
        self.model.max   = self.image.max
        
        self.image.min   = np.min([self.model, self.image])
        self.model.min   = self.image.min
        
        self.res         = hdul[3].data
        
        self.res.max     = np.max(self.res)
        self.res.min     = np.min(self.res)
        
    def updateFigures(self):
        '''Updates the main three figures'''
        
        self.root.bottomPane.plot1.updateImage(self.image, maxi=self.image.max, mini=self.image.min, cmap=self.cmap.get())
        self.root.bottomPane.plot2.updateImage(self.model, maxi=self.model.max, mini=self.model.min, cmap=self.cmap.get())
        self.root.bottomPane.plot3.updateImage(self.res,   maxi=self.res.max,   mini=self.res.min,   cmap=self.cmap.get())
        
        
class rightFrame:
    '''
    Right frame window with different option to trigger.
    '''
    
    def __init__(self, parent, root):
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
        
        self.parent      = parent
        
        # Making main three frames
        self.topFrame    = tk.Frame(self.parent, bg='grey')
        self.rightFrame  = tk.Frame(self.parent, bg='red')
        self.bottomFrame = tk.Frame(self.parent, bg='green')
        
        # Creating widgets within frames
        self.topPane     = topFrame(self.topFrame, self)
        self.rightPane   = rightFrame(self.rightFrame, self)
        self.bottomPane  = graphFrame(self.bottomFrame, self)
        
        # Setting grid geometry for main frames
        tk.Grid.rowconfigure(self.parent, 0, weight=2)
        tk.Grid.columnconfigure(self.parent, 0, weight=7)
        tk.Grid.rowconfigure(self.parent, 1, weight=7)
        tk.Grid.columnconfigure(self.parent, 1, weight=3)
        
        # Drawing frames
        self.topFrame.grid(row=0, columnspan=2, sticky=tk.N+tk.S+tk.W+tk.E)
        self.bottomFrame.grid(row=1, sticky=tk.N+tk.S+tk.W+tk.E)
        self.rightFrame.grid(row=1, column=1, sticky=tk.N+tk.S+tk.W+tk.E)
        

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