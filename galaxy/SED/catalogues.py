#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Hugo Plombat - LUPM & Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Base classes used to generate catalogues for LePhare or Cigale SED fitting codes.
"""

from   astropy.table import Table
from   .misc         import Property

class Catalogue:
    r'''Class implementing a catalogue consisting of as Astropy Table and additional information used by the SED fitting codes.'''
    
    def __init__(self, fname, table, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Init general catalogue. This is supposed to be subclassed to account for specificities of LePhare or Cigale catalogues.

        :param str fname: name of the output file containing the catalogue when it is saved
        :param table: input table
        :type table: Astropy Table
        
        :raises TypeError:
             
            * if **table** is not an astropy Table
            * if **fname** is not of type str
        ''' 
        
        if not isinstance(table, Table):
            raise TypeError(f'table has type {type(table)} but it must be an Astropy Table.')
            
        if not isinstance(fname, str):
            raise TypeError(f'fname parameter has type {type(fname)} but it must have type str.')
            
        self.fname  = fname
        self.data   = table
        
        
    def save(self, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Save the catalogue into the given file.
        
        :param *args: parameters passed to Astropy.table.Table.writeto method
        :param **kwargs: optional parameters passed to Astropy.table.Table.writeto method
        '''
            
        self.data.writeto(self.name, *args, **kwargs)
        return
    
    def text(self, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Return a text representation of the catalogue used when making the parameter files.
        '''
        
        raise NotImplementedError('this method must be implemented in subclasses.')
        return
    
class LePhareCat(Catalogue):
    r'''Class implementing a catalogue compatible with LePhare SED fitting code.'''
    
    def __init__(self, fname, table, tunit='M', magtype='AB', tformat='MEME', ttype='LONG', nlines=[0, 100000000]):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Init LePhare catalogue.

        :param str fname: name of the output file containing the catalogue when it is saved
        :param table: input table
        :type table: Astropy Table
        
        :param str tunit: (**Optional**) unit of the table data. Must either be 'M' for magnitude or 'F' for flux.
        :param str magtype: (**Optional**) magnitude type if data are in magnitude unit. Must either be 'AB' or 'VEGA'.
        :param str tformat: (**Optional**) format of the table. Must either be 'MEME' if data and error columns are intertwined or 'MMEE' if columns are first data and then errors.
        :param str ttype: (**Optional**) data type. Must either be SHORT or LONG.
        :param list[int] nlines: (**Optional**) first and last line of the catalogue to be used during the SED fitting

        :raises TypeError:
             
            * if **table** is not an astropy Table
            * if one of **fname**, **tunit**, **magtype**, **tformat** or **ttype**  is not of type str
            * if **nlines** is not a list
            
        :raises ValueError:
        
            * if **tunit** is neither F, nor M
            * if **magtype** is neither AB nor VEGA
            * if **tformat** is neither MEME nor MMEE
            * if **ttype** is neither SHORT nor LONG
            * if **nlines** values are not int, or if first value is less than 0, or if second value is less than the first one
        ''' 
            
        if not all([isinstance(i, str) for i in [tunit, magtype, tformat, ttype]]):
            raise TypeError(f'one of the parameters in tunit, magtype, tformat or ttype is not of type str.')
            
        super().__init__(fname, table)
        
        #####################################################
        #             Define default properties             #
        #####################################################
        
        self.unit   = Property('M', str, 
                               testFunc = lambda value: value not in ['F', 'M'], 
                               testMsg  = f'tunit has value {tunit} but it must either be F (for flux) or M (for magnitude).')
        
        self.mtype  = Property('AB', str,
                               testFunc = lambda value: value not in ['AB', 'VEGA'],
                               testMsg  = f'magtype has value {magtype} but it must either be AB or VEGA.')
        
        self.format = Property('MEME', str,
                               testFunc = lambda value: value not in ['MEME', 'MMEE'],
                               testMsg  = f'tformat has value {tformat} but it must either be MEME (if each error column follows its associated magnitude column) or MMEE (if magnitudes are listed first and then errors).')
        
        self.ttype  = Property('LONG', str,
                               testFunc = lambda value: value not in ['SHORT', 'LONG'],
                               testMsg  = f'ttype has value {ttype} but it must either be SHORT or LONG.')
        
        self.nlines = Property([0, 100000000], list, subtypes=int, minBound=0,
                               testFunc = lambda value: value[1] < value[0],
                               testMsg  = f'maximum number of lines ({nlines[1]}) is less than minimum one ({nlines[0]}).')
        
        #################################################
        #              Set to given values              #
        #################################################
        
        self.unit.set(  tunit)
        self.mtype.set( magtype)
        self.format.set(tformat)
        self.ttype.set( ttype)
        self.nlines.set(nlines)
    
    @property
    def text(self, *args, **kwargs):
        r'''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
        
        Return a text representation of the catalogue used when making the parameter files.
        
        :returns: output representation
        :rtype: str
        '''
        
        text =  f'''
        #-------    Input Catalog Informations   
        CAT_IN \t{self.fname}

        INP_TYPE \t{self.unit} \t# Input type (F:Flux or M:MAG)
        CAT_MAG \t{self.mtype} \t# Input Magnitude (AB or VEGA)
        CAT_FMT \t{self.format} \t# MEME: (Mag,Err)i or MMEE: (Mag)i,(Err)i  
        CAT_LINES \t{self.nlines} \t# MIN and MAX RANGE of ROWS used in input cat [def:-99,-99]
        CAT_TYPE \t{self.ttype} # Input Format (LONG,SHORT-def)
        '''
        
        return text