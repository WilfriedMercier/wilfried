#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

Geometrical transformations between coordinate systems.
"""

import numpy         as     np
from   numpy         import ndarray
from   astropy.units import Quantity
from   typing        import Union, Tuple

class DiskGeometry:
    '''Geometry of a razor-thin disk.'''
    
    def __init__(self, inc: Union[int, float]=0, PA: Union[int, float]=0, e: Union[int, float]=None) -> None:
        '''
        Initialise a razor-thin disk.
        
        :param inc: disk inclination in degrees (0 means face-on, 90 means edge-on). If **e** is provided, it is not used.
        :type inc: int or float
        :param PA: position angle (0 means major axis is vertical)
        :type PA: int or float
        :param e: disk ellipticity (0 means circular, 1 means edge-on). If provided, it overrides **inc**.
        '''
        
        for pname, param in zip(['inc', 'PA'], [inc, PA]):
            if not isinstance(param, (int, float)):
                raise TypeError(f'parameter {pname} has type {type(param)} but it must have type int or float.')
        
        if inc<0 or inc>90:
            raise ValueError('inclination must be between 0 and 90 degrees.')
            
        if PA<-180 or PA>180:
            raise ValueError('position angle must be between -180 and 180 degrees.')
        
        if e is None:
            self.e = 1 - np.cos(Quantity(inc, unit='deg').to('rad').value)
        elif not isinstance(e, (int, float)):
            raise TypeError(f'parameter e has type {type(e)} but it must have type int or float.')
        elif e<0 or e>1:
            raise ValueError(f'parameter e has value {e} but it must be between 0 and 1.')
        else:
            #: Ellipticity
            self.e = e
        
        #: Position angle
        self.PA    = Quantity(PA, unit='deg')
        
    def XY(self, X_sky: ndarray, Y_sky: ndarray, *args, **kwargs) -> Tuple[ndarray, ndarray]:
        '''
        Compute the X, Y coordinates in the disk plane from sky coordinates.
        
        :param ndarray X_sky: X-coordinates on sky plane
        :param ndarray Y_sky: Y-coordinates on sky plane
        
        :returns: X, Y in disk plane
        :rtype: ndarray, ndarray
        '''
            
        cos_PA = np.cos(self.PA.to('rad').value)
        sin_PA = np.sin(self.PA.to('rad').value)
        
        X = (X_sky*cos_PA + Y_sky*sin_PA) / (1-self.e)
        Y = (Y_sky*cos_PA - X_sky*sin_PA)
        
        return X, Y
    
    def distance(self, X_sky: ndarray, Y_sky: ndarray, *args, **kwargs) -> ndarray:
        '''
        Compute the X, Y coordinates in the disk plane from sky coordinates.
        
        :param ndarray X_sky: X-coordinates on sky plane
        :param ndarray Y_sky: Y-coordinates on sky plane
        
        :returns: X, Y in disk plane
        :rtype: ndarray, ndarray
        '''
        
        X, Y = self.XY(X_sky, Y_sky, *args, **kwargs)
        return np.sqrt(X*X + Y*Y)
        
            
            
            
            
            
            
            