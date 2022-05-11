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
from   abc           import ABC, abstractmethod

class Geometry(ABC):
   
   def __init__(self, *args, **kwargs):
      pass
   
   @abstractmethod
   def XY(self, X_sky: Union[float, ndarray], Y_sky: Union[float, ndarray], *args, **kwargs) -> Tuple[Union[float, ndarray], Union[float, ndarray]]:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Compute the X, Y coordinates in the the given geometry from sky coordinates.
      
      :param X_sky: X-coordinates on sky plane
      :type X_sky: float or numpy ndarray
      :param Y_sky: Y-coordinates on sky plane
      :type Y_sky: float or numpy ndarray
      
      :returns: X, Y in disk plane
      :rtype: tuple(float or ndarray, float or ndarray)
      '''
      
      return
   
   def distance(self, X_sky: Union[float, ndarray], Y_sky: Union[float, ndarray], *args, **kwargs) -> Union[float, ndarray]:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Compute the distance in the given geometry from sky coordinates.
      
      :param X_sky: X-coordinates on sky plane
      :type X_sky: float or ndarray
      :param Y_sky: Y-coordinates on sky plane
      :type Y_sky: float or ndarray
      
      :returns: distance
      :rtype: float or ndarray
      '''
      
      X, Y = self.XY(X_sky, Y_sky, *args, **kwargs)
      return np.sqrt(X*X + Y*Y)

class BulgeGeometry(Geometry):
   r'''
   .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
   
   Geometry of a spherically symetric bulge.
   '''
   
   def __init__(self, *args, **kwargs) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Initialise a spherically symetric bulge.
      '''
      
      super().__init__(*args, **kwargs)
   
   def XY(self, X_sky: Union[float, ndarray], Y_sky: Union[float, ndarray], *args, **kwargs) -> Tuple[Union[float, ndarray], Union[float, ndarray]]:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Compute the X, Y coordinates in the bulge from sky coordinates.
      
      :param X_sky: X-coordinates on sky plane
      :type X_sky: float or ndarray
      :param Y_sky: Y-coordinates on sky plane
      :type Y_sky: float or ndarray
      
      :returns: X, Y in disk plane
      :rtype: tuple(float or ndarray, float or ndarray)
      '''
      
      return X_sky, Y_sky

class DiskGeometry(Geometry):
   r'''
   .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
   
   Geometry of a razor-thin disk.
    
   :param inc: (**Optional**) disk inclination in degrees (0 means face-on, 90 means edge-on). If **e** is provided, it is not used.
   :type inc: int or float
   :param PA: (**Optional**) position angle (0 means major axis is vertical)
   :type PA: int or float
   :param e: (**Optional**) disk ellipticity (0 means circular, 1 means edge-on). If provided, it overrides **inc**.
   '''
    
   def __init__(self, inc: Union[int, float] = 0, PA: Union[int, float] = 0, e: Union[int, float] = None) -> None:
      r'''
      .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
      
      Initialise a razor-thin disk.
      '''
      
      super().__init__()
      
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
       
   def XY(self, X_sky: Union[float, ndarray], Y_sky: Union[float, ndarray], *args, **kwargs) -> Tuple[Union[float, ndarray], Union[float, ndarray]]:
       r'''
       .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
       
       Compute the X, Y coordinates in the disk plane from sky coordinates.
       
       :param X_sky: X-coordinates on sky plane
       :type X_sky: float or ndarray
       :param Y_sky: Y-coordinates on sky plane
       :type Y_sky: float or ndarray
       
       :returns: X, Y in disk plane
       :rtype: tuple(float or ndarray, float or ndarray)
       '''
           
       cos_PA = np.cos(self.PA.to('rad').value)
       sin_PA = np.sin(self.PA.to('rad').value)
       
       X = (X_sky*cos_PA + Y_sky*sin_PA) / (1-self.e)
       Y = (Y_sky*cos_PA - X_sky*sin_PA)
       
       return X, Y
       
           
           
        
            
            
            
            