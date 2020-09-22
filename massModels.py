#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:39:08 2020

@author: Mercier Wilfried - IRAP

Test functions on Galaxy mass modelling.
"""

import numpy             as     np
from   .models           import compute_bn, checkAndComputeIe, sersic_profile
from   scipy.special     import gamma, gammainc
from   math              import factorial
from   astropy.constants import G

#######################################################################################
#                           3D profiles and their functions                           #
#######################################################################################

class Hernquist:
    '''3D Hernquist model class with useful methods.'''
    
    def __init__(self, a, M):
        '''
        Parameters
        ----------
            a : float or int
                scale factor
            M : float or int
               total "mass" (can also be understood as luminosity)
        '''
        
        if not isinstance(a, (int, float)) or not isinstance(M, (int, float)):
            raise TypeError('One of the parameters is neither int nor float.')
        
        self.a     = a
        self.M     = M
        self._dim  = 3
        
        
    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
        
    def gfield(self, r):
        '''
        Compute the gravitational field of a single Hernquist profile at radius r.

        Parameters
        ----------
            r : float or int
                radius where the gravitational field is computed

        Return g(r).
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
        
        return (-G*self.luminosity(r)/r**2).value
        
    
    def luminosity(self, r):
        '''
        Compute the enclosed luminosity or mass at radius r.

        Parameters
        ----------
            r : float or int
                radius where to compute the luminosity

        Return the luminosity or mass enclosed in a sphere of radius r.
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
        
        return self.M*(1 - 1.0/(1+r/self.a)**2)
        
        
    def profile(self, r):
        '''
        Compute the light or mass profile at radius r.

        Parameters
        ----------
            r : float or int
                radius where to compute the profile

        Return the light or mass profile at radius r.
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
        
        return self.M/(2*np.pi*self.a**2) / (r * (1+r/self.a)**3)
    
    @property
    def toSersic(self):
        '''Create the best fit Sersic instance from this Hernquist instance.'''
        
        Re = self.Re**(1.0/0.6468628627045541) / 0.3514010020474344
        bn = compute_bn(4)
        Ie = self.M*bn**8 / (factorial(8)*np.pi*np.exp(bn)*Re**2)
        
        return Sersic(4, Re, Ie=Ie)
    
    
    def velocity(self, r):
        '''
        Velocity profile for a self-sustaining Hernquist 3D profile against its own gravity through centripedal acceleration.

        Parameters
        ----------
            r : float or int
                radius where the velocity profile is computed

        Return V(r).
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
        
        return np.sqrt(G*self.luminosity(r)/r).value


class Sersic:
    '''
    2D Sersic profile class.
    
    This class combine into a single objects useful functions related to Sersic profiles
    '''
    
    def __init__(self, n, Re, Ie=None, mag=None, offset=None):
        """
        General Sersic profile. You can either provide n, Re and Ie, or instead n, Re, mag and offset.
        If neither Ie, nor mag and offset are provided a ValueError is raised.  
        
        Mandatory parameters
        --------------------
            n : float or int
                Sersic index
            re : float or int
                half-light radius
                
        Optional parameters
        -------------------
            Ie : float
                intensity at half-light radius. Default is None.
            mag : float
                galaxy total integrated magnitude used to compute Ie if not given. Default is None.
            offset : float
                magnitude offset in the magnitude system used. Default is None.
        """
        
        if not isinstance(n, (float, int)) or not isinstance(Re, (float, int)):
            raise TypeError('Either n or Re is neither int nor float.')
            
        if n<0:
            raise ValueError('n must be positive valued. Cheers !')
            
        if Re<0:
            raise ValueError('The effective radius Re must be positive valued. Cheers !')
        
        self.n    = n
        self.Re   = Re
        self.bn   = compute_bn(self.n)
        self.Ie   = checkAndComputeIe(Ie, n, self.bn, Re, mag, offset)
        self.ndim = 2
        

    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
        
    def profile(self, r):
        '''
        Sersic profile at radius r.

        Parameters
        ----------
            r : float or int
                radius where the profile is computed

        Return the Sersic profiles computed at the radius r.
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
        
        return sersic_profile(r, self.n, self.Re, Ie=self.Ie, bn=self.bn)
    
    
    def luminosity(self, r):
        '''
        Luminosity at radius r (encompassed within a disk since this is a 2D profile).

        Parameters
        ----------
            r : float or int
                radius where the profile is computed

        Return the luminosity computed within a disk of radius r.
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
           
        n2 = 2*self.n
        return n2*np.pi*self.Ie*np.exp(self.bn)*gammainc(n2, self.bn*(r/self.Re)**(1.0/self.n))*self.Re**2 / self.bn**n2
    
    @property
    def Ltot(self):
        '''Total luminosity of the profile.'''
        
        n2 = 2*self.n
        return np.pi*n2*self.Ie*np.exp(self.bn)*gamma(n2)*self.Re**2 / self.bn**(n2)
    
    @property
    def toHernquist(self):
        '''Create the best-fit Hernquist instance from this Sersic instance (based on my own calculations).'''
        
        if self.n==4:
            a = 0.3514010020474344*self.Re**0.6468628627045541
            M = self.Ltot
        else:
            raise ValueError('Only de Vaucouleurs profiles, i.e. Sersic with n=4, can be converted into Hernquist profiles.')
        
        return Hernquist(a, M)
    
    
class ExponentialDisk(Sersic):
    '''
    2D/3D Exponential disk profile class.
    
    This class combine into a single objects useful functions related to Sersic profiles
    '''
    
    def __init__(self, Re, Ie=None, mag=None, offset=None):        
        """
        You can either provide n, Re and Ie, or instead n, Re, mag and offset.
        If neither Ie, nor mag and offset are provided a ValueError is raised.  
        
        Mandatory parameters
        --------------------
            re : float or int
                half-light radius
                
        Optional parameters
        -------------------
            Ie : float
                intensity at half-light radius. Default is None.
            mag : float
                galaxy total integrated magnitude used to compute Ie if not given. Default is None.
            offset : float
                magnitude offset in the magnitude system used. Default is None.
        """
        
        super().__init__(1, Re, Ie=Ie, mag=mag, offset=offset)
        
        
    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
        
    def gfield(self, r):
        '''
        Compute the gravitational field of a single infinitely thin exponential disk profile at radius r.

        Parameters
        ----------
            r : float or int
                radius where the gravitational field is computed

        Return g(r).
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
                
        return (-2*G*self.luminosity(r)/r).value
    
    
    def velocity(self, r):
        '''
        Velocity profile for a self-sustaining 3D inifinitely thin disk against its own gravity through centripedal acceleration.

        Parameters
        ----------
            r : float or int
                radius where the velocity profile is computed

        Return V(r).
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
        
        return np.sqrt(2*G*self.luminosity(r)).value


