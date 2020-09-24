    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:39:08 2020

@author: Mercier Wilfried - IRAP

Test functions on Galaxy mass modelling.
"""

import scipy.integrate   as     integrate
import numpy             as     np
from   .models           import compute_bn, checkAndComputeIe, sersic_profile
from   .kinematics       import Vprojection
from   scipy.special     import gamma, gammainc
from   math              import factorial
from   astropy.constants import G, c


#######################################################################################
#                           3D profiles and their functions                           #
#######################################################################################

class Multiple3DModels:
    '''A master class used when combining two 3D models into a single object'''
    
    def __init__(self, model1, model2):
        
        if not hasattr(model1, '_dim') or not hasattr(model2, '_dim'):
            raise AttributeError('One of the objects is not a valid model instance.')
        
        if model1._dim < 3 or model2._dim < 3:
            raise ValueError('One of the models is not 3D or higher. Only 3D models can be combined. Cheers !')
        else:
            self._dim   = 3
            
        # We keep track of models in a list
        if not isinstance(model1, Multiple3DModels) and not isinstance(model2, Multiple3DModels):
            self.models = [model1, model2]
        elif not isinstance(model1, Multiple3DModels) and isinstance(model2, Multiple3DModels):
            self.models = model2.models
            self.models.append(model1)
        elif isinstance(model1, Multiple3DModels) and not isinstance(model2, Multiple3DModels):
            self.models = model1.models
            self.models.append(model2)
        else:
            self.models = model1.models + model2.models
            
            
    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
            
    def __add__(self, other):
        '''Add this instance with any other object. Only adding 3D models is allowed.'''
        
        return Multiple3DModels(self, other)
    
    def __checkArgs__(self, args, kwargs):
        '''Check args and kwargs so that they have the same len'''
        
        if len(args) > len(kwargs):
            for i in range(len(args) - len(kwargs)):
                kwargs.append({})
        else:
            for i in range(len(kwargs) - len(args)):
                args.append([])
        
        return args, kwargs
            
    
    def gfield(self, args=[[]], kwargs=[{}]):
        '''
        Compute the full gravitational field at radius r.
        
        Notes: 
            - Because models can require different arguments (for instance 3D radial distance r for one and 2D R in disk plane for the other), these are separated in lists.
              Each element will be passed to the corresponding model. The order is the same as in models list variable.
        
        Parameters
        ----------
            args : list of list
                arguments to pass to each model. Order is that of models list.
            kwargs : list of dict
                kwargs to pass to each model. Order is that of models list.
        '''
        
        args, kwargs = self.__checkArgs__(args, kwargs)
        return np.sum([i.gfield(*args[pos], **kwargs[pos]) for pos, i in enumerate(self.models)], axis=0)
        
        
        
    def luminosity(self, args=[], kwargs=[{}]):
        '''
        Compute the enclosed luminosity or mass at radius r.
        
        Notes: 
            - Because models can require different arguments (for instance 3D radial distance r for one and 2D R in disk plane for the other), these are separated in lists.
              Each element will be passed to the corresponding model. The order is the same as in models list variable.
        
        Parameters
        ----------
            args : list 
                arguments to pass to each model. Order is that of models list.
            kwargs : list of dict
                kwargs to pass to each model. Order is that of models list.
        '''
        
        args, kwargs = self.__checkArgs__(args, kwargs)
        return np.sum([i.luminosity(*args[pos], **kwargs[pos]) for pos, i in enumerate(self.models)], axis=0)
        
            
    def profile(self, args=[], kwargs=[{}]):
        '''
        Compute the light or mass profile at radius r.

        Notes: 
            - Because models can require different arguments (for instance 3D radial distance r for one and 2D R in disk plane for the other), these are separated in lists.
              Each element will be passed to the corresponding model. The order is the same as in models list variable.
        
        Parameters
        ----------
            args : list 
                arguments to pass to each model. Order is that of models list.
            kwargs : list of dict
                kwargs to pass to each model. Order is that of models list.
        '''
        
        args, kwargs = self.__checkArgs__(args, kwargs)
        return np.sum([i.profile(*args[pos], **kwargs[pos]) for pos, i in enumerate(self.models)], axis=0)
    
    
    ################################################################
    #                          Velocities                          #
    ################################################################
    
    def meanV(self, D, R, offsetFactor=0):
        
        def correctDistance(s, D, R):
            '''Depending on the value of s, either return the radial distance for the foreground part or the background part'''
            
            dFor, dBac, theta = Vprojection(None).distance(s, D, R)
            
            if s**2 > D**2 + R**2:
                return dBac
            else:
                return dFor
        
        def numerator(s, D, R):
            '''Numerator to integrate'''
            
            velocity = self.vprojection(s, D, R)[0]
            
            return velocity*denominator(s, D, R)
        
        def denominator(s, D, R):
            '''Denominator to integrate'''
            
            distance     = correctDistance(s, D, R)
            density      = self.profile([distance]*len(self.models))
            
            return density
        
        if offsetFactor > 0:
            offsetFactor = 1
        if offsetFactor < 0:
            offsetFactor = -1
        
        res1, err1       = integrate.quad(numerator, 0, np.inf, args=(D, R))
        res2, err2       = integrate.quad(denominator, 0, np.inf, args=(D, R))
        
        cenVel           = abs(self.velocity([[0]]*len(self.models)))
        
        return (res1/res2 - offsetFactor*cenVel)/(1 + offsetFactor*cenVel/c.value)
            
    
    def velocity(self, args=[], kwargs=[{}]):
        '''
        Velocity profile for the 3D models against their own gravity through centripedal acceleration.

        Notes: 
            - Because models can require different arguments (for instance 3D radial distance r for one and 2D R in disk plane for the other), these are separated in lists.
              Each element will be passed to the corresponding model. The order is the same as in models list variable.
        
        Parameters
        ----------
            args : list 
                arguments to pass to each model. Order is that of models list.
            kwargs : list of dict
                kwargs to pass to each model. Order is that of models list.
        '''
        
        args, kwargs = self.__checkArgs__(args, kwargs)
        return np.sqrt(np.sum([i.velocity(*args[pos], **kwargs[pos])**2 for pos, i in enumerate(self.models)], axis=0))
    
    
    def vprojection(self, s, D, R, which='edge-on'):
        '''
        Apply a projection of the velocity along the line of sight depending on the geometry used.

        Mandatory parameters
        --------------------
            s : float/int or array of floats/int
                distance from the point to our location (same unit as L and D)
            D : float
                cosmological angular diameter distance between the galaxy centre and us
            R : float/int
                projected distance of the point along the major axis relative to the galaxy centre
                
        Optional parameters
        -------------------
            which : str
                which type of projection to do. Default is edge-on galaxy geometry.

        Return the projected line of sight velocity.
        '''
        
        # To overcome the issue with the args and kwargs definition we redefine the velocitys
        def vel(x):
            return self.velocity([x]*len(self.models))
        
        return Vprojection(vel).projection(s, D, R, which=which)


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
        
    def __add__(self, other):
        '''Add this instance with any other object. Only adding 3D models is allowed.'''
        
        return Multiple3DModels(self, other)
        
    
    def gfield(self, r, *args, **kwargs):
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
        
    
    def luminosity(self, r, *args, **kwargs):
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
        
        
    def profile(self, r, *args, **kwargs):
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
    
    
    def vprojection(self, s, D, R, which='edge-on'):
        '''
        Apply a projection of the velocity along the line of sight depending on the geometry used.

        Mandatory parameters
        --------------------
            s : float/int or array of floats/int
                distance from the point to our location (same unit as L and D)
            D : float
                cosmological angular diameter distance between the galaxy centre and us
            R : float/int
                projected distance of the point along the major axis relative to the galaxy centre
                
        Optional parameters
        -------------------
            which : str
                which type of projection to do. Default is edge-on galaxy geometry.

        Return the projected line of sight velocity.
        '''
        
        return Vprojection(self.velocity).projection(s, D, R, which=which)
    
    
    @property
    def toSersic(self):
        '''Create the best fit Sersic instance from this Hernquist instance.'''
        
        Re = self.Re**(1.0/0.6468628627045541) / 0.3514010020474344
        bn = compute_bn(4)
        Ie = self.M*bn**8 / (factorial(8)*np.pi*np.exp(bn)*Re**2)
        
        return Sersic(4, Re, Ie=Ie)
    
    
    def velocity(self, r, *args, **kwargs):
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
            
        if r==0:
            return np.sqrt(2*G*self.M/self.a).value
        else:
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
        
        self.n     = n
        self.Re    = Re
        self.bn    = compute_bn(self.n)
        self.Ie    = checkAndComputeIe(Ie, n, self.bn, Re, mag, offset)
        self._dim  = 2
        

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
        self._dim = 3
        
        
    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
        
    def __add__(self, other):
        '''Add this instance with any other object. Only adding 3D models is allowed.'''
        
        return Multiple3DModels(self, other)
        
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
    
    
    def vprojection(self, s, D, R, which='edge-on'):
        '''
        Apply a projection of the velocity along the line of sight depending on the geometry used.

        Mandatory parameters
        --------------------
            s : float/int or array of floats/int
                distance from the point to our location (same unit as L and D)
            D : float
                cosmological angular diameter distance between the galaxy centre and us
            R : float/int
                projected distance of the point along the major axis relative to the galaxy centre
                
        Optional parameters
        -------------------
            which : str
                which type of projection to do. Default is edge-on galaxy geometry.

        Return the projected line of sight velocity.
        '''
        
        return Vprojection(self.velocity).projection(s, D, R, which=which)
    
    
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
        
        if r==0:
            return 0
        else:
            return np.sqrt(2*G*self.luminosity(r)).value


