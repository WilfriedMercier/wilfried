    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:39:08 2020

@author: Mercier Wilfried - IRAP

Test functions on Galaxy mass modelling.
"""

import scipy.integrate   as     integrate
import numpy             as     np
import astropy.units     as     u
from   .models           import compute_bn, checkAndComputeIe, sersic_profile
from   .kinematics       import Vprojection
from   scipy.special     import gamma, gammainc
from   math              import factorial
from   astropy.constants import G


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
    
    def meanV(self, D, R):
        '''
        Compute the mean velocity along the line of sight.
        
        Warning
        -------
            BE SURE THAT THE MODELS ALL HAVE UNITS WHICH ARE SIMILAR SO THAT ADDIND VELOCITIES^2 IS DONE WITH THE SAME VELOCITY UNITS.

        Parameters
        ----------
            D : TYPE
                DESCRIPTION.
            R : TYPE
                DESCRIPTION.

        Return the compute velocity value (unit will be the same as the velocities computed for each model).
        '''
        
        def correctDistance(s, D, R):
            '''Depending on the value of s, either return the radial distance for the foreground part or the background part'''
            
            dFor, dBac, theta = Vprojection(None).distance(s, D, R)
            
            if s**2 > D**2 + R**2:
                return dBac
            else:
                return dFor
        
        def numerator(s, D, R):
            '''Numerator to integrate'''
            
            velocity     = self.vprojection(s, D, R)[0]
            
            return velocity*denominator(s, D, R)
        
        def denominator(s, D, R):
            '''Denominator to integrate'''
            
            distance     = correctDistance(s, D, R)
            density      = self.profile([distance]*len(self.models))
            
            return density
        
        res1, err1       = integrate.quad(numerator, 0, np.inf, args=(D, R))
        res2, err2       = integrate.quad(denominator, 0, np.inf, args=(D, R))
        
        return res1/res2
            
    
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
    
    def __init__(self, a, M, unit_a='kpc', unit_M='erg/cm^2/s/Hz'):
        '''
        Mandatory parameters
        --------------------
            a : float or int
                scale factor
            M : float or int
               total "mass" (can also be understood as luminosity)
               
        Optional parameters
        -------------------
            unit_a : str
                unit desired for the scale parameter a. This will be used when computing a value in a certain unit. Default is kpc.
            unit_M : str
                unit desired for the amplitude parameter M. Default is 
        '''
        
        # If parameters are already passed as Astropy Quantities, just convert them
        if hasattr(a, 'unit'):
            self.a  = a.to(unit_a)
        else:
            self.a  = u.Quantity(a, unit_a)
            
        if hasattr(M, 'unit'):
            self.M  = M.to(unit_M)
        else:
            self.M  = u.Quantity(M, unit_M)
            
        if not isinstance(self.a.value, (int, float)) or not isinstance(self.M.value, (int, float)):
            raise TypeError('One of the parameters is neither int nor float.')
        
        # Other hidden properties
        self._dim   = 3
        self._beta  = 0.6468628627045541 # unitless (powerlaw)
        self._alpha = u.Quantity(0.3514010020474344 * 10**(1-self._beta), unit='kpc(%s)' %(1-self._beta))
        
        
    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
        
    def __add__(self, other):
        '''Add this instance with any other object. Only adding 3D models is allowed.'''
        
        return Multiple3DModels(self, other)
    
    @property
    def __units__(self):
        '''Print on screen the units of the parameters and of the available functions.'''
        
        print('Model: Hernquist, units\n\
              Parameters:\n\
                  a : %s\n\
                  M : %s\n\n\
              Functions:\n\
                  gfield     : %s\n\
                  luminosity : %s\n\
                  profile    : %s\n\
                  velocity   : %s\n\
                  vprojection: %s\n' %(self.a.unit, self.M.unit, self.gfield(0).unit, self.luminosity(0).unit, 
                                       self.profile(0).unit, self.velocity(0).unit, self.vprojection(10, 10, 0).unit))
        
    def gfield(self, r, *args, **kwargs):
        '''
        Compute the gravitational field of a single Hernquist profile at radius r.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where the gravitational field is computed

        Return g(r) as an Astropy Quantity object.
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        # If r is without unit, we assume it is the zame as the scale parameter
        if not hasattr(r, 'unit'):
            r *= self.a.unit
        
        return -G*self.luminosity(r)/r**2
        
    
    def luminosity(self, r, *args, **kwargs):
        '''
        Compute the enclosed luminosity or mass at radius r.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where to compute the luminosity

        Return the luminosity or mass enclosed in a sphere of radius r as an astropy Quantity object.
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        # If r is without unit, we assume it is the zame as the scale parameter
        if not hasattr(r, 'unit'):
            r *= self.a.unit
        
        return self.M*(1 - 1.0/(1+r/self.a)**2)
        
        
    def profile(self, r, *args, **kwargs):
        '''
        Compute the light or mass profile at radius r.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where to compute the profile

        Return the light or mass profile at radius r as an Astropy Quantity object.
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        # If r is without unit, we assume it is the zame as the scale parameter
        if not hasattr(r, 'unit'):
            r *= self.a.unit
        
        return self.M/(2*np.pi*self.a**2) / (r * (1+r/self.a)**3)
    
    
    def vprojection(self, s, D, R, which='edge-on'):
        '''
        Apply a projection of the velocity along the line of sight depending on the geometry used.
        
        WARNING
        -------
            UNIT OF s, D and R MUSE BE IDENTICAL TO THE SCALE PARAMETER a.

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
        
        # If s, D or R are without unit, we assume it is the zame as the scale parameter
        if not hasattr(R, 'unit'):
            R *= self.a.unit
            
        if not hasattr(s, 'unit'):
            s *= self.a.unit
            
        if not hasattr(D, 'unit'):
            D *= self.a.unit
        
        return Vprojection(self.velocity).projection(s, D, R, which=which)
    
    
    @property
    def toSersic(self):
        '''Create the best fit Sersic instance from this Hernquist instance.'''
        
        Re = (self.a/self._alpha)**(1.0/self._beta)
        bn = compute_bn(4)
        Ie = self.M*bn**8 / (factorial(8)*np.pi*np.exp(bn)*Re**2)
        
        return Sersic(4, Re, Ie=Ie)
    
    
    def velocity(self, r, *args, **kwargs):
        '''
        Velocity profile for a self-sustaining Hernquist 3D profile against its own gravity through centripedal acceleration.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float/int or numpy array of float/int
                radius where the velocity profile is computed

        Return V(r) in units of:
            - (m^3 "M")/("a" kg s^2)
            
        with "M" and "a" the units of the corresponding parameters passed as input.
        '''
        
        if not hasattr(r, 'unit'):
            r *= self.a.unit
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        try:
            if r.value==0:
                return np.sqrt(2*G*self.M/self.a)
            else:
                return np.sqrt(G*self.luminosity(r)/r)
        except:
            vel0            = np.sqrt(2*G*self.M/self.a)
            velocity        = r.value * vel0.unit
            mask            = r.value==0
            velocity[mask]  = np.sqrt(2*G*self.M/self.a)
            velocity[~mask] = np.sqrt(G*self.luminosity(r)/r)
            return velocity


class Sersic:
    '''
    2D Sersic profile class.
    
    This class combines into a single objects useful functions related to Sersic profiles
    '''
    
    def __init__(self, n, Re, Ie=None, mag=None, offset=None, unit_Re='kpc', unit_Ie='erg/cm^2/s/Hz/kpc^2'):
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
            unit_Ie : str
                unit for the surface brightness. Default is similar to specfic flux/angular surface
            unit_Re : str
                unit for the scale radius Re. Default is kpc.
        """
        
        if not isinstance(n, (float, int)) or not isinstance(Re, (float, int)):
            raise TypeError('Either n or Re is neither int nor float.')
            
        if n<0:
            raise ValueError('n must be positive valued. Cheers !')
            
        if Re<0:
            raise ValueError('The effective radius Re must be positive valued. Cheers !')
            
        self.n      = n
        self.bn     = compute_bn(self.n)
        
        # If the parameters already are Astropy quantities, we just convert them to the required unit
        if hasattr(Re, 'unit'):
            self.Re = self.Re.to(unit_Re)
        else:
            self.Re = u.Quantity(Re, unit=unit_Re)
            
        if hasattr(Ie, 'unit'):
            self.Ie = self.Ie.to(unit_Ie)
        else:
            self.Ie = u.Quantity(checkAndComputeIe(Ie, n, self.bn, Re, mag, offset), unit=unit_Ie)
            
        # Other hidden properties
        self._dim   = 2
        self._beta  = 0.6468628627045541 # unitless (powerlaw)
        self._alpha = u.Quantity(0.3514010020474344 * 10**(1-self._beta), unit='kpc(%s)' %(1-self._beta))
        

    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
        
    def profile(self, r):
        '''
        Sersic profile at radius r.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER Re.

        Parameters
        ----------
            r : float or int
                radius where the profile is computed

        Return the Sersic profiles computed at the radius r.
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        if not hasattr(r, 'unit'):
            r *= self.Re.unit
        
        return sersic_profile(r, self.n, self.Re, Ie=self.Ie, bn=self.bn)
    
    
    def luminosity(self, r):
        '''
        Luminosity at radius r (encompassed within a disk since this is a 2D profile).
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER Re.

        Parameters
        ----------
            r : float or int
                radius where the profile is computed

        Return the luminosity computed within a disk of radius r.
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        if not hasattr(r, 'unit'):
            r *= self.Re.unit
           
        n2 = 2*self.n
        return n2*np.pi*self.Ie*np.exp(self.bn)*gammainc( n2, (self.bn*(r/self.Re)**(1.0/self.n)).value )*self.Re**2 / self.bn**n2
    
    @property
    def Ltot(self):
        '''Total luminosity of the profile.'''
        
        n2 = 2*self.n
        return np.pi*n2*self.Ie*np.exp(self.bn)*gamma(n2)*self.Re**2 / self.bn**(n2)
    
    @property
    def toHernquist(self):
        '''Create the best-fit Hernquist instance from this Sersic instance (based on my own calculations).'''
        
        if self.n==4:
            a = self._alpha*self.Re**self._beta
            M = self.Ltot
        else:
            raise ValueError('Only de Vaucouleurs profiles, i.e. Sersic with n=4, can be converted into Hernquist profiles.')
        
        return Hernquist(a, M)
    
    
class ExponentialDisk(Sersic):
    '''
    2D/3D Exponential disk profile class.
    
    This class combine into a single objects useful functions related to Sersic profiles
    '''
    
    def __init__(self, Re, Ie=None, mag=None, offset=None, unit_Re='kpc', unit_Ie='erg/cm^2/s/Hz/arcsec^2'):        
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
            unit_Ie : str
                unit for the surface brightness. Default is similar to specfic flux/angular surface
            unit_Re : str
                unit for the scale radius Re. Default is kpc.
        """
        
        super().__init__(1, Re, Ie=Ie, mag=mag, offset=offset, unit_Re=unit_Re, unit_Ie=unit_Ie)
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
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER Re.

        Parameters
        ----------
            r : float or int
                radius where the gravitational field is computed

        Return g(r).
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        if not hasattr(r, 'unit'):
            r *= self.Re.unit
                
        return -2*G*self.luminosity(r)/r
    
    
    def vprojection(self, s, D, R, which='edge-on'):
        '''
        Apply a projection of the velocity along the line of sight depending on the geometry used.
        
        WARNING
        -------
            UNIT OF s, D and R MUSE BE IDENTICAL TO THE SCALE PARAMETER Re.

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
        
        # If s, D or R are without unit, we assume it is the zame as the scale parameter
        if not hasattr(R, 'unit'):
            R *= self.Re.unit
            
        if not hasattr(s, 'unit'):
            s *= self.Re.unit
            
        if not hasattr(D, 'unit'):
            D *= self.Re.unit
        
        return Vprojection(self.velocity).projection(s, D, R, which=which)
    
    
    def velocity(self, r):
        '''
        Velocity profile for a self-sustaining 3D inifinitely thin disk against its own gravity through centripedal acceleration.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER Re.

        Parameters
        ----------
            r : float/int or numpy array of float/int
                radius where the velocity profile is computed

        Return V(r) in units of:
            - (m^3 "Luminosity") / ("Re" kg s^2)
            
        where "Luminosity" = "Ie" "Re"^2 and "Re" are the units of the parameters passed as inputs.
        '''
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        if not hasattr(r, 'unit'):
            r *= self.Re.unit
        
        try:
            if r.value==0:
                return 0
            else:
                return np.sqrt(2*G*self.luminosity(r)).value
        except:
            vel0            = 0*np.sqrt(2*G*self.luminosity(1) / u.Quantity(1, unit=r.unit))
            velocity        = r.value * vel0.unit
            mask            = r.value==0
            velocity[mask]  = vel0
            velocity[~mask] = np.sqrt(2*G*self.luminosity(r) / u.Quantity(1, unit=r.unit))
            return velocity


