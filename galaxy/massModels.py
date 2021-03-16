    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:39:08 2020

@author: Mercier Wilfried - IRAP

Test functions on Galaxy mass modelling.
"""

import numpy                 as     np
import astropy.units         as     u
from   astropy.units.core    import UnitConversionError
from   .models               import checkAndComputeIe, sersic_profile
from   .misc                 import compute_bn, realGammainc
from   .kinematics           import Vprojection
from   astropy.constants     import G
from   scipy.special         import gamma, i0, i1, k0, k1

try:
    from   astropy.cosmology import Planck18 as cosmo
except ImportError:
    from   astropy.cosmology import Planck15 as cosmo
    

#######################################################################################
#                           3D profiles and their functions                           #
#######################################################################################

class MassModelBase:
    '''Base class for the mass models.'''
    
    def __init__(self, dim, M_L, unit_M_L='solMass.s.cm^2.A/erg)', **kwargs):
        '''
        Init function.

        Parameter
            ----------
            dim : int
                number of dimensions of the model
            M_L : float
                mass to light ratio. Refer to the specific mass model to know which unit to provide.
        '''
        
        if not isinstance(dim, int) or dim < 1:
            raise ValueError('Given dimension is not correct.')
            
        if M_L <= 0:
            raise ValueError('Mass to light ratio is negative or null.')
        
        if hasattr(M_L, 'unit'):
            if not isinstance(M_L.value, (int, float, np.float16, np.float32, np.float64)):
                raise ValueError('Mass to light ratio must either be an int, float, np.float16, np.float32 or np.float64.')
            else:
                self.M_L = M_L.to(unit_M_L)
                
        else:
            if not isinstance(M_L, (int, float, np.float16, np.float32, np.float64)):
                raise ValueError('Mass to light ratio must either be an int or a float.')
            else:
                self.M_L = u.Quantity(M_L, unit=unit_M_L)
        
        self._dim = dim
        
        
    def __add__(self, other):
        '''Add this instance with any other object. Only adding 3D models is allowed.'''
        
        return Multiple3DModels(self, other)
    
    
    def _checkR(self, r, against):
        '''Check whether radial distance is positive and has a unit.'''
        
        if r<0:
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        if not hasattr(r, 'unit'):
            r *= against.unit

        return r
    
    
    ###########################################################
    #       Default methods (some need to be overriden)       #
    ###########################################################
    
    def flux(self, r, *args, **kwargs):
        '''
        Compute the enclosed flux at radius r.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where to compute the luminosity

        Return the flux enclosed in a sphere of radius r as an astropy Quantity object.
        '''
        
        raise NotImplementedError('flux method not implemented yet.')
        return
    
    
    def gfield(self, r, *args, **kwargs):
        '''
        Compute the gravitational field of a single Hernquist profile at radius r.
        
        WARNING
        -------
            UNIT OF r MUST BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where the gravitational field is computed

        Return g(r) as an Astropy Quantity object.
        '''
        
        raise NotImplementedError('gfield method not implemented yet.')
        return
    
    
    def mass(self, r, *args, **kwargs):
        '''
        Compute the enclosed mass at radius r.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where to compute the luminosity

        Return the mass enclosed in a sphere of radius r as an astropy Quantity object.
        '''
        
        return self.M_L*self.flux(r, *args, **kwargs)
    
    
    def mass_profile(self, r, *args, **kwargs):
        '''
        Compute the mass density profile at radius r.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where to compute the profile

        Return the mass density profile at radius r as an Astropy Quantity object.
        '''
        
        return self.M_L*self.profile(r, *args, **kwargs)
    
    
    def profile(self, r, *args, **kwargs):
        '''
        Compute the light profile at radius r.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where to compute the profile

        Return the light profile at radius r as an Astropy Quantity object.
        '''
        
        raise NotImplementedError('profile method not implemented yet.')
        return
    
    
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

        Return V(r).
        '''

        return np.sqrt(G*self.mass(r)/r)
    
    
    #######################################################
    #           Projection methods (not tested)           #
    #######################################################
    
    def vprojection(self, s, D, R, which='edge-on', **kwargs):
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
    
    
    ################################
    #          Properties          #
    ################################
    
    @property
    def Ftot(self):
        '''Total flux of the profile.'''
        
        raise NotImplementedError('Ftot method not implemented yet.')
        return
    
    @property
    def Mtot(self):
        '''Total mass of the profile.'''
        
        return self.M_L*self.Ftot


class Multiple3DModels(MassModelBase):
    '''A master class used when combining two 3D models into a single object'''
    
    def __init__(self, model1, model2, *args, **kwargs):
        '''
        Init function
        
        Parameters
        ----------
            model1 : MassModelBase instance with _ndim = 3
                first model
            model2 : MassModelBase instance with _ndim = 3
                second model
        '''
        
        if not hasattr(model1, '_dim') or not hasattr(model2, '_dim'):
            raise AttributeError('One of the objects is not a valid model instance.')
        
        if model1._dim < 3 or model2._dim < 3:
            raise ValueError('One of the models is not 3D or higher. Only 3D models can be combined. Cheers !')
        
        super().__init__(3)
            
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
        
        
        
    def flux(self, args=[], kwargs=[{}]):
        '''
        Compute the enclosed flux at radius r.
        
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
        return np.sum([i.mass(*args[pos], **kwargs[pos]) for pos, i in enumerate(self.models)], axis=0)
        
    
    def mass(self, args=[], kwargs=[{}]):
        '''
        Compute the enclosed mass at radius r.
        
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
        return np.sum([i.mass(*args[pos], **kwargs[pos]) for pos, i in enumerate(self.models)], axis=0)
    
    
    def mass_profile(self, args=[], kwargs=[{}]):
        '''
        Compute the mass profile at radius r.

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
        return np.sum([i.mass_profile(*args[pos], **kwargs[pos]) for pos, i in enumerate(self.models)], axis=0)
    
            
    def profile(self, args=[], kwargs=[{}]):
        '''
        Compute the light profile at radius r.

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
    
    
    ##############################
    #         Properties         #
    ##############################
    
    @property
    def Ftot(self):
        '''Total flux of the profiles.'''
        
        return np.sum([i.Ftot for i in self.models], axis=0)
    
    @property
    def Mtot(self):
        '''Total mass of the profiles.'''
        
        return np.sum([i.Mtot for i in self.models], axis=0)


class Hernquist(MassModelBase):
    '''3D Hernquist model class with useful methods.'''
    
    def __init__(self, a, F, M_L, unit_a='kpc', unit_F='erg/(s.A)', unit_M_L='solMass.s.A.cm^2/(erg.kpc^2)', **kwargs):
        '''
        Mandatory parameters
        --------------------
            a : float or int
                scale factor
            F : float or int
               amplitude parameter (total flux)
            M_L : float or int
                mass to light ratio.
               
        Optional parameters
        -------------------
            unit_a : str
                unit desired for the scale parameter a. This will be used when computing a value in a certain unit. Default is kpc.
            unit_F : str
                unit desired for the amplitude parameter M. Default is 'erg/s/A'.
            unit_M_L : str
                unit used to convert from light to mass profiles. Default is in Solar masses per unit of F, i.e. solMass/(erg/(s.cm^2.A)).
        '''
        
        super().__init__(3, M_L, unit_M_L=unit_M_L)
        
        # If parameters are already passed as Astropy Quantities, just convert them
        if hasattr(a, 'unit'):
            self.a  = a.to(unit_a)
        else:
            self.a  = u.Quantity(a, unit_a)
            
        if hasattr(F, 'unit'):
            self.F  = F.to(unit_F)
        else:
            self.F  = u.Quantity(F, unit_F)
            
        if not isinstance(self.a.value, (int, float, np.float16, np.float32, np.float64)) or not isinstance(self.F.value, (int, float, np.float16, np.float32, np.float64)):
            raise TypeError('One of the parameters is not an int, float, np.float16, np.float32 or np.float64.')
            
        try:
            self.Vmax = 0.5*np.sqrt(G*self.M_L*self.F/self.a).to('km/s')
        except UnitConversionError:
            raise UnitConversionError('The unit of Vmax (%s) could not be converted to km/s. Please check carefully the units of F, a and M_L parameters. Cheers !' %(np.sqrt(G*self.M_L*self.F/self.a).unit))
        
        # Other hidden properties
        self._alpha_a = -0.454
        self._beta_a  = 0.725
        self._alpha_F = 1.194
        self._beta_F  = 1.75
        
        
    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################    
    
    def flux(self, r, *args, **kwargs):
        '''
        Compute the enclosed flux at radius r.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where to compute the luminosity

        Return the flux enclosed in a sphere of radius r as an astropy Quantity object.
        '''
        
        r = self._checkR(r, self.a)
        return self.F*(r/(r+self.a))**2
    
    
    def gfield(self, r, *args, **kwargs):
        '''
        Compute the gravitational field of a single Hernquist profile at radius r.
        
        WARNING
        -------
            UNIT OF r MUST BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where the gravitational field is computed

        Return g(r) as an Astropy Quantity object.
        '''
        
        r = self._checkR(r, self.a)
            
        if r.value == 0:
            return -G*self.F/self.a**2
        else:   
            return -G*self.flux(r)*self.M_L/r**2
        
        
    def profile(self, r, *args, **kwargs):
        '''
        Compute the light profile at radius r.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where to compute the profile

        Return the light profile at radius r as an Astropy Quantity object.
        '''
        
        r = self._checkR(r, self.a)
        return self.F*self.a/(2*np.pi) / (r * (r + self.a)**3)
    
    
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

        Return V(r) in units of G*M/L*F/sqrt(a).
        '''

        r = self._checkR(r, self.a)
        
        if r.value==0:
            return 0
        else:
            return np.sqrt(G*self.M_L*self.F*r)/(self.a+r)
    
    
    ################################
    #          Properties          #
    ################################
    
    @property
    def Ftot(self):
        '''Total flux of the profile.'''
        
        return self.F
    
    @property
    def todeVaucouleur(self):
        '''Similar to toSersic method.'''
        
        return self.toSersic
    
    @property
    def toSersic(self):
        '''Create the best fit Sersic instance from this Hernquist instance.'''
        
        Re = ((self.a.value/10**self._alpha_a)**(1.0/self._beta_a)) * self.a.unit
        Ie = (self.F.value / (10**self._alpha_F * Re.value**self._beta_F)) * (self.F.unit/(Re.unit**2))
        
        return deVaucouleur(Re, Ie=Ie, unit_Re=str(Re.unit), unit_Ie=str(Ie.unit))
        
        
class NFW(MassModelBase):
    '''3D NFW profile.'''
    
    def __init__(self, Rs, c=None, Vmax=None, unit_Rs='kpc', unit_Vmax='km/s'):
        '''
        Initialise NFW profile. Two pairs of parameters can be passed. Either Rs and c, or Rs and Vmax.

        Parameters
        ----------
            Rs : int/float or astropy Quantity with distance unit
                scale parameter
            
        Optional parameters
        -------------------
            c : int/float
                concentration parameter. If None, Vmax must be given.
            Vmax : int/float or astropy Quantity with velocity unit
                maximum circular velocity at 2.15*Rs. If None, c must be given. If not given as an astropy Quantity, provide the correct unit with unit_Vmax.
            unit_Rs : str
                unit of Rs. Default is kpc.
            unit_Vmax : str
                unit of Vmax. Defauly is km/s.
        '''
        
        self._factor      = np.log(3.15)/2.15 - 1/3.15
        
        if (Vmax is None and c is None) or (Vmax is not None and c is not None):
            raise ValueError('Either Vmax or c must be None.')
        
        if hasattr(Rs, 'unit'):
            self.Rs       = Rs.to(unit_Rs)
        else:
            self.Rs       = u.Quantity(Rs, unit_Rs)
            
        if Vmax is None:
            
            if hasattr(c, 'unit') and c.unit != u.dimensionless_unscaled:
                raise TypeError('concentration parameter must have no unit.')
            else:
                self.c    = c
                
            self.delta_c  = 200*self.c**3 / (3*(np.log(1+self.c) - self.c/(1+self.c)))
            
            try:
                self.Vmax = (2*self.Rs * np.sqrt(np.pi*G*self.delta_c*cosmo.critical_density(0)*self._factor)).to('km/s')
            except UnitConversionError:
                raise UnitConversionError('The unit of Vmax (%s) could not be converted to km/s. Please check carefully the units of Rs. Cheers !' %(self.Rs*np.sqrt(G*self.delta_c*cosmo.critical_density(0))).unit)
            
        else:
            self.c        = c
            
            if hasattr(Vmax, 'unit'):
                self.Vmax = self.Vmax.to(unit_Vmax) 
            else:
                self.Vmax = u.Quantity(Vmax, unit_Vmax)
            
            try:
                self.delta_c  = (self.Vmax**2/(4*np.pi*G*self._factor*cosmo.critical_density(0)*self.Rs**2)).to('')
            except UnitConversionError:
                raise UnitConversionError('delta_c parameter could not be computed as a dimensionless quantity.')
            
        super().__init__(3, 1, unit_M_L='solMass.s.A.cm^2/(erg.kpc^2)')
      
        
    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
    
    def flux(self, r, *args, **kwargs):
        '''
        Compute the enclosed flux at radius r (0 since Dark Matter).
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where to compute the luminosity

        Return the flux enclosed in a sphere of radius r as an astropy Quantity object.
        '''
        
        r = self._checkR(r, self.Rs).value
        return u.Quantity(0*r, unit='erg/(s.A)')
    
    
    def gfield(self, r, *args, **kwargs):
        '''
        Compute the gravitational field of a single NFW profile at radius r.
        
        WARNING
        -------
            UNIT OF r MUST BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where the gravitational field is computed

        Return g(r) as an Astropy Quantity object.
        '''
        
        raise NotImplementedError('gfield not implemented for a NFW profile')
        return
    
    
    def mass(self, r, *args, **kwargs):
        '''
        Compute the mass profile at radius r.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER a.

        Parameters
        ----------
            r : float or int
                radius where to compute the profile

        Return the mass profile at radius r as an Astropy Quantity object.
        '''
        
        r = self._checkR(r, self.Rs)
        return r*self.velocity(r, *args, **kwargs)**2 / G

    
    def mass_profile(self, r, *args, **kwargs):
        '''
        Compute the mass density profile at radius r.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER Rs.

        Parameters
        ----------
            r : float or int
                radius where to compute the profile

        Return the mass density profile at radius r as an Astropy Quantity object.
        '''
        
        r = self._checkR(r, self.Rs)
        return self.Vmax**2 / (4*np.pi*G*self.Rs*self._factor * r*(1+r/self.Rs)**2)
    
    
    def profile(self, r, *args, **kwargs):
        '''
        Compute the light profile at radius r (0 since Dark Matter).
        
        Parameters
        ----------
            r : float or int
                radius where to compute the profile
        '''
        
        r = self._checkR(r, self.Rs).value
        return u.Quantity(0*r, unit='erg/(s.A.cm^2)')
    
    
    def velocity(self, r, *args, **kwargs):
        '''
        Velocity at radius r.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER Rs.

        Parameters
        ----------
            r : float or int
                radius where the profile is computed

        Return the velocity computed at radius r.
        '''
        
        r = self._checkR(r, self.Rs)
        
        if r.value==0:
            return 0 * self.Vmax.unit
        else:
            return self.Vmax * np.sqrt(self.Rs * (np.log(1+r/self.Rs)/r - 1/(r+self.Rs)) / self._factor)
    
    ################################
    #          Properties          #
    ################################
    
    @property
    def Ftot(self):
        '''Total flux (infinite).'''
        
        return np.inf
    
    @property
    def Mvir(self):
        '''Virial mass.'''
        
        return (800*np.pi*cosmo.critical_density(0)*self.Rvir**3)/3

    @property    
    def Rvir(self):
        '''Virial radius.'''
        
        if self.c is None:
            raise ValueError('Virial radius cannot be computed if the concentration parameter is not provided.')
        else:
            return self.c*self.Rs
        

#################################################################################
#                                Sersic profiles                                #
#################################################################################

class Sersic:
    '''2D Sersic profile class.'''
    
    def __init__(self, n, Re, Ie=None, mag=None, offset=None, unit_Re='kpc', unit_Ie='erg/(cm^2.s.A)', **kwargs):
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
                unit for the surface brightness. Default is erg/cm^2/s/A. If Ie already has a unit, it is converted to this unit.
            unit_Re : str
                unit for the scale radius Re. Default is kpc. If Re already has a unit, it is converted to this unit.
        """
        
        if not isinstance(n, (int, float, np.float16, np.float32, np.float64)):
            raise TypeError('n is not an int, float, np.float16, np.float32 or np.float64.')
            
        if not isinstance(Re, (int, float, np.float16, np.float32, np.float64, u.quantity.Quantity)):
            raise TypeError('Re is not an int, float, np.float16, np.float32 or np.float64.')
            
        if Ie is None and (mag is None or offset is None):
            raise ValueError('Ie is None but no magnitude and no offset is given.')
            
        if n<=0:
            raise ValueError('n must be positive valued. Cheers !')
            
        if Re<=0:
            raise ValueError('The effective radius Re must be positive valued. Cheers !')
            
        if Ie is not None and Ie<=0:
            raise ValueError('The effective radius Re must be positive valued. Cheers !')
        
        self._dim     = 2
        self.n        = n
        self.bn       = compute_bn(self.n)
        
        # If the parameters already are Astropy quantities, we just convert them to the required unit
        if hasattr(Re, 'unit'):
            self.Re   = Re.to(unit_Re)
        else:
            self.Re   = u.Quantity(Re, unit=unit_Re)
            
        if hasattr(Ie, 'unit'):
            self.Ie   = Ie.to(unit_Ie)
        else:
            self.Ie   = u.Quantity(checkAndComputeIe(Ie, self.n, self.bn, self.Re.value, mag, offset), unit=unit_Ie)
            
        # Central intensity
        self.I0       = self.Ie*np.exp(self.bn)
        

    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
    
    def _checkR(self, r, against):
        '''Check whether radial distance is positive and has a unit.'''
        
        if r<0:
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        if not hasattr(r, 'unit'):
            r *= against.unit

        return r
        
    def profile(self, r, *args, **kwargs):
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
        
        r = self._checkR(r, self.Re)
        return sersic_profile(r, self.n, self.Re, Ie=self.Ie, bn=self.bn)
    
    
    def flux(self, r, *args, **kwargs):
        '''
        Flux at radius r (encompassed within a disk since this is a 2D profile).
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER Re.

        Parameters
        ----------
            r : float or int
                radius where the profile is computed

        Return the flux computed within a disk of radius r.
        '''
        
        r  = self._checkR(r, self.Re)
        n2 = 2*self.n
        return n2*np.pi*self.Ie*np.exp(self.bn)*realGammainc(n2, (self.bn*(r/self.Re)**(1.0/self.n)).value)*self.Re**2 / self.bn**n2
    
    @property
    def Ftot(self):
        '''Total flux of the profile.'''
        
        n2 = 2*self.n
        return n2*np.pi*self.Ie*np.exp(self.bn)*gamma(n2)*self.Re**2 / self.bn**n2
    
    
class deVaucouleur(Sersic):
    '''2D de Vaucouleur profile.'''
    
    def __init__(self, Re, Ie=None, mag=None, offset=None, unit_Re='kpc', unit_Ie='erg/(cm^2.s.A)', **kwargs):
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
                unit for the surface brightness. Default is 'erg/(cm^2.s.A)'.
            unit_Re : str
                unit for the scale radius Re. Default is kpc.
        """
        
        super().__init__(4, Re, Ie=Ie, mag=mag, offset=offset, unit_Re=unit_Re, unit_Ie=unit_Ie, **kwargs)
        
        # Other hidden properties
        self._alpha_a = -0.454
        self._beta_a  = 0.725
        self._alpha_F = 1.194
        self._beta_F  = 1.75
        
        
    def toHernquist(self, M_L, unit_M_L='solMass.s.A.cm^2/(erg.kpc^2)', **kwargs):
        '''
        Create the best-fit Hernquist instance from this de Vaucouleur instance.
        
        Parameters
        ----------
            M_L : float/int
                mass to light ratio
            unit_M_L : str
                unit of the mass to light ratio. If M_L already has a unit, it is converted to this unit.
        '''
        
        if not isinstance(M_L, (int, float, np.float16, np.float32, np.float64, u.quantity.Quantity)):
            raise TypeError('Mass to light ratio must either be an int, float, np.float16, np.float32 or np.float64.')
            
        if hasattr(M_L, 'unit'):
            M_L = M_L.to(unit_M_L)
        else:
            M_L = u.Quantity(M_L, unit=unit_M_L)
    
        a       = 10**self._alpha_a * self.Re.value**self._beta_a * self.Re.unit
        F       = self.Ie.value * 10**self._alpha_F * self.Re.value**self._beta_F * (self.Ie.unit*self.Re.unit**2)
        
        return Hernquist(a, F, M_L, unit_a=str(a.unit), unit_F=str(F.unit), unit_M_L=str(M_L.unit))
    

class ExponentialDisk(Sersic, MassModelBase):
    '''
    2D/3D Exponential disk profile class.
    Some functions such as the light profile correspond to the 2D Sersic profiles, while others such as the velocity assume a 3D infinitely thin disk.
    '''
    
    def __init__(self, Re, M_L, Ie=None, mag=None, offset=None, unit_Re='kpc', unit_Ie='erg/(cm^2.s.A)', unit_M_L='solMass.s.A.cm^2/(erg.kpc^2)', **kwargs):        
        """
        You can either provide n, Re and Ie, or instead n, Re, mag and offset.
        If neither Ie, nor mag and offset are provided a ValueError is raised.  
        
        Mandatory parameters
        --------------------
            M_L : int/float
                mass to light ratio
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
                unit for the surface brightness. Default is 'erg/(cm^2.s.A)'.
            unit_M_L : str
                unit of the mass to light ratio. Default is 'solMass.s.A.cm^2/(erg.kpc^2)'.
            unit_Re : str
                unit for the scale radius Re. Default is kpc.
        """
        
        MassModelBase.__init__(self, 3, M_L, unit_M_L)
        Sersic.__init__(       self, 1, Re, Ie=Ie, mag=mag, offset=offset, unit_Re=unit_Re, unit_Ie=unit_Ie)
        
        # Disk scale length
        self.Rd       = self.Re/self.bn
        
        # Position of maximum velocity
        self.Rmax     = 2.15*self.Rd
        
        try:
            self.Vmax = 0.8798243*np.sqrt(np.pi*G*self.Rd*self.M_L*self.I0).to('km/s')
        except UnitConversionError:
            raise UnitConversionError('The unit of Vmax (%s) could not be converted to km/s. Please check carefully the units of Ie, Re and M_L parameters. Cheers !' %np.sqrt(np.pi*G*self.Rd*self.M_L*self.Ie*np.exp(self.bn)).unit)
        
        
    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
    
    def velocity(self, r, *args, **kwargs):
        '''
        Velocity profile for a self-sustaining 3D inifinitely thin disk against its own gravity through centripedal acceleration.
        
        WARNING
        -------
            UNIT OF r MUSE BE IDENTICAL TO THE SCALE PARAMETER Re.

        Parameters
        ----------
            r : float/int or numpy array of float/int
                radius where the velocity profile is computed

        Return V(r).
        '''
        
        r = self._checkR(r, self.Re)
        
        if r.value == 0:
            return 0
        else:
            y = r/(2*self.Rd)
            return (self.Vmax*r/(0.8798243*self.Rd)) * np.sqrt(i0(y)*k0(y) - i1(y)*k1(y))
        
class DoubleExponentialDisk(Sersic, MassModelBase):
    '''
    Double exponential disk profile class:
    
    rho(R, z) = \Sigma_0 / (2 hz) e^{-R/Re - |z|/hz}
    
    with \Sigma_0 = Ie e^{b_1} the central surface density, Re the disk effective radius in the plane of symmetry and hz the vertical scale height.
    '''
    
    def __init__(self, Re, hz, M_L, q0=None, Ie=None, mag=None, offset=None, 
                 unit_Re='kpc', unit_hz='kpc', unit_Ie='erg/(cm^2.s.A)', unit_M_L='solMass.s.A.cm^2/(erg.kpc^2)', **kwargs):        
        """
        You can either provide n, Re, hz and Ie, or instead n, Re, hz, mag and offset.
        If neither Ie, nor mag and offset are provided a ValueError is raised.  
        
        Mandatory parameters
        --------------------
            hz : int.float
                vertical scale height
            M_L : int/float
                mass to light ratio
            re : int/float
                half-light radius
                
        Optional parameters
        -------------------
            Ie : float
                intensity at half-light radius. Default is None.
            mag : float
                galaxy total integrated magnitude used to compute Ie if not given. Default is None.
            offset : float
                magnitude offset in the magnitude system used. Default is None.
            q0 : float between 0 and 1
                axis ratio equal to hz/Rd with Rd the disk scale length. If this value is different from None, it overrides the hz value. Default is None.
            unit_Ie : str
                unit for the surface brightness. Default is 'erg/(cm^2.s.A)'.
            unit_M_L : str
                unit of the mass to light ratio. Default is 'solMass.s.A.cm^2/(erg.kpc^2)'.
            unit_Re : str
                unit for the scale radius Re. Default is kpc.
        """
        
        MassModelBase.__init__(self, 3, M_L, unit_M_L)
        Sersic.__init__(       self, 1, Re, Ie=Ie, mag=mag, offset=offset, unit_Re=unit_Re, unit_Ie=unit_Ie)
        
        # Disk scale length
        self.Rd       = self.Re/self.bn
        
        if q0 is not None:
            if q0 >= 0 and q0 <= 1:
                self.q0 = q0
                self.hz = self.q0*self.Rd
            else:
                raise ValueError('q0 is equal to %s, but it must be between 0 and 1.' %q0)
        else:
            self.hz     = hz
            self.q0     = self.hz/self.Rd
        
        
    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
    
    @property
    def _Rmax_razor_thin(self):
        '''Position of the maximum of the razor-thin disk curve'''
        
        return 2.2*self.Rd
    
    def _velocity_correction(self, R, *args, **kwargs):
        '''
        Velocity correction necessary for the rotation curve.
        
        WARNING
        -------
            UNIT OF R MUST BE IDENTICAL TO THE SCALE PARAMETER Re.

        Parameters
        ----------
            R : float or numpy array of float
                radius where the velocity correction is computed

        Return the velocity correction.
        '''
        
        R = self._checkR(R, self.Re)
        return self._Vmax_correction * np.sqrt(R * np.exp(1-R/self.Rd) / self.Rd)
    
    
    def _velocity_razor_thin(self, R, *args, **kwargs):
        '''
        Velocity if the disk was razor thin.
        
        WARNING
        -------
            UNIT OF R MUSE BE IDENTICAL TO THE SCALE PARAMETER Re.

        Parameters
        ----------
            R : float or numpy array of float
                radius where the velocity is computed

        Return the velocity if the disk was razor thin.
        '''
        
        R       = self._checkR(R, self.Re)
        disk_RT = ExponentialDisk(self.Re, self.M_L, Ie=self.Ie)
        return disk_RT.velocity(R)
        
    
    @property
    def _Vmax_correction(self):
        '''Maximum of the velocity correction.'''
        
        return np.sqrt(2*np.pi*G*self.hz*self.M_L*self.I0/np.exp(1))
    
    
    @property
    def _Vmax_razor_thin(self):
        '''Maximum velocity if the disk was razor thin.'''
        
        disk_RT = ExponentialDisk(self.Re, self.M_L, Ie=self.Ie)
        return disk_RT.Vmax
        
        
    def profile(self, R, z, *args, **kwargs):
        '''
        Sersic profile at radius r.
        
        WARNING
        -------
            UNIT OF R MUST BE GIVEN, OTHERWISE IT IS ASSUMED IDENTICAL TO THE SCALE PARAMETER Re.
            UNIT OF R MUST BE GIVEN, OTHERWISE IT IS ASSUMED IDENTICAL TO THE SCALE PARAMETER hz.

        Parameters
        ----------
            R : int/float
                radial position in the plane of symmetry
            z : int/float
                vertical position

        Return the double exponential profile computed at position (R, z).
        '''
        
        R = self._checkR(R, self.Re)
        z = self._checkR(z, self.hz)
        return sersic_profile(R, self.n, self.Re, Ie=self.Ie, bn=self.bn) * np.exp(-np.abs(z)/self.hz) / (2*self.hz)
    
    
    def velocity(self, R, *args, **kwargs):
        '''
        Velocity profile for a self-sustaining 3D double exponential disk against its own gravity through centripedal acceleration.
        This formula is an approximation for a thin (but non-zero thickness) disk.
        
        WARNING
        -------
            UNIT OF R MUSE BE IDENTICAL TO THE SCALE PARAMETER Re.

        Parameters
        ----------
            R : float/int or numpy array of float/int
                radius where the velocity profile is computed

        Return V(R, z=0).
        '''
        
        R = self._checkR(R, self.Re)
    
        v_RT2   = self.velocity_razor_thin(R)**2
        v_corr2 = self.velocity_correction(R)**2
        
        if isinstance(R.value, (int, float, np.float16, np.float32, np.float64)):
            if v_RT2 < v_corr2:
                vr = 0
            else:
                vr = np.sqrt(v_RT2 - v_corr2)
            
        elif isinstance(R.value, np.ndarray):
            vr       = np.zeros(np.shape(R))
            mask     = v_corr2 < v_RT2
            vr[mask] = np.sqrt(v_RT2 - v_corr2)
        
        else:
            raise TypeError('R array has type %s but only float and numpy array are expected.' %type(R))
        
        return vr
            
    