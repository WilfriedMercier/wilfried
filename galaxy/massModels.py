    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*Author:* Wilfried Mercier - IRAP

3D and 2D mass models for different mass and light profiles.
"""

import numpy                 as     np
import astropy.units         as     u
from   astropy.units.core    import UnitConversionError
from   .models               import checkAndComputeIe, sersic_profile
from   .misc                 import compute_bn, realGammainc
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
    '''
    Base class for mass models.
    
    .. warning::
    
        When using a method, make sure the unit of the radial distance is identical to that of the scale parameter of the model.
    '''
    
    def __init__(self, dim, M_L, unit_M_L='solMass.s.cm^2.A/erg)', **kwargs):
        '''
        Init function.
        
        .. note::
            
            Units must be given such that they are recognised by astropy.units module.

        :param int dim: number of dimensions of the model
        :param float M_L: mass to light ratio
        :param str unit_M_L: unit of the mass to light ratio. Refer to the specific mass model to know which unit to provide.
        :raises ValueError: if **dim** is neither an int or **dim** < 1, if **M_L** <= 0, if **M_L** is neither an int or float or np.float16 or np.float32 or np.float64
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
        '''
        Check whether radial distance is positive and has a unit.
        
        :param r: radial distance
        :type r: int or float or astropy.units Quantity
        :param against: parameter to retrieve the unit from
        :type against: astropy.unit Quantity
        
        :returns: the radial distance
        :rtype: astropy.units Quantity
        
        :raises ValueError: if **r** < 0
        '''
        
        if r<0:
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        if not hasattr(r, 'unit'):
            r *= against.unit

        return r
    
    
    ###########################################################
    #       Default methods (some need to be overriden)       #
    ###########################################################
    
    def flux(self, r, *args, **kwargs):
        r'''
        Compute the enclosed flux at radius r
        
        .. math::
            
            F(<r) = 4\pi \int_0^r dx~x^2 \rho (<x),
            
        with :math:`\rho(r)` the light density profile.

        :param r: radius where to compute the luminosity
        :type r: int or float
        :returns: flux enclosed in a sphere of radius r
        :rtype: astropy.units Quantity
        
        :raises NotImplementedError: This method needs be implemented in a subclass first in order to be used
        '''
        
        raise NotImplementedError('flux method not implemented yet.')
        return
    
    
    def gfield(self, r, *args, **kwargs):
        '''
        Compute the gravitational field at radius r.

        :param r: radius where the gravitational field is computed
        :type r: int or float

        :returns: g(r)
        :rtype: astropy.units Quantity
        
        :raises NotImplementedError: This method needs be implemented in a subclass first in order to be used
        '''
        
        raise NotImplementedError('gfield method not implemented yet.')
        return
    
    
    def mass(self, r, *args, **kwargs):
        '''
        Compute the enclosed (integrated) mass at radius r.

        :param r: radius where to compute the mass
        :returns: mass enclosed in a sphere of radius r
        :rtype: astropy.units Quantity
        
        :raises NotImplementedError: This method needs be implemented in a subclass first in order to be used
        '''
        
        return self.M_L*self.flux(r, *args, **kwargs)
    
    
    def mass_profile(self, r, *args, **kwargs):
        '''
        Compute the **mass density** profile at radius r.

        :param r: radius where to compute the profile
        :type r: int or float

        :returns: mass density profile at radius r
        :rtype: astropy.units Quantity
        '''
        
        return self.M_L*self.profile(r, *args, **kwargs)
    
    
    def profile(self, r, *args, **kwargs):
        '''
        Compute the light density profile at radius r.

        :param r: radius where to compute the profile
        :type r: int or float
        
        :returns: light profile at radius r
        :rtype: astropy.units Quantity
        
        :raises NotImplementedError: This method needs be implemented in a subclass first in order to be used
        '''
        
        raise NotImplementedError('profile method not implemented yet.')
        return
    
    
    def velocity(self, r, *args, **kwargs):
        r'''
        Velocity profile for a self-sustaining 3D profile against its own gravity through centripedal acceleration
        
        .. math::
            
            V(r) = \sqrt{\frac{G M(<r)}{r}},
            
        where :math:`G` is the gravitational constant and :math:`M(<r)` is the enclosed mass.

        :param r: radius where the velocity profile is computed
        :type r: int or float
        :returns: V(r)
        :rtype: astropy.units Quantity
        '''

        return np.sqrt(G*self.mass(r)/r)
    
    
    ################################
    #          Properties          #
    ################################
    
    @property
    def Ftot(self):
        '''
        Total flux of the profile.
        
        :raises NotImplementedError: This property needs be implemented in a subclass first in order to be used
        '''
        
        raise NotImplementedError('Ftot method not implemented yet.')
        return
    
    @property
    def Mtot(self):
        '''
        Total mass of the profile.
        
        :raises NotImplementedError: This property needs be implemented in a subclass first in order to be used
        '''
        
        return self.M_L*self.Ftot


class Multiple3DModels(MassModelBase):
    '''A master class used when combining two 3D models into a single object.'''
    
    def __init__(self, model1, model2, *args, **kwargs):
        '''
        Init function.
    
        :param model1: 1st model
        :type model1: MassModelBase[_dim = 3]
        :param model2: 2nd model
        :type model2: MassModelBase[_dim = 3]
        
        :raises AttributeError: if **model1** or **model2** does not have a **_dim** attribute
        :raises ValueError: if **model1** or **model2** do not have **_dim** = 3
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
        '''Check args and kwargs so that they have the same len.'''
        
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
        
        .. note:: 
            
            Because models can require different arguments (for instance 3D radial distance r for one and 2D R in disk plane for the other), these are separated in lists.
            Each element will be passed to the corresponding model. The order is the same as in models list variable.
    
        :param list[list] args: (**Optional**) arguments to pass to each model. Order is that of models list.
        :param list[list] kwargs: (**Optional**) kwargs to pass to each model. Order is that of models list.
        
        :returns: sum of the gravitational fields for each model
        '''
        
        args, kwargs = self.__checkArgs__(args, kwargs)
        return np.sum([i.gfield(*args[pos], **kwargs[pos]) for pos, i in enumerate(self.models)], axis=0)
        
        
        
    def flux(self, args=[], kwargs=[{}]):
        '''
        Compute the enclosed flux at radius r.
        
        .. note:: 
            
            Because models can require different arguments (for instance 3D radial distance r for one and 2D R in disk plane for the other), these are separated in lists.
            Each element will be passed to the corresponding model. The order is the same as in models list variable.
    
        :param list[list] args: (**Optional**) arguments to pass to each model. Order is that of models list.
        :param list[list] kwargs: (**Optional**) kwargs to pass to each model. Order is that of models list.
        
        :returns: sum of the flux for each model
        '''
        
        args, kwargs = self.__checkArgs__(args, kwargs)
        return np.sum([i.mass(*args[pos], **kwargs[pos]) for pos, i in enumerate(self.models)], axis=0)
        
    
    def mass(self, args=[], kwargs=[{}]):
        '''
        Compute the enclosed mass at radius r.
        
        .. note:: 
            
            Because models can require different arguments (for instance 3D radial distance r for one and 2D R in disk plane for the other), these are separated in lists.
            Each element will be passed to the corresponding model. The order is the same as in models list variable.
    
        :param list[list] args: (**Optional**) arguments to pass to each model. Order is that of models list.
        :param list[list] kwargs: (**Optional**) kwargs to pass to each model. Order is that of models list.
        
        :returns: sum of the mass for each model
        '''
        
        args, kwargs = self.__checkArgs__(args, kwargs)
        return np.sum([i.mass(*args[pos], **kwargs[pos]) for pos, i in enumerate(self.models)], axis=0)
    
    
    def mass_profile(self, args=[], kwargs=[{}]):
        '''
        Compute the mass profile at radius r.

        .. note:: 
            
            Because models can require different arguments (for instance 3D radial distance r for one and 2D R in disk plane for the other), these are separated in lists.
            Each element will be passed to the corresponding model. The order is the same as in models list variable.
    
        :param list[list] args: (**Optional**) arguments to pass to each model. Order is that of models list.
        :param list[list] kwargs: (**Optional**) kwargs to pass to each model. Order is that of models list.
        
        :returns: sum of the mass density profile for each model
        '''
        
        args, kwargs = self.__checkArgs__(args, kwargs)
        return np.sum([i.mass_profile(*args[pos], **kwargs[pos]) for pos, i in enumerate(self.models)], axis=0)
    
            
    def profile(self, args=[], kwargs=[{}]):
        '''
        Compute the light profile at radius r.

        .. note:: 
            
            Because models can require different arguments (for instance 3D radial distance r for one and 2D R in disk plane for the other), these are separated in lists.
            Each element will be passed to the corresponding model. The order is the same as in models list variable.
    
        :param list[list] args: (**Optional**) arguments to pass to each model. Order is that of models list.
        :param list[list] kwargs: (**Optional**) kwargs to pass to each model. Order is that of models list.
        
        :returns: sum of the light profile for each model
        '''
        
        args, kwargs = self.__checkArgs__(args, kwargs)
        return np.sum([i.profile(*args[pos], **kwargs[pos]) for pos, i in enumerate(self.models)], axis=0)
    
    
    ################################################################
    #                          Velocities                          #
    ################################################################            
    
    def velocity(self, args=[], kwargs=[{}]):
        '''
        Velocity profile for the 3D models against their own gravity through centripedal acceleration.

        .. note:: 
            
            Because models can require different arguments (for instance 3D radial distance r for one and 2D R in disk plane for the other), these are separated in lists.
            Each element will be passed to the corresponding model. The order is the same as in models list variable.
    
        :param list[list] args: (**Optional**) arguments to pass to each model. Order is that of models list.
        :param list[list] kwargs: (**Optional**) kwargs to pass to each model. Order is that of models list.
        
        :returns: sum of the velocity profile for each model
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
    '''
    3D Hernquist model class.
    
    .. warning::
    
        When using a method, make sure the unit of the radial distance is identical to that of the scale parameter of the model.
    '''
    
    def __init__(self, a, F, M_L, unit_a='kpc', unit_F='erg/(s.A)', unit_M_L='solMass.s.A.cm^2/(erg.kpc^2)', **kwargs):
        '''
        Init instance.
    
        :param a: scale factor
        :type a: int or float
        :param F: amplitude parameter (total flux)
        :type F: int or float
        :param M_L: mass to light ratio
        :type M_L: int or float
               
        :param str unit_a: unit of the scale parameter a
        :param str unit_F: unit of the amplitude parameter M
        :param str unit_M_L: unit used to convert from light to mass profiles
        
        :raises TypeError: if **a** or **F** are neither int, float, np.foat16, np.float32 or np.float64
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
        
        #: Offset parameter to make the conversion Re (Sersic) <-> a (Hernquist) (see Mercier et al., 2021)
        self._alpha_a = -0.454
        
        #: Slope parameter to make the conversion Re (Sersic) <-> a (Hernquist) (see Mercier et al., 2021)
        self._beta_a  = 0.725
        
        #: Offset parameter to make the conversion Ie (Sersic) <-> F (Hernquist) (see Mercier et al., 2021)
        self._alpha_F = 1.194
        
        #: Slope parameter to make the conversion Ie (Sersic) <-> F (Hernquist) (see Mercier et al., 2021)
        self._beta_F  = 1.75
        
        
    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################    
    
    def flux(self, r, *args, **kwargs):
        '''
        Compute the enclosed flux at radius r.

        :param r: radius where to compute the luminosity
        :type r: int or float
        
        :returns: flux enclosed in a sphere of radius r
        :rtype: astropy.units Quantity 
        '''
        
        r = self._checkR(r, self.a)
        return self.F*(r/(r+self.a))**2
    
    
    def gfield(self, r, *args, **kwargs):
        '''
        Compute the gravitational field of a single Hernquist profile at radius r.

        :param r: radius where the gravitational field is computed
        :type r: int or float

        :returns: g(r)
        :rtype: astropy.units Quantity
        '''
        
        r = self._checkR(r, self.a)
            
        if r.value == 0:
            return -G*self.F/self.a**2
        else:   
            return -G*self.flux(r)*self.M_L/r**2
        
        
    def profile(self, r, *args, **kwargs):
        '''
        Compute the light density profile at radius r.

        :param r: radius where to compute the profile
        :type r: int or float
        
        :returns: light profile at radius r
        :rtype: astropy.units Quantity
        '''
        
        r = self._checkR(r, self.a)
        return self.F*self.a/(2*np.pi) / (r * (r + self.a)**3)
    
    
    def velocity(self, r, *args, **kwargs):
        r'''
        Velocity profile for a self-sustaining Hernquist 3D profile against its own gravity through centripedal acceleration.
        
        .. math::
            
            V(r) = \sqrt{\frac{G M(<r)}{r}},
            
        where :math:`G` is the gravitational constant and :math:`M(<r)` is the enclosed mass.

        :param r: radius where the velocity profile is computed
        :type r: int or float
        :returns: V(r) in units of G*M/L*F/sqrt(a)
        :rtype: astropy.units Quantity
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
        '''Alias of :py:attr:`toSersic`.'''
        
        return self.toSersic
    
    @property
    def toSersic(self):
        '''
        Create the best fit Sersic instance from this Hernquist instance.
        
        :returns: Best-fit Sersic instance (see Mercier et al., 2021)
        :rtype: Sersic
        '''
        
        Re = ((self.a.value/10**self._alpha_a)**(1.0/self._beta_a)) * self.a.unit
        Ie = (self.F.value / (10**self._alpha_F * Re.value**self._beta_F)) * (self.F.unit/(Re.unit**2))
        
        return deVaucouleur(Re, Ie=Ie, unit_Re=str(Re.unit), unit_Ie=str(Ie.unit))
        
        
class NFW(MassModelBase):
    '''
    Navarro Frenk and White profile.
    
    .. warning::
    
        When using a method, make sure the unit of the radial distance is identical to that of the scale parameter of the model.
    '''
    
    def __init__(self, Rs, c=None, Vmax=None, unit_Rs='kpc', unit_Vmax='km/s'):
        '''
        Init NFW profile. 
        
        .. note::
            
            Two pairs of parameters can be passed:
                - **Rs** and **c** 
                - **Rs** and **Vmax**

        :param Rs: scale parameter
        :type Rs: int or float or astropy.units Quantity with distance unit
        
        :param c: (**Optional**)  concentration parameter. If None, **Vmax** must be given.
        :type c: int or float
        :param Vmax: (**Optional**) maximum circular velocity at 2.15*Rs. If None, **c** must be given. If not given as an astropy Quantity, provide the correct unit with **unit_Vmax**.
        :type Vmax: int or float or astropy.units Quantity with velocity unit
        :param str unit_Rs: (**Optional**) unit of **Rs**
        :param str unit_Vmax: (**Optional**) unit of **Vmax**
        
        :raises ValueError: if both **Vmax** and **c** are None or if both are not None
        :raises TypeError: if **c** is not dimensionless
        :raises astropy.units.core.UnitConversionError: if **Vmax** could not be broadcast to km/s unit
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
        Compute the enclosed flux at radius r.
        
        .. note::
            
            The flux returned is 0 since this is a DM profile.

        :param r: radius where to compute the luminosity
        :type r: int or float
        :returns: flux enclosed in a sphere of radius r
        :rtype: astropy.units Quantity
        '''
        
        r = self._checkR(r, self.Rs).value
        return u.Quantity(0*r, unit='erg/(s.A)')
    
    
    def gfield(self, r, *args, **kwargs):
        '''
        Compute the gravitational field of a single NFW profile at radius r.

        :param r: radius where the gravitational field is computed
        :type r: int or float

        :returns: g(r)
        :rtype: astropy.units Quantity
        
        :raises NotImplementedError: This method needs be implemented
        '''
        
        raise NotImplementedError('gfield not implemented for a NFW profile')
        return
    
    
    def mass(self, r, *args, **kwargs):
         '''
        Compute the enclosed (integrated) mass at radius r.
        
        :param r: radius where to compute the mass
        :type r: int or float
        :returns: mass enclosed in a sphere of radius r
        :rtype: astropy.units Quantity'''
        
         r = self._checkR(r, self.Rs)
         return r*self.velocity(r, *args, **kwargs)**2 / G

    
    def mass_profile(self, r, *args, **kwargs):
        '''
        Compute the **mass density** profile at radius r.

        :param r: radius where to compute the profile
        :type r: int or float

        :returns: mass density profile at radius r
        :rtype: astropy.units Quantity
        '''
        
        r = self._checkR(r, self.Rs)
        return self.Vmax**2 / (4*np.pi*G*self.Rs*self._factor * r*(1+r/self.Rs)**2)
    
    
    def profile(self, r, *args, **kwargs):
        '''
        Compute the light density profile at radius r.
        
        .. note::
            
            The light density profile returned is 0 since this is a DM profile.

        :param r: radius where to compute the profile
        :type r: int or float
        
        :returns: light profile at radius r
        :rtype: astropy.units Quantity
        '''
        
        r = self._checkR(r, self.Rs).value
        return u.Quantity(0*r, unit='erg/(s.A.cm^2)')
    
    
    def velocity(self, r, *args, **kwargs):
        r'''
        Velocity profile for a self-sustaining 3D profile against its own gravity through centripedal acceleration
        
        .. math::
            
            V(r) = \sqrt{\frac{G M(<r)}{r}},
            
        where :math:`G` is the gravitational constant and :math:`M(<r)` is the enclosed mass.

        :param r: radius where the velocity profile is computed
        :type r: int or float
        :returns: V(r)
        :rtype: astropy.units Quantity
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
        '''
        Virial radius.
        
        :raises ValueError: if **c** is None
        '''
        
        if self.c is None:
            raise ValueError('Virial radius cannot be computed if the concentration parameter is not provided.')
        else:
            return self.c*self.Rs
        

#################################################################################
#                                Sersic profiles                                #
#################################################################################

class Sersic:
    '''
    2D Sersic profile class.
    
    .. warning::
    
        When using a method, make sure the unit of the radial distance is identical to that of the scale parameter of the model.
    '''
    
    def __init__(self, n, Re, Ie=None, mag=None, offset=None, unit_Re='kpc', unit_Ie='erg/(cm^2.s.A)', **kwargs):
        """
        Init mass model.
        
        .. note::
            
            You can either provide:
                - **n**, **Re** and **Ie**
                - **n**, **Re**, **mag** and **offset**
        
        :param n: Sersic index
        :type n: int or flat
        :param Re: half-light radius
        :type Re: int or float
        
        :param float Ie: (**Optional**) intensity at half-light radius
        :param float mag: (**Optional**) galaxy total integrated magnitude used to compute Ie if not given
        :param float offset: (**Optional**) magnitude offset in the magnitude system used
        :param str unit_Ie: (**Optional**) unit for the surface brightness. If **Ie** already has a unit, it is converted to this unit.
        :param str unit_Re: (**Optional**) unit for the scale radius **Re**. If **Re** already has a unit, it is converted to this unit.
        
        :raises TypeError: if n or Re are neither int, float, np.float16, np.float32 or np.float64
        :raises ValueError: if Ie and mag and offset are None, if n <= 0, if Re <= 0 or if Ie <= 0
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
        
        #: Sersic index
        self.n        = n
        
        #: Sersic :math:`b_n` factor defined as :math:`2\gamma(2n, b_n) = \Gamma(2n)`
        self.bn       = compute_bn(self.n)
        
        # If the parameters already are Astropy quantities, we just convert them to the required unit
        if hasattr(Re, 'unit'):
            #: Effective (half-light) radius
            self.Re   = Re.to(unit_Re)
        else:
            self.Re   = u.Quantity(Re, unit=unit_Re)
            
        if hasattr(Ie, 'unit'):
            #: Surface brightness at Re
            self.Ie   = Ie.to(unit_Ie)
        else:
            self.Ie   = u.Quantity(checkAndComputeIe(Ie, self.n, self.bn, self.Re.value, mag, offset), unit=unit_Ie)
            
        #: Central intensity
        self.I0       = self.Ie*np.exp(self.bn)
        

    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
    
    def _checkR(self, r, against):
        '''
        Check whether radial distance is positive and has a unit.
        
        :param r: radial distance
        :type r: int or float or astropy.units Quantity
        :param against: parameter to retrieve the unit from
        :type against: astropy.unit Quantity
        
        :returns: the radial distance
        :rtype: astropy.units Quantity
        
        :raises ValueError: if **r** < 0
        '''
        
        if r<0:
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        if not hasattr(r, 'unit'):
            r *= against.unit

        return r
        
    def profile(self, r, *args, **kwargs):
        '''
        Sersic surface brightness profile at radius r.

        :param r: radius where the profile is computed
        :type r: int or float
        :returns: Sersic profile computed at the radius r
        :rtype: float
        '''
        
        r = self._checkR(r, self.Re)
        return sersic_profile(r, self.n, self.Re, Ie=self.Ie, bn=self.bn)
    
    
    def flux(self, r, *args, **kwargs):
        '''
        Flux at radius r (encompassed within a disk since this is a 2D profile).

        :param r: radius where the profile is computed
        :type r: int or float
        :returns: flux computed within a disk of radius r
        :rtype: float
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
    '''
    2D de Vaucouleur profile.

    .. warning::

        When using a method, make sure the unit of the radial distance is identical to that of the scale parameter of the model.
    '''
    
    def __init__(self, Re, Ie=None, mag=None, offset=None, unit_Re='kpc', unit_Ie='erg/(cm^2.s.A)', **kwargs):
        """
        Init mass model.
        
        .. note::
            
            You can either provide:
                - **Re** and **Ie**
                - **Re**, **mag** and **offset**
        
        :param Re: half-light radius
        :type Re: int or float
        
        :param float Ie: (**Optional**) intensity at half-light radius
        :param float mag: (**Optional**) galaxy total integrated magnitude used to compute Ie if not given
        :param float offset: (**Optional**) magnitude offset in the magnitude system used
        :param str unit_Ie: (**Optional**) unit for the surface brightness. If **Ie** already has a unit, it is converted to this unit.
        :param str unit_Re: (**Optional**) unit for the scale radius **Re**. If **Re** already has a unit, it is converted to this unit.
        """
        
        super().__init__(4, Re, Ie=Ie, mag=mag, offset=offset, unit_Re=unit_Re, unit_Ie=unit_Ie, **kwargs)
        
        #: Offset parameter to make the conversion Re (Sersic) <-> a (Hernquist) (see Mercier et al., 2021)
        self._alpha_a = -0.454
        
        #: Slope parameter to make the conversion Re (Sersic) <-> a (Hernquist) (see Mercier et al., 2021)
        self._beta_a  = 0.725
        
        #: Offset parameter to make the conversion Ie (Sersic) <-> F (Hernquist) (see Mercier et al., 2021)
        self._alpha_F = 1.194
        
        #: Slope parameter to make the conversion Ie (Sersic) <-> F (Hernquist) (see Mercier et al., 2021)
        self._beta_F  = 1.75
        
        
    def toHernquist(self, M_L, unit_M_L='solMass.s.A.cm^2/(erg.kpc^2)', **kwargs):
        '''
        Create the best-fit Hernquist instance from this de Vaucouleur instance (see Mercier et al., 2021).
    
        :param M_L: mass to light ratio
        :type M_L: int or float
        
        :param str unit_M_L: unit of the mass to light ratio. If M_L already has a unit, it is converted to this unit.
        
        :returns: best-fit Hernquist instance (see Mercier et al., 2021)
        :rtype: Hernquist
        
        :raises TypeError: if M_L is neither an int, float, float16, float32, float64 or astropy.units Quantity
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
    
    .. note::
        
        Some functions such as the light profile correspond to the 2D Sersic profiles, while others such as the velocity assume a 3D **razor-thin disk**.
        
    .. warning::
    
        When using a method, make sure the unit of the radial distance is identical to that of the scale parameter of the model.
    '''
    
    def __init__(self, Re, M_L, Ie=None, mag=None, offset=None, unit_Re='kpc', unit_Ie='erg/(cm^2.s.A)', unit_M_L='solMass.s.A.cm^2/(erg.kpc^2)', **kwargs):        
        """
         Init mass model.
        
        .. note::
            
            You can either provide:
                - **M_L**, **Re** and **Ie**
                - **M_L**, **Re**, **mag** and **offset**
        
        :param M_L: mass to light ratio
        :type M_L: int or float
        :param Re: half-light radius
        :type Re: int or float
        
        :param float Ie: (**Optional**) intensity at half-light radius
        :param float mag: (**Optional**) galaxy total integrated magnitude used to compute Ie if not given
        :param float offset: (**Optional**) magnitude offset in the magnitude system used
        :param str unit_Ie: (**Optional**) unit for the surface brightness. If **Ie** already has a unit, it is converted to this unit.
        :param str unit_M_L: unit of the mass to light ratio
        :param str unit_Re: (**Optional**) unit for the scale radius **Re**. If **Re** already has a unit, it is converted to this unit.
        
        :raises astropy.units.core.UnitConversionError: if **Vmax** could not be broadcast to km/s unit
        """
        
        MassModelBase.__init__(self, 3, M_L, unit_M_L)
        Sersic.__init__(       self, 1, Re, Ie=Ie, mag=mag, offset=offset, unit_Re=unit_Re, unit_Ie=unit_Ie)
        
        #: Disk scale length :math:`R_{\rm{d}} = R_{\rm{e}} / b_n`
        self.Rd       = self.Re/self.bn
        
        #: Position of maximum velocity :math:`2.15 \times R_{\rm{d}}`
        self.Rmax     = 2.15*self.Rd
        
        try:
            #: Maximum velocity :math:`0.8798243 \times \sqrt{\pi G R_{\rm{d}} \Sigma_0 \Upsilon }` with :math:`\Sigma_0` the central surface brightness and :math:`\Upsilon` the mass to light ratio
            self.Vmax = 0.8798243*np.sqrt(np.pi*G*self.Rd*self.M_L*self.I0).to('km/s')
        except UnitConversionError:
            raise UnitConversionError('The unit of Vmax (%s) could not be converted to km/s. Please check carefully the units of Ie, Re and M_L parameters. Cheers !' %np.sqrt(np.pi*G*self.Rd*self.M_L*self.Ie*np.exp(self.bn)).unit)
        
        
    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
    
    def velocity(self, r, *args, **kwargs):
        r'''
        Velocity profile for a self-sustaining 3D inifinitely thin disk against its own gravity through centripedal acceleration
        
        .. math::
            
            V(r) = V_{\rm{max}} \frac{r}{0.8798243 \times R_{\rm{d}}} \sqrt{I_0(y) K_0(y) - I_1 (y) K_1 (y)}
            
        where :math:`I_i, K_i` are the modified Bessel functions of the first and second kind, respectively, of order i, and  :math:`y = r/(2 R_{\rm{d}})`.

        :param r: radius where the velocit profile is computed
        :type r: int or float
        :returns: velocity profile at a radius **r**
        :rtype: float
        '''
        
        r = self._checkR(r, self.Re)
        
        if r.value == 0:
            return 0
        else:
            y = r/(2*self.Rd)
            return (self.Vmax*r/(0.8798243*self.Rd)) * np.sqrt(i0(y)*k0(y) - i1(y)*k1(y))
        

class DoubleExponentialDisk(Sersic, MassModelBase):
    '''
    Double exponential disk profile class implementing Bovy rotation curve.
    
    .. warning::
    
        When using a method, make sure the unit of the radial distance is identical to that of the scale parameter of the model.
    
    '''
    
    def __init__(self, Re, hz, M_L, q0=None, Ie=None, mag=None, offset=None, 
                 unit_Re='kpc', unit_hz='kpc', unit_Ie='erg/(cm^2.s.A)', unit_M_L='solMass.s.A.cm^2/(erg.kpc^2)', **kwargs):        
        """
         Init mass model.
        
        .. note::
            
            You can either provide:
                - **M_L**, **hz**, **Re** and **Ie**
                - **M_L**, **hz**, **Re**, **mag** and **offset**
                
            You can also provide **q0** instead of **hz** (which must be None in this case).
            
        :param hz: vertical scale height
        :type hz: int or float
        :param M_L: mass to light ratio
        :type M_L: int or float
        :param Re: half-light radius in the plane of symmetry
        :type Re: int or float
                
        :param float Ie: (**Optional**) intensity at half-light radius
        :param float mag: (**Optional**) galaxy total integrated magnitude used to compute **Ie** if not given
        :param float offset: (**Optional**) magnitude offset in the magnitude system used
        :param float q0: (**Optional**) axis ratio equal to hz/Rd with Rd the disk scale length. If this value is different from None, it overrides **hz**.
        :param str unit_Ie: (**Optional**) unit for the surface brightness
        :param str unit_M_L: (**Optional**) unit of the mass to light ratio
        :param str unit_Re: (**Optional**) unit for the scale radius **Re**
        """
        
        MassModelBase.__init__(self, 3, M_L, unit_M_L)
        Sersic.__init__(       self, 1, Re, Ie=Ie, mag=mag, offset=offset, unit_Re=unit_Re, unit_Ie=unit_Ie)
        
        #: Disk scale length :math:`R_{\rm{d}} = R_{\rm{e}} / b_n`
        self.Rd       = self.Re/self.bn
        
        if q0 is not None:
            if q0 >= 0 and q0 <= 1:
                #: Axis ratio
                self.q0 = q0
                
                #: Scale height
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
        r'''
        Velocity correction necessary for the rotation curve
        
        .. math::
        
            V_{\rm{corr}} (r) = V_{\rm{c, max}} \times \sqrt{\frac{r e^{1-r/R_{\rm{d}}}}{R_{\rm{d}}}},
        
        where :math:`R_{\rm{d}}` is the disk scale length, and :math:`V_{\rm{c, max}}` is the maximum of the velocity correction.

        :param R: radius where the velocity correction is computed
        :type R: float or ndarray[float]
        :returns: velocity correction
        :rtype: float or ndarray[float]
        '''
        
        R = self._checkR(R, self.Re)
        return self._Vmax_correction * np.sqrt(R * np.exp(1-R/self.Rd) / self.Rd)
    
    
    def _velocity_razor_thin(self, R, *args, **kwargs):
        '''
        Velocity if the disk was razor thin.
                
        :param R: radius where the velocity is computed
        :type R: float or ndarray[float]
        :returns: velocity if the disk was razor thin
        :rtype: float or ndarray[float]
        
        .. seealso:: :py:meth:`ExponentialDisk.velocity`
        '''
        
        R       = self._checkR(R, self.Re)
        disk_RT = ExponentialDisk(self.Re, self.M_L, Ie=self.Ie)
        return disk_RT.velocity(R)
        
    
    @property
    def _Vmax_correction(self):
        r'''
        Maximum of the velocity correction
        
        .. math ::
            
            V_{\rm{c, max}} = \sqrt{2\pi G h_z \Upsilon \Sigma_0 / \exp\lbrace 1 \rbrace},
            
        with :math:`\Sigma_0` the central surface brightness if the disk was seen face-on, and :math:`\Upsilon` the mass to light ratio.
        '''
        
        return np.sqrt(2*np.pi*G*self.hz*self.M_L*self.I0/np.exp(1))
    
    
    @property
    def _Vmax_razor_thin(self):
        '''
        Maximum velocity if the disk was razor thin.
        
        .. seealso:: :py:meth:`ExponentialDisk.velocity`
        '''
        
        disk_RT = ExponentialDisk(self.Re, self.M_L, Ie=self.Ie)
        return disk_RT.Vmax
        
        
    def profile(self, R, z, *args, **kwargs):
        r'''
        Light density profile at radius r
        
        .. math::
            
            \rho(R, z) = \frac{\Sigma_0}{2 h_z} e^{-R/R_{\rm{e}} - |z|/h_z}
    
        with :math:`\Sigma_0 = I_{\rm{e}} e^{b_1}` the central surface density in the plane of the disk, :math:`R_{\rm{e}}` the disk effective radius in the plane of symmetry and :math:`h_z` the vertical scale height.

        :param R: radial position in the plane of symmetry
        :type R: int or float
        :param z: vertical position
        :type z: int or float
        :returns: double exponential profile computed at position (**R**, **z**)
        :rtype: float
        '''
        
        R = self._checkR(R, self.Re)
        z = self._checkR(z, self.hz)
        return sersic_profile(R, self.n, self.Re, Ie=self.Ie, bn=self.bn) * np.exp(-np.abs(z)/self.hz) / (2*self.hz)
    
    
    def velocity(self, R, *args, **kwargs):
        r'''
        Velocity profile for a self-sustaining 3D double exponential disk against its own gravity through centripedal acceleration
        
        .. math::
        
            V(r) = \sqrt{V_{\rm{RT}}^2 (r) - V_{\rm{corr}}^2 (r)},
        
        where :math:`V_{\rm{RT}}` is the rotation curve of a similar razor-thin disk, and :math:`V_{\rm{corr}}` is the correction to apply to take into account the finite thickness of the disk.
        
        .. note::
            
            This formula is a `Bovy rotation curve <http://astro.utoronto.ca/~bovy/AST1420/notes-2017/notebooks/05.-Flattened-Mass-Distributions.html#Potentials-for-disks-with-finite-thickness>`_, i.e. an approximation for a thin (but non-zero thickness) disk.
                
        :param R: radius where the velocity is computed
        :type R: float or ndarray[float]
        :returns: velocity if the disk was razor thin
        :rtype: float or ndarray[float]
        :raises TypeError: if **R** is neither int, float, np.float16, np.float32, np.foat64 or ndarray
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
            
    