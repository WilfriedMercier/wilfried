    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

3D and 2D mass models for different mass and light profiles.
"""

import numpy                      as     np
import astropy.units              as     u
import wilfried.galaxy.morphology as     morpho
from   astropy.units.core         import UnitConversionError
from   .models                    import checkAndComputeIe, sersic_profile
from   .misc                      import compute_bn, realGammainc
from   astropy.constants          import G
from   scipy.special              import gamma, i0, i1, k0, k1
from   numpy                      import ndarray

from   typing                     import Type, Union, List, Dict

try:
    from   astropy.cosmology      import Planck18 as cosmo
except ImportError:
    from   astropy.cosmology      import Planck15 as cosmo

# Custom types
QuantityType = Type[u.Quantity]

#######################################################################################
#                           3D profiles and their functions                           #
#######################################################################################

class MassModelBase:
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Base class for mass models.
    
    .. warning::
    
        When using a method, make sure the unit of the radial distance is identical to that of the scale parameter of the model.
        
    .. note::
            
        Units must be given such that they are recognised by astropy.units module.

    :param int dim: number of dimensions of the model
    :param float M_L: mass to light ratio
    :param str unit_M_L: unit of the mass to light ratio. Refer to the specific mass model to know which unit to provide.
    :raises ValueError: if 
    
    * **dim** is not an integer
    * **dim** < 1
    * **M_L** <= 0
    * **M_L** is neither an int or float or np.float16 or np.float32 or np.float64
    '''
    
    def __init__(self, dim: int, M_L: float, unit_M_L: str = 'solMass.s.cm^2.A/erg)', **kwargs) -> None:
        
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
    
    
    def _checkR(self, r: Union[int, float], against: QuantityType) -> QuantityType:
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
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        if not hasattr(r, 'unit'):
            r *= against.unit

        return r
    
    
    ###########################################################
    #       Default methods (some need to be overriden)       #
    ###########################################################
    
    def flux(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
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
    
    
    def gfield(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
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
    
    
    def mass(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
        '''
        Compute the enclosed (integrated) mass at radius r.

        :param r: radius where to compute the mass
        :type r: int or float
        
        :returns: mass enclosed in a sphere of radius r
        :rtype: astropy.units Quantity
        
        :raises NotImplementedError: This method needs be implemented in a subclass first in order to be used
        '''
        
        return self.M_L*self.flux(r, *args, **kwargs)
    
    
    def mass_profile(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
        '''
        Compute the **mass density** profile at radius r.

        :param r: radius where to compute the profile
        :type r: int or float

        :returns: mass density profile at radius r
        :rtype: astropy.units Quantity
        '''
        
        return self.M_L*self.profile(r, *args, **kwargs)
    
    
    def profile(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
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
    
    
    def velocity(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
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
    def Ftot(self) -> None:
        '''
        Total flux of the profile.
        
        :raises NotImplementedError: This property needs be implemented in a subclass first in order to be used
        '''
        
        raise NotImplementedError('Ftot method not implemented yet.')
        return
    
    @property
    def Mtot(self) -> None:
        '''
        Total mass of the profile.
        
        :raises NotImplementedError: This property needs be implemented in a subclass first in order to be used
        '''
        
        return self.M_L*self.Ftot

class Multiple3DModels(MassModelBase):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    A master class used when combining two 3D models into a single object.
    '''
    
    def __init__(self, model1, model2, *args, **kwargs):
        '''
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
            
    
    def gfield(self, args: List[List] = [[]], kwargs: List[Dict] = [{}]) -> QuantityType:
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
        
        
        
    def flux(self, args: List[List] = [[]], kwargs: List[Dict] = [{}]) -> QuantityType:
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
        
    
    def mass(self, args: List[List] = [[]], kwargs: List[Dict] = [{}]) -> QuantityType:
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
    
    
    def mass_profile(self, args: List[List] = [[]], kwargs: List[Dict] = [{}]) -> QuantityType:
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
    
            
    def profile(self, args: List[List] = [[]], kwargs: List[Dict] = [{}]) -> QuantityType:
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
    
    def velocity(self, args: List[List] = [[]], kwargs: List[Dict] = [{}]) -> QuantityType:
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
    def Ftot(self) -> QuantityType:
        '''Total flux of the profiles.'''
        
        return np.sum([i.Ftot for i in self.models], axis=0)
    
    @property
    def Mtot(self) -> QuantityType:
        '''Total mass of the profiles.'''
        
        return np.sum([i.Mtot for i in self.models], axis=0)


class Hernquist(MassModelBase):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    3D Hernquist model class.
    '''
    
    def __init__(self, a: Union[int, float], F: Union[int, float], M_L: Union[int, float], 
                 unit_a: str='kpc', unit_F: str='erg/(s.A)', unit_M_L: str='solMass.s.A.cm^2/(erg.kpc^2)', 
                 **kwargs) -> None:
        '''
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
    
    def flux(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
        '''
        Compute the enclosed flux at radius r.

        :param r: radius where to compute the luminosity
        :type r: int or float
        
        :returns: flux enclosed in a sphere of radius r
        :rtype: astropy.units Quantity 
        '''
        
        r = self._checkR(r, self.a)
        return self.F*(r/(r+self.a))**2
    
    
    def gfield(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
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
        
        
    def profile(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
        '''
        Compute the light density profile at radius r.

        :param r: radius where to compute the profile
        :type r: int or float
        
        :returns: light profile at radius r
        :rtype: astropy.units Quantity
        '''
        
        r = self._checkR(r, self.a)
        return self.F*self.a/(2*np.pi) / (r * (r + self.a)**3)
    
        
    def velocity(self, r: Union[int, float, QuantityType], *args, **kwargs) -> QuantityType:
        r'''
        Velocity profile for a self-sustaining Hernquist 3D profile against its own gravity through centripedal acceleration.
        
        .. math::
            
            V(r) = \sqrt{\frac{G M(<r)}{r}},
            
        where :math:`G` is the gravitational constant and :math:`M(<r)` is the enclosed mass.

        :param r: radius where the velocity profile is computed
        :type r: int, float, astropy Quantity or ndarray[float]
        
        :returns: velocity profile at a radius **r** in :math:`\rm{km/s}`
        :rtype: Astropy Quantity
        
        :raises TypeError: if **r** is neither int, float, np.float16, np.float32, np.foat64 or ndarray
        '''
        
        r             = self._checkR(r, self.a)
        
        if isinstance(r.value, (int, float, np.float16, np.float32, np.float64)):
            
            if r.value == 0:
                vr   = u.Quantity(0, 'km/s')
            else:
                vr   = np.sqrt(G*self.M_L*self.F*r)/(self.a+r)
            
        elif isinstance(r.value, np.ndarray):
            vr       = u.Quantity(np.zeros(np.shape(r)), unit='km/s')
            mask     = r != 0
            rmask    = r[mask]
            vr[mask] = np.sqrt(G*self.M_L*self.F*rmask)/(self.a+rmask)
        
        else:
            raise TypeError('r array has type %s but only float and numpy array are expected.' %type(r))
        
        return vr.to('km/s')
    
    
    ################################
    #          Properties          #
    ################################
    
    @property
    def Ftot(self) -> QuantityType:
        '''Total flux of the profile.'''
        
        return self.F
    
    @property
    def todeVaucouleur(self) -> QuantityType:
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
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Navarro Frenk and White profile.
    '''
    
    def __init__(self, Rs: Union[int, float, QuantityType], 
                 c: Union[int, float] = None, Vmax: Union[int, float, QuantityType] = None, 
                 unit_Rs: str = 'kpc', unit_Vmax: str = 'km/s') -> None:
        '''
        .. note::
            
            Two pairs of parameters can be passed:
                
                * **Rs** and **c** 
                * **Rs** and **Vmax**

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
        
        Rmax              = 2.163
        self._factor      = np.log(1+Rmax)/Rmax - 1/(1+Rmax)
        
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
    
    def flux(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
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
    
    
    def gfield(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
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
    
    
    def mass(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
         '''
        Compute the enclosed (integrated) mass at radius r.
        
        :param r: radius where to compute the mass
        :type r: int or float
        :returns: mass enclosed in a sphere of radius r
        :rtype: astropy.units Quantity'''
        
         r = self._checkR(r, self.Rs)
         return r*self.velocity(r, *args, **kwargs)**2 / G

    
    def mass_profile(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
        '''
        Compute the **mass density** profile at radius r.

        :param r: radius where to compute the profile
        :type r: int or float

        :returns: mass density profile at radius r
        :rtype: astropy.units Quantity
        '''
        
        r = self._checkR(r, self.Rs)
        return self.Vmax**2 / (4*np.pi*G*self.Rs*self._factor * r*(1+r/self.Rs)**2)
    
    
    def profile(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
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
        
        
    def velocity(self, r: Union[int, float, QuantityType], *args, **kwargs) -> QuantityType:
        r'''
        Velocity profile for a self-sustaining 3D profile against its own gravity through centripedal acceleration
        
        .. math::
            
            V(r) = \sqrt{\frac{G M(<r)}{r}},
            
        where :math:`G` is the gravitational constant and :math:`M(<r)` is the enclosed mass.

        :param r: radius where the velocity profile is computed
        :type r: int, float, astropy Quantity or ndarray[float]
        
        :returns: velocity profile at a radius **r** in :math:`\rm{km/s}`
        :rtype: Astropy Quantity
        
        :raises TypeError: if **r** is neither int, float, np.float16, np.float32, np.foat64 or ndarray
        '''
        
        r             = self._checkR(r, self.Rs)
        
        if isinstance(r.value, (int, float, np.float16, np.float32, np.float64)):
            
            if r.value == 0:
                vr   = u.Quantity(0, 'km/s')
            else:
                vr   = self.Vmax * np.sqrt(self.Rs * (np.log(1+r/self.Rs)/r - 1/(r+self.Rs)) / self._factor)
            
        elif isinstance(r.value, np.ndarray):
            vr       = u.Quantity(np.zeros(np.shape(r)), unit='km/s')
            mask     = r != 0
            rmask    = r[mask]
            vr[mask] = self.Vmax * np.sqrt(self.Rs * (np.log(1+rmask/self.Rs)/rmask - 1/(rmask+self.Rs)) / self._factor)
        
        else:
            raise TypeError('r array has type %s but only float and numpy array are expected.' %type(r))
        
        return vr.to('km/s')
        
    
    ################################
    #          Properties          #
    ################################
    
    @property
    def Ftot(self) -> float:
        '''Total flux (infinite).'''
        
        return np.inf
    
    @property
    def Mvir(self) -> QuantityType:
        '''Virial mass.'''
        
        return (800*np.pi*cosmo.critical_density(0)*self.Rvir**3)/3

    @property    
    def Rvir(self) -> QuantityType:
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
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    2D Sersic profile class.
    
    .. warning::
    
        When using a method, make sure the unit of the radial distance is identical to that of the scale parameter of the model.
    '''
    
    def __init__(self, n: Union[int, float], Re: Union[int, float], 
                 Ie: float = None, mag: float = None, offset: float = None, 
                 unit_Re: str = 'kpc', unit_Ie: str = 'erg/(cm^2.s.A)', **kwargs) -> None:
        """
        .. note::
            
            You can either provide:
                
                * **n**, **Re** and **Ie**
                * **n**, **Re**, **mag** and **offset**
        
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
    
    def _checkR(self, r: Union[int, float, QuantityType], against: QuantityType) -> QuantityType:
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
        
        if np.any(r<0):
            raise ValueError('Radial distance is < 0. Please provide a positive valued distance.')
            
        if not hasattr(r, 'unit'):
            r *= against.unit

        return r
        
    def profile(self, r: Union[int, float], *args, **kwargs) -> float:
        '''
        Sersic surface brightness profile at radius r.

        :param r: radius where the profile is computed
        :type r: int or float
        :returns: Sersic profile computed at the radius r
        :rtype: float
        '''
        
        r = self._checkR(r, self.Re)
        return sersic_profile(r, self.n, self.Re, Ie=self.Ie, bn=self.bn)
    
    
    def flux(self, r: Union[int, float], *args, **kwargs) -> QuantityType:
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
    def Ftot(self) -> QuantityType:
        '''Total flux of the profile.'''
        
        n2 = 2*self.n
        return n2*np.pi*self.Ie*np.exp(self.bn)*gamma(n2)*self.Re**2 / self.bn**n2
    
    
class deVaucouleur(Sersic):
    '''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    2D de Vaucouleur profile.
    '''
    
    def __init__(self, Re: Union[int, float], Ie: float = None, mag: float  = None, offset: float = None, 
                 unit_Re: str ='kpc', unit_Ie: str = 'erg/(cm^2.s.A)', **kwargs) -> None:
        """
        .. note::
            
            You can either provide:
                
                * **Re** and **Ie**
                * **Re**, **mag** and **offset**
        
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
        
        
    def toHernquist(self, M_L: Union[int, float], unit_M_L: str = 'solMass.s.A.cm^2/(erg.kpc^2)', **kwargs) -> Type[Hernquist]:
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
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    2D/3D exponential disk profile class (razor-thin).
    
    .. note::
        
        Some functions such as the light profile correspond to the 2D Sersic profiles, while others such as the velocity assume a 3D **razor-thin disk**.
        
    .. note::
        
        You can either provide:
        
        * **M_L**, **Re** and **Ie**
        * **M_L**, **Re**, **mag** and **offset**
    
    :param M_L: mass to light ratio
    :type M_L: int or float
    :param Re: half-light radius
    :type Re: int or float
    
    :param float Ie: (**Optional**) intensity at half-light radius
    :param float mag: (**Optional**) galaxy total integrated magnitude used to compute Ie if not given
    :param float offset: (**Optional**) magnitude offset in the magnitude system used
    :param str unit_Ie: (**Optional**) unit for the surface brightness. If **Ie** already has a unit, it is converted to this unit.
    :param str unit_M_L: (**Optional**) unit of the mass to light ratio
    :param str unit_Re: (**Optional**) unit for the scale radius **Re**. If **Re** already has a unit, it is converted to this unit.
    
    :raises astropy.units.core.UnitConversionError: if **Vmax** could not be broadcast to km/s unit
    '''
    
    def __init__(self, Re: Union[int, float], M_L: Union[int, float], Ie: float = None, mag: float = None, offset: float = None, 
                 unit_Re: str = 'kpc', unit_Ie: str = 'erg/(cm^2.s.A)', unit_M_L: str = 'solMass.s.A.cm^2/(erg.kpc^2)', **kwargs) -> None:        
        
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
    
    def velocity(self, r: Union[int, float, QuantityType], *args, **kwargs) -> QuantityType:
        r'''
        Velocity profile for a self-sustaining 3D inifinitely thin disk against its own gravity through centripedal acceleration
        
        .. math::
            
            V(r) = V_{\rm{max}} \frac{r}{0.8798243 \times R_{\rm{d}}} \sqrt{I_0(y) K_0(y) - I_1 (y) K_1 (y)}
            
        where :math:`I_i, K_i` are the modified Bessel functions of the first and second kind, respectively, of order i, and  :math:`y = r/(2 R_{\rm{d}})`.

        :param r: radius where the velocity profile is computed
        :type r: int, float, astropy Quantity or ndarray[float]
        
        :returns: velocity profile at a radius **r** in :math:`\rm{km/s}`
        :rtype: Astropy Quantity
        
        :raises TypeError: if **r** is neither int, float, np.float16, np.float32, np.foat64 or ndarray
        '''
        
        r             = self._checkR(r, self.Re)
        
        if isinstance(r.value, (int, float, np.float16, np.float32, np.float64)):
            
            if r.value == 0:
                vr   = u.Quantity(0, 'km/s')
            else:
                y    = r/(2*self.Rd)
                vr   = (self.Vmax*r/(0.8798243*self.Rd)) * np.sqrt(i0(y)*k0(y) - i1(y)*k1(y))
            
        elif isinstance(r.value, np.ndarray):
            vr       = u.Quantity(np.zeros(np.shape(r)), unit='km/s')
            mask     = r != 0
            rmask    = r[mask]
            y        = rmask/(2*self.Rd)
            vr[mask] = (self.Vmax*rmask/(0.8798243*self.Rd)) * np.sqrt(i0(y)*k0(y) - i1(y)*k1(y))
        
        else:
            raise TypeError('r array has type %s but only float and numpy array are expected.' %type(r))
        
        return vr.to('km/s')
            
        
        

class DoubleExponentialDisk(Sersic, MassModelBase):
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Double exponential disk profile class implementing Bovy rotation curve.
    
    .. note::
        
        You can either provide:
        
        * **M_L**, **hz**, **Re** and **Ie**
        * **M_L**, **hz**, **Re**, **mag** and **offset**
            
        You can also provide **q0** instead of **hz** (which must be None in this case).
        
        The observed or intrinsic inclination must be given with **inc** or **inc0**, respectively, if one wants to correct the central surface brightness for the effect of finite thickness.
        
    :param hz: vertical scale height
    :type hz: int or float
    :param M_L: mass to light ratio
    :type M_L: int or float
    :param Re: half-light radius in the plane of symmetry
    :type Re: int or float
    
    :param float inc: (**Optional**) observed inclination (not corrected of the galaxy thickness)
    :param float inc0: (**Optional**) intrinsic inclination (corrected of the galaxy thickness)
            
    :param float Ie: (**Optional**) intensity at half-light radius if the galaxy was seen face-on (must correct for inclination effects)
    :param float mag: (**Optional**) galaxy total integrated magnitude used to compute **Ie** if not given
    :param float offset: (**Optional**) magnitude offset in the magnitude system used
    :param float q0: (**Optional**) axis ratio equal to hz/Rd with Rd the disk scale length. If this value is different from None, it overrides **hz**.
    :param str unit_Ie: (**Optional**) unit for the surface brightness
    :param str unit_M_L: (**Optional**) unit of the mass to light ratio
    :param str unit_Re: (**Optional**) unit for the scale radius **Re**
    '''
    
    def __init__(self, Re: Union[int, float], hz: Union[int, float], M_L: Union[int, float], 
                 q0: float = None, Ie: float = None, mag: float = None, offset: float = None, inc: float = None, inc0: float = None, 
                 unit_Re: str = 'kpc', unit_hz: str = 'kpc', unit_Ie: str = 'erg/(cm^2.s.A)', unit_M_L: str = 'solMass.s.A.cm^2/(erg.kpc^2)', **kwargs) -> None:
        
        MassModelBase.__init__(self, 3, M_L, unit_M_L)
        Sersic.__init__(       self, 1, Re, Ie=Ie, mag=mag, offset=offset, unit_Re=unit_Re, unit_Ie=unit_Ie)
        
        #: Disk scale length :math:`R_{\rm{d}} = R_{\rm{e}} / b_n`
        self.Rd         = self.Re/self.bn
        
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
            
        # Surface density computed in Sersic assumes no thickness so we must correct for this effect if the user provides an intrinsic inclination (see Mercier et al., in prep.)
        if inc0 is not None or inc is not None:
            self.I0     = morpho.correct_I0(self.I0, self.q0, inc=inc, inc0=inc0)
            self.Ie     = self.I0/np.exp(self.bn)
        
        
    ######################################################
    #            Methods (alphabetical order)            #
    ######################################################
    
    @property
    def _Rmax_razor_thin(self) -> QuantityType:
        '''Position of the maximum of the razor-thin disk curve'''
        
        return 2.2*self.Rd
    
    def _velocity_correction(self, R: Union[float, ndarray], *args, **kwargs) -> Union[float, QuantityType]:
        r'''
        Velocity correction necessary for the rotation curve
        
        .. math::
        
            V_{\rm{corr}} (r) = V_{\rm{c, max}} \times \sqrt{\frac{r~e^{1-r/R_{\rm{d}}}}{R_{\rm{d}}}},
        
        where :math:`R_{\rm{d}}` is the disk scale length, and :math:`V_{\rm{c, max}}` is the maximum of the velocity correction.

        :param R: radius where the velocity correction is computed
        :type R: float or ndarray[float]
        
        :returns: velocity correction
        :rtype: float or ndarray[float]
        '''
        
        R = self._checkR(R, self.Re)
        return self._Vmax_correction * np.sqrt(R * np.exp(1-R/self.Rd) / self.Rd)
    
    
    def _velocity_razor_thin(self, R: Union[float, ndarray], *args, **kwargs) -> Union[float, ndarray]:
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
    def _Vmax_correction(self) -> QuantityType:
        r'''
        Maximum of the velocity correction
        
        .. math ::
            
            V_{\rm{c, max}} = \sqrt{2\pi G h_z \Upsilon \Sigma_0 / \exp\lbrace 1 \rbrace},
            
        with :math:`\Sigma_0` the central surface brightness if the disk was seen face-on, and :math:`\Upsilon` the mass to light ratio.
        '''
        
        return np.sqrt(2*np.pi*G*self.hz*self.M_L*self.I0/np.exp(1))
    
    
    @property
    def _Vmax_razor_thin(self) -> QuantityType:
        '''
        Maximum velocity if the disk was razor thin.
        
        .. seealso:: :py:meth:`ExponentialDisk.velocity`
        '''
        
        disk_RT = ExponentialDisk(self.Re, self.M_L, Ie=self.Ie)
        return disk_RT.Vmax
        
        
    def profile(self, R: Union[int, float], z: Union[int, float], *args, **kwargs) -> float:
        r'''
        Light density profile at radius **R** and height **z**
        
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
    
    
    def velocity(self, R: Union[float, ndarray], *args, inner_correction: bool = False, **kwargs) -> QuantityType:
        r'''
        Velocity profile for a self-sustaining 3D double exponential disk against its own gravity through centripedal acceleration
        
        .. math::
        
            V(r) = \sqrt{V_{\rm{RT}}^2 (r) - V_{\rm{corr}}^2 (r)},
        
        where :math:`V_{\rm{RT}}` is the rotation curve of a similar razor-thin disk, and :math:`V_{\rm{corr}}` is the correction to apply to take into account the finite thickness of the disk.
        
        .. note::
            
            This formula is a `Bovy rotation curve <http://astro.utoronto.ca/~bovy/AST1420/notes-2017/notebooks/05.-Flattened-Mass-Distributions.html#Potentials-for-disks-with-finite-thickness>`_, i.e. an approximation for a thin (but non-zero thickness) disk.
                
            This rotation curve is ill-defined below a certain radius which depends on the disk thickness since the correction will become too important near the centre of the galaxy. 
            
            To correct for this effect, one can use the **inner_correction** optional parameter. This parameter will approximate the rotation curve in the inner parts. The approximation, called inner ramp, corresponds to the tangent which passes through the origin. This simply writes as
            
            .. math::
                
                V(r) = V_{\rm{d}} (R_0) \times r / R_0,
                
            where :math:`V_{\rm{d}}` is the Bovy rotation curve defined above and :math:`R_0` is the radius at which the tangent line passes through the origin. From personal derivations, this radius is the solution of the following equation
        
            .. math::

                x^2 \left [ I_1(x) K_0(x) - I_0(x) K_1(x) \right ] + x I_1(x) K_1(x) + q_0 (x + 0.5) e^{-2x} = 0,
            
            where :math:`I_n` and :math:`K_n` are the modified Bessel functions of first and second kind, respectively, and :math:`q_0` is the disk thickness.
        
            A numerical approximation, incorrect by less than 1% in the :math:`0 < q_0 < 1` range, is given by
        
            .. math::
            
                \log R / R_{\rm{d}} = 0.767 + 0.86 \times x - 0.14 \times x^2 - 0.023 \times x^3 + 0.005 \times x^4 + 0.001 \times x^5,
        
            where :math:`x = \log q_0`.
            
        :param R: radius where the velocity is computed
        :type R: float or ndarray[float]
        
        :param bool inner_correction: (**Optional**) whether to apply the correction in the inner parts (ramp approximation) or not
        
        :returns: velocity in :math:`\rm{km/s}`
        :rtype: Astropy Quantity
        
        :raises TypeError: if **R** is neither int, float, np.float16, np.float32, np.foat64 or ndarray
        '''
        
        R                 = self._checkR(R, self.Re)
        v_RT2             = self._velocity_razor_thin(R)**2 # Razor thin velocity squared
        v_corr2           = self._velocity_correction(R)**2 # Bovy approximation correction squared
        
        if isinstance(R.value, (int, float, np.float16, np.float32, np.float64)):
            
            if inner_correction and R < self._ramp_radius and self.q0 > 0:
                vr        = self.inner_ramp_approximation(R)
            elif v_RT2 < v_corr2:
                vr        = u.Quantity(0, unit='km/s')
            else:
                vr        = np.sqrt(v_RT2 - v_corr2)
            
        elif isinstance(R.value, np.ndarray):
            
            vr            = u.Quantity(np.zeros(np.shape(R)), unit='km/s')
            
            if inner_correction and self.q0 > 0:
                mask      = R >= self._ramp_radius
                vr[~mask] = self.inner_ramp_approximation(R[~mask])
            else:
                mask      = v_corr2 < v_RT2
                
            vr[mask]      = np.sqrt(v_RT2[mask] - v_corr2[mask])
        
        else:
            raise TypeError('R array has type %s but only float and numpy array are expected.' %type(R))
        
        return vr.to('km/s')
     
      
    #####################################################
    #       Inner parts linear ramp approximation       #
    #####################################################
    
    @property
    def _ramp_radius(self, *args, **kwargs) -> float:
        '''Compute the radius where the ramp approximation ends (same unit as Rd).'''
     
        lq0 = np.log(self.q0)
        return self.Rd * np.exp(0.767 + 0.86*lq0 - 0.14*lq0**2 - 0.023*lq0**3 + 0.005*lq0**4 + 0.001*lq0**5)
   
    @property
    def _ramp_slope(self, *args, **kwargs) -> float:
        '''Compute the slope for the ramp approximation.'''
       
        radius = self._ramp_radius
        return self.velocity(radius, inner_correction=False) / radius
    
    def inner_ramp_approximation(self, R: Union[float, ndarray], *args, **kwargs) -> Union[float, ndarray]:
        '''
        Inner ramp approximation. This gives the linear model which is tagent to the bovy rotation curve. It is used to correct the rotation curves in the inner parts where the bovy correction becomes larger than the razor thin disk velocity.

        :param R: radius where to compute the velocity
        :type R: float or ndarray[float]
      
        :returns: velocity profile for the inner ramp
        :rtype: float or ndarray[float]
      
        :raises TypeError: if **R** is neither int, float, np.float16, np.float32, np.foat64 or ndarray
        :raises ValueError: if **R** is larger than the ramp radius
        '''
      
        R = self._checkR(R, self.Re)
      
        if np.any(R >= self._ramp_radius):
            raise ValueError('At least one radius in %s is larger than the ramp radius %s' %(R, self._ramp_radius))
         
        return self._ramp_slope * R
    