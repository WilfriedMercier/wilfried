#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
.. codeauthor:: Riley Smith & Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>

A custom Self Organising Map which can write and read its values into and from an external file.

Most of the code comes from Riley Smith implementation found in `sklearn-som <https://pypi.org/project/sklearn-som/>`_ python library. 
"""

import numpy                   as     np
from   astropy.io              import fits
from   astropy.table           import Table
from   typing                  import Optional, Union, List, Tuple

class SOM():
    """
    .. codeauthor:: Riley Smith
    
    The 2-D, rectangular grid self-organizing map class using Numpy.
    
    :param int m: (**Optional**) shape along dimension 0 (vertical) of the SOM
    :param int n: (**Optional**) shape along dimesnion 1 (horizontal) of the SOM
    :param int dim: (**Optional**) dimensionality (number of features) of the input space
    :param float lr: (**Optional**) initial step size for updating the SOM weights.
    :param float sigma: (**Optional**) magnitude of change to each weight. Does not update over training (as does learning rate). Higher values mean more aggressive updates to weights.
    :param int max_iter: Optional parameter to stop training if you reach this many interation.
    :param int random_state: (**Optional**) integer seed to the random number generator for weight initialization. This will be used to create a new instance of Numpy's default random number generator (it will not call np.random.seed()). Specify an integer for deterministic results.
    """
    
    def __init__(self, m: int = 3, n: int = 3, dim: int = 3, lr: float = 1, sigma: float = 1, max_iter: int = 3000, random_state: Optional[int] = None) -> None:
        """
        .. codeauthor:: Riley Smith
        
        Init method.
        """
        
        # Initialize descriptive features of SOM
        self.m            = m
        self.n            = n
        self.dim          = dim
        self.shape        = (m, n)
        self.initial_lr   = lr
        self.lr           = lr
        self.sigma        = sigma
        self.max_iter     = max_iter
        
        # Physical parameters associated to each cell in the SOM
        self.phys         = {}

        # Initialize weights
        self.random_state = random_state
        rng               = np.random.default_rng(random_state)
        self.weights      = rng.normal(size=(m * n, dim))
        self._locations   = self._get_locations(m, n)

        # Set after fitting
        self._inertia     = None
        self._n_iter_     = None
        self._trained     = False

    def _get_locations(self, m: int, n: int) -> np.ndarray:
        """
        .. codeauthor:: Riley Smith
        
        Return the indices of an m by n array.
        
        :param int m: shape along dimension 0 (vertical) of the SOM
        :param int n: shape along dimension 1 (horizontal) of the SOM
        
        :returns: indices of the array
        :rtype: ndarray[int]
        """
        
        return np.argwhere(np.ones(shape=(m, n))).astype(np.int64)

    def _find_bmu(self, x: Union[List[float], np.ndarray]) -> int:
        """
        .. codeauthor:: Riley Smith
        
        Find the index of the best matching unit for the input vector x.
        
        :param x: input vector (1D)
        :type x: list or ndarray
        
        :returns: index of the best matching unit
        :rtype: int
        """
        
        diff     = self.weights-x
        distance = np.sum(diff*diff, axis=1)
        
        return np.argmin(distance)

    def step(self, x: Union[List[float], np.ndarray]) -> None:
        """
        .. codeauthor:: Riley Smith
        
        Do one step of training on the given input vector.
        
        :param x: input vector (1D)
        :type x: list or ndarray
        """
        
        import time

        # Get index of best matching unit
        bmu_index        = self._find_bmu(x)

        # Find location of best matching unit
        bmu_location     = self._locations[bmu_index, :]

        # Find square distance from each weight to the BMU
        beg = time.time()
        stacked_bmu      = np.stack([bmu_location]*(self.m*self.n), axis=0)
        bmu_distance     = np.sum(np.power(self._locations.astype(np.float64) - stacked_bmu.astype(np.float64), 2), axis=1)
        #print('dist', time.time()-beg)
        
        # Compute update neighborhood
        neighborhood     = np.exp((bmu_distance / (self.sigma ** 2)) * -1)
        local_step       = self.lr * neighborhood

        # Stack local step to be proper shape for update
        local_multiplier = np.stack([local_step]*(self.dim), axis=1)

        # Multiply by difference between input and weights
        delta            = local_multiplier * (x - self.weights)

        # Update weights
        self.weights    += delta
        
        return

    def _compute_point_intertia(self, x: Union[List[float], np.ndarray]) -> float:
        """
        .. codeauthor:: Riley Smith
        
        Compute the inertia of a single point. Inertia defined as squared distance from point to closest cluster center (BMU)
        
        :param x: input vector (1D)
        :type x: list or ndarray
        """
        # Find BMU
        bmu_index = self._find_bmu(x)
        bmu       = self.weights[bmu_index]
        
        # Compute sum of squared distance (just euclidean distance) from x to bmu
        inertia   = np.sum(np.square(x - bmu))
        
        return inertia

    def fit(self, X: np.ndarray, epochs: int = 1, shuffle: bool = True) -> None:
        """
        .. codeauthor:: Riley Smith
        
        Take data (a tensor of type float64) as input and fit the SOM to that data for the specified number of epochs.

        :param ndarray X: training data. Must have shape (n, self.dim) where n is the number of training samples.
        :param int epochs: (**Optional**) number of times to loop through the training data when fitting
        :param bool shuffle: (**Optional**) whether or not to randomize the order of train data when fitting. Can be seeded with np.random.seed() prior to calling fit.
        """
        
        import time
        
        # Count total number of iterations
        global_iter_counter          = 0
        n_samples                    = X.shape[0]
        total_iterations             = np.minimum(epochs * n_samples, self.max_iter)

        for epoch in range(epochs):
            
            # Break if past max number of iterations
            if global_iter_counter > self.max_iter:
                break

            if shuffle:
                rng                  = np.random.default_rng(self.random_state)
                indices              = rng.permutation(n_samples)
            else:
                indices              = np.arange(n_samples)

            # Train
            beg = time.time()
            for idx in indices:
                
                # Break if past max number of iterations
                if global_iter_counter > self.max_iter:
                    break
                
                # Do one step of training
                inp                  = X[idx]
                self.step(inp)
                
                
                # Update learning rate
                global_iter_counter += 1
                self.lr              = (1 - (global_iter_counter / total_iterations)) * self.initial_lr
                
            print(time.time()-beg)

        #beg = time.time()
        # Compute inertia
        inertia                      = np.sum(np.array([float(self._compute_point_intertia(x)) for x in X]))
        self._inertia_               = inertia

        # Set n_iter_ attribute
        self._n_iter_                = global_iter_counter

        # Set trained flag
        self._trained                = True
        #print(time.time()-beg)

        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        .. codeauthor:: Riley Smith
        
        Predict cluster for each element in X.

        :param ndarray X: training data. Must have shape (n, self.dim) where n is the number of training samples.

        :returns: an ndarray of shape (n,). The predicted cluster index for each item in X.
        :rtype: ndarray[int]
        
        :raises NotImplmentedError: if fit() method has not been called already
        """
        
        # Check to make sure SOM has been fit
        if not self._trained:
            raise NotImplementedError('SOM object has no predict() method until after calling fit().')

        # Make sure X has proper shape
        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'
        assert X.shape[1] == self.dim, f'This SOM has dimension {self.dim}. Received input with dimension {X.shape[1]}'

        labels = np.array([self._find_bmu(x) for x in X])
        return labels

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        .. codeauthor:: Riley Smith
        
        Transform the data X into cluster distance space.

        :param ndarray X: training data. Must have shape (n, self.dim) where n is the number of training samples.

        :returns: tansformed data of shape (n, self.n*self.m). The Euclidean distance from each item in X to each cluster center.
        :rtype: ndarray[float]
        """
        # Stack data and cluster centers
        X_stack       = np.stack([X]*(self.m*self.n), axis=1)
        cluster_stack = np.stack([self.weights]*X.shape[0], axis=0)

        # Compute difference
        diff          = X_stack - cluster_stack

        return np.linalg.norm(diff, axis=2)

    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        .. codeauthor:: Riley Smith
        
        Convenience method for calling fit(X) followed by predict(X).

        :param ndarray X: data of shape (n, self.dim). The data to fit and then predict.
        :param **kwargs: optional keyword arguments for the .fit() method

        :returns: ndarray of shape (n,). The index of the predicted cluster for each item in X (after fitting the SOM to the data in X).
        :rtype: ndarray[int]
        """
        # Fit to data
        self.fit(X, **kwargs)

        # Return predictions
        return self.predict(X)

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        .. codeauthor:: Riley Smith
        
        Convenience method for calling fit(X) followed by transform(X). Unlike in sklearn, this is not implemented more efficiently (the efficiency is the same as calling fit(X) directly followed by transform(X)).

        :param ndarray X: data of shape (n, self.dim) where n is the number of samples
        :param **kwargs: optional keyword arguments for the .fit() method

        :returns: ndarray of shape (n, self.m*self.n). The Euclidean distance from each item in X to each cluster center.
        :rtype: ndarray[float]
        """
        # Fit to data
        self.fit(X, **kwargs)

        # Return points in cluster distance space
        return self.transform(X)
    
    ######################################
    #             IO methods             #
    ######################################
    
    def read(self, fname: str, *args, **kwargs) -> None:
        '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Read the result of a SOM written into a FITS file with the .write() method.
        
        :param str fname: input file. Must be a FITS file.
        
        :raises TypeError: if **fname** is not of type str
        '''
        
        if not isinstance(fname, str):
            raise TypeError(f'fname is of type {type(fname)} but it must be of type str.')
            
        with fits.open(fname) as hdul:
            hdr   = hdul[1].header
            table = np.asarray(hdul[1].data)
            
        # Recover parameters required for the SOM
        self.m        = hdr['DIM1']
        self.n        = hdr['DIM2']
        self._n_iter_ = hdr['NITER']
        self.dim      = hdr['NFEAT']
        
        # List of fields corresponding to weights
        fields        = [f'F{pos}' for pos in range(self.dim)]
        
        # Extract weights
        self.weights  = np.asarray(table[fields].tolist())
        
        # Extract physical parameter
        for name in table.dtype.names:
            if name[:2] == 'P_':
                self.phys[name[2:]] = table[name]
        
        self._trained = True
        return
        
    
    def write(self, fname: str, colnames: Union[List[str], Tuple[str]] = [], **kwargs) -> None:
        '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Write the result of the SOM into a FITS file.
        
        :param str fname: output filename. Output file will always be a FITS file.
        
        :param colnames: list of column names to add into the header of the VOtable
        :type colnames: list[str] or tuple[str]
        :param **kwargs: (**Optional**) additional columns to add to the table
        
        :raises NotImplementedError: if the fit method has not been used yet
        :raises TypeError: if **fname** is not of type str or **colnames** is neither a list nor a tuple
        '''
        
        # Check to make sure SOM has been fit
        if not self._trained:
            raise NotImplementedError('SOM object has no write() method until after calling fit().')
            
        if not isinstance(fname, str):
            raise TypeError(f'fname is of type {type(fname)} but it must be of type str.')
            
        if not isinstance(colnames, (list, tuple)):
            raise TypeError(f'colnames is of type {type(colnames)} but it must be of type list.')
        
        # Create an empty astropy Table
        table                = Table()
        
        # Add features
        for pos, weight in enumerate(self.weights.T):
            table[f'F{pos}'] = weight
        
        # Add physical parameters
        for key, value in kwargs.items():
            table[f'P_{key.upper()}'] = value
            
        # Header to add into the FITS object
        hdr = fits.Header({'DIM1'   : self.m,
                           'DIM2'   : self.n,
                           'NITER'  : self.n_iter_,
                           'NFEAT'  : self.dim
                         })
        
        # Add column names into header for information purposes only
        for pos, col in enumerate(colnames):
            hdr[f'COL{pos}'] = col
            
        # Convert to a FITS object and write
        hdu0    = fits.PrimaryHDU()
        hdu1    = fits.BinTableHDU(data=table, header=hdr, name='SOM')
        hdulist = fits.HDUList([hdu0, hdu1])
        hdulist.writeto(fname, overwrite=True)
        
        return
    
    #################################################
    #          Physical parameters methods          #
    #################################################
    
    def get(self, param: str) -> np.ndarray:
        '''
        .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu> 
        
        Return the given physical parameters if it exists.
        
        :param str param: parameter to return
        :returns: array of physical parameter value associated to each node
        :rtype: ndarray
        
        :raises KeyError: if **param** is not found
        '''
        
        if param not in self.phys:
            raise KeyError(f'physical parameter {param} not found.')
            
        return self.phys[param]
    
    ##########################################
    #               Properties               #
    ##########################################

    @property
    def cluster_centers_(self) -> np.ndarray:
        '''
        .. codeauthor:: Riley Smith
        
        Give the coordinates of each cluster centre as an array of shape (m, n, dim).
        
        :returns: cluster centres
        :rtype: ndarray[int]
        '''
        
        return self.weights.reshape(self.m, self.n, self.dim)

    @property
    def inertia_(self) -> np.ndarray:
        '''
        .. codeauthor:: Riley Smith
        
        Inertia.
        
        :returns: computed inertia
        :rtype: ndarray[float]
        
        :raises AttributeError: if the SOM does not have the inertia already computed
        '''
        
        if self._inertia_ is None:
            raise AttributeError('SOM does not have inertia until after calling fit()')
            
        return self._inertia_

    @property
    def n_iter_(self) -> int:
        '''
        .. codeauthor:: Riley Smith
        
        Number of iterations.
        
        :returns: number of iterations
        :rtype: int
        
        :rtype AttributeError: if the number of iterations is not initialised yet
        '''
        if self._n_iter_ is None:
            raise AttributeError('SOM does not have n_iter_ attribute until after calling fit()')
            
        return self._n_iter_

