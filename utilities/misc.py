from   ipywidgets      import IntProgress
from   IPython.display import display, clear_output
from   copy            import deepcopy
from   collections.abc import Iterator


class ProgressBar:
    r'''
    .. codeauthor:: Wilfried Mercier - IRAP <wilfried.mercier@irap.omp.eu>
    
    Implements an easy to use progressbar for Jupyter notebooks.
    
    .. note::
        
        A description can be provided with the keyword 'description'. Three custom descriptors can be used to have a dynamical description:
        
        * {index} which prints the current index in the loop
        * {len} which prints the total length of the loop
        * {fraction} which prints the current fraction of the loop
        
        For instance, :code:`description={index}/{len} ({fraction})` will print something like :code:`11/1000 (1.1%)`.
    
    :param args: arguments passed to ipywidgets.IntProgress
    :param kwargs: keyword arguments passed to ipywidgets.IntProgress
    '''
    
    def __init__(self, *args, **kwargs) -> None:
        r'''Init method.'''
        
        self.args   = args
        self.kwargs = kwargs
        
    def __call__(self, arr: Iterator):
        '''Method called when the object is associated to a list or array the loop will be performed unto.'''
        
        #: Lenth of the list or array associated to the progress bar
        self.len    = len(arr)
        
        return self
        
    def __iter__(self):
        '''Method called before looping.'''
        
        #: Index during the loop
        self.index    = 0
        
        #: Position along the loop (index - 1)
        self.pos      = 0
        
        #: Fraction of loop
        self.fraction = 0
        return self
    
    def __next__(self):
        '''Method at each loop iteration.'''
        
        # When reaching the end, we stop the iteration
        if self.index == self.len:
            raise StopIteration
            
        # Update index and fraction
        self.index   += 1
        self.fraction = self.index/self.len
            
        # Copy entirely the kwargs to avoid overwriting the original ones
        kwargs                    = deepcopy(self.kwargs)
            
        # If some specific keywords are found we modify them accordingly
        if 'description' in kwargs:
            kwargs['description'] = kwargs['description'].replace('{index}', str(self.index)).replace('{len}', str(self.len)).replace('{fraction}', f'{self.fraction:.1%}')
           
        # Implement a custom style if no style is provided by the user
        if 'style' not in kwargs:
            kwargs['style']       = {}
        
        if 'bar_color' not in kwargs['style']:
            if self.fraction < 0.33:
                kwargs['style']['bar_color'] = 'firebrick'
            elif self.fraction > 0.66:
                kwargs['style']['bar_color'] = 'darkgreen'
            else:
                kwargs['style']['bar_color'] = 'orange'

        # Clear and update the progress bar
        clear_output(wait=True)
        display(IntProgress(value=self.index, min=0, max=self.len, *self.args, **kwargs))
        
        self.pos += 1
          
        return self