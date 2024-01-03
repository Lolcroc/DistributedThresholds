import copy
import functools
import time
from abc import ABC, abstractmethod
from typing import Callable

def parameters_bound(func: Callable) -> Callable:
    """A decorator for methods that require parameters to be bound and finalized.

    Examples are sampling in parameterized error models, or parameterized lattices.

    Args:
        func (Callable): The method to decorate.

    Returns:
        Callable: A wrapper for func that ensures all its parameters are bound and finalized.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._bound:
            self.finalize_parameters()
        return func(self, *args, **kwargs)
    return wrapper

def time_method(timings: dict, name=None) -> Callable:
    """A decorator to keep track of execution time on a certain function.

    Args:
        timings (dict): A dictionary whose name field will keep track of time
        name (str, optional): The name to track function calls on. Defaults to None.

    Returns:
        Callable: A decorator that may be invoked on any function to keep track of its time.
    """
    def decorator(func: Callable) -> Callable:
        label = name or func.__name__
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tic = time.time_ns()
            result = func(*args, **kwargs)
            toc = time.time_ns()

            timings[label] += toc - tic
            return result
        return wrapper
    return decorator

def count_method(counts: dict, name=None) -> Callable:
    """A decorator to keep track of execution counts on a certain function.

    Args:
        counts (dict): A dictionary whose name field will keep track of counts
        name (str, optional): The name to track function calls on. Defaults to None.

    Returns:
        Callable: A decorator that may be invoked on any function to keep track of its counts.
    """
    def decorator(func: Callable) -> Callable:
        label = name or func.__name__
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            counts[label] += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator

def count_generator(counts: dict, name=None) -> Callable:
    def decorator(func: Callable) -> Callable:
        label = name or func.__name__
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for out in func(*args, **kwargs):
                counts[label] += 1
                yield out
        return wrapper
    return decorator

def count_lens(counts: dict, name=None) -> Callable:
    def decorator(func: Callable) -> Callable:
        label = name or func.__name__
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            counts[label] += len(out)
            return out
        return wrapper
    return decorator

class Parameterized(ABC):
    """Mixin for classes with parameters.
    
    Note that in order for this mixin to work properly with super() calls, it should appear
    first in the class hierarchy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bound = False

    @property
    @abstractmethod
    def parameters(self) -> set:
        """Return a set of parameters of this object."""
        pass

    @abstractmethod
    def copy(self):
        """Return a deepcopy of this object."""
        pass

    @abstractmethod
    def _bind_parameters(self, **parameters):
        """Bind parameters by name and value pairs.
        
        This function should not be called directly. Instead, use the version without underscore."""
        pass

    @abstractmethod
    def _finalize_parameters(self):
        """Perform any cleanups necessary after all parameters are bound.
        
        This method is called lazily and automatically whenever the caller is decorated
        with @parameters_bound. 
    
        This function should not be called directly. Instead, use the version without underscore."""
        pass

    def bind_parameters(self, inplace=False, **parameters):
        if self._bound:
            raise RuntimeError(f"{self} has already bound all its parameters")

        obj = self if inplace else self.copy()
        obj._bind_parameters(**parameters)

        # Optional unless users make sure that every attribute access to parameterized objects is
        # annotated with @parameters_bound. Will finalize parameters greedily in addition to lazily.
        if not obj.parameters:
            obj.finalize_parameters()

        return obj

    def finalize_parameters(self):
        if params := self.parameters:
            raise RuntimeError(f"{self} has unbound paramaters {params}")

        self._finalize_parameters()
        self._bound = True
