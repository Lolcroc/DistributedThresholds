# Python
from collections import defaultdict
from pathlib import Path
import json
import re
import warnings
import copy

# Math
import numpy as np
import sympy as sp

# Manticore
from manticore.utils import Parameterized, parameters_bound

class ErrorSet(Parameterized):
    """A set of correlated errors under some probabilities in a Crystal.
    
    This object is essentially an ErrorKernel, placed at the correct position in the Crystal object.
    Error configurations with the same probabilities are grouped for faster sampling in the location attributes.
    For both errors and erasures, each row correspond to a distinct location of a configuration.
    """
    def __init__(self, probabilities, error_locations, erasure_locations=None, indices=None, name=None, check_normalized=True, expand=True, float_tol=1e-6):
        """Create a new set of errors.

        Args:
            probabilities (list): Probabilities for each of the error configurations. The ordering with respect to errors and
                erasures is implicit; the n-th entry of this vector forms a bitmask bin(indices[n]) for each row in error_locations
                and erasure_locations. More details given below.
            indices (list): Integer representations of the locations to sample. Defaults to None. More details given below.
            error_locations (list): A list-of-lists, each row describing the locations of a correlated error.
            erasure_locations (list): A list-of-lists, each row describing the locations of a (correlated) erasure.

                Error and erasure locations are sampled according to the bitmask bin(indices[n]) with probability probabilities[n].
                Erasures are always MSB (left-most bits), whereas errors are always LSB (right-most bits).

                Examples:
                    * probabilities=[p0, p1, p2, p3, p4, p5, p6, p7], erasure_locations=[[5]], error_locations=[[1, 2]] 
                      # Corresponds to the following table

                      erasures | errors | idx | probability
                          None |   None | 000 | p0
                          None |      2 | 001 | p1
                          None |      1 | 010 | p2
                          None |   1, 2 | 011 | p3
                             5 |   None | 100 | p4
                             5 |      2 | 101 | p5
                             5 |      1 | 110 | p6
                             5 |   1, 2 | 111 | p7

                    * probabilities=[p0, p1, p4, p7], indices=[0, 1, 4, 7], error_locations=[[1, 2, 3]]
                      # Corresponds to the following table

                       errors | idx | probability
                         None | 000 | p0
                            3 | 001 | p1
                            1 | 100 | p4
                      1, 2, 3 | 111 | p7
                
            name (str, optional): Name of this error set. Defaults to None.
            check_normalized (bool, optional): Check if the probabilities sum to unity. Default to True.
            expand (bool, optional): Expand the probabilities using sp.expand. Only relevant for parameterized probabilities.
                Defaults to True.
            float_tol (float, optional): Maximum tolerable error for the sum of probabilities away from 1. Defaults to 1e-6.

        Raises:
            ValueError: If probabilities are not normalized.
        """
        super().__init__()  # For parameterized
        self.name = name or ""

        # Initialize error locations
        self.error_locations = _validate_locations("Error locations", error_locations)
        M, P = self.error_locations.shape

        # Initialize erasure locations
        if erasure_locations:
            self.erasure_locations = _validate_locations("Erasure locations", erasure_locations)
        else:
            self.erasure_locations = np.empty((M, 0), dtype=self.error_locations.dtype)
        N, Q = self.erasure_locations.shape

        if M != N:
            raise ValueError(f"Erasure locations len={M} and error locations len={N} are different")

        # Initialize the indices vector
        self.indices = np.asarray(indices) if indices is not None else np.arange(2**self.num_bits)

        # Pad the probability vector uniformly based on missing entries w.r.t. given indices
        missing = len(self.indices) - len(probabilities) # Number of missing entries in the probability vector
        if missing > 0:
            if probabilities.pop() != "*":
                raise ValueError(f"Badly formatted {probabilities}")

            missing += 1 # Fix the extra slot opened with pop() above
            p0 = sp.sympify(probabilities)
            probabilities = p0 + missing * [(1 - sum(p0)) / missing]
        elif missing == 0:
            probabilities = sp.sympify(probabilities)
        else:
            raise ValueError(f"Too many probabilities, found len={len(probabilities)} but expected len={len(self.indices)}")
        
        # Expand probabilities if flag given
        if expand:
            probabilities = np.asarray([sp.expand(p) for p in probabilities])
        
        # Store probabilities as a numpy object array
        self.probabilities = np.asarray(probabilities)

        if check_normalized and abs((s := sp.simplify(sum(self.probabilities))) - 1) > float_tol:
            raise ValueError(f"Probabilities {self.probabilities} not normalized, found sum={s}")

        # The order is important! Errors are LSB (right), erasures are MSB (left)
        self._bit_mask = _bit_mask(self.indices, erasures=Q, errors=P)

    # TODO bind ErrorSampler to a single Process. Get incoming requests of size A and distribute A samples using the size=(A, M) parameter
    @parameters_bound
    def sample(self, rng):
        mask = rng.choice(self._bit_mask, size=len(self.error_locations), p=self.probabilities)
        
        errors = self.error_locations[mask["errors"]]
        erasures = self.erasure_locations[mask["erasures"]]

        return errors, erasures

    @property
    def parameters(self):
        return set() if self.probabilities.dtype == float else set.union(*(p.free_symbols for p in self.probabilities))

    def copy(self):
        return copy.deepcopy(self)

    def _bind_parameters(self, **parameters):
        """Bind parameters and evaluate the probabilities accordingly.
        
        Args:
            **parameters: Parameter names and corresponding values.
        """
        self.probabilities = np.array([p.subs(parameters) for p in self.probabilities])

    def _finalize_parameters(self):
        self.probabilities = np.asarray(self.probabilities, dtype=float)

        if not all(self.probabilities >= 0):
            raise RuntimeError(f"ErrorSet {self.name} probabilities {self.probabilities} are not all positive. Please check your parameters")

    @property
    def num_error_bits(self) -> int:
        return self.error_locations.shape[1]

    @property
    def num_erasure_bits(self) -> int:
        return self.erasure_locations.shape[1]

    @property
    def num_bits(self) -> int:
        return self.num_error_bits + self.num_erasure_bits

    @property
    def _prob_rep(self):
        # if self.parameters:
        #     func = lambda p: f"{p}"
        # else:
        #     func = lambda p: f"{p:.2e}"

        return f"[{', '.join(str(p) for p in self.probabilities[:self.num_bits])}, ...]"

    @property
    def _indices_rep(self):
        return f"[{', '.join(str(i) for i in self.indices[:self.num_bits])}, ...]"

    def __repr__(self):
        return f"""{self.__class__.__name__}(name={self.name}, parameters={self.parameters}, indices={self._indices_rep},
            probabilities={self._prob_rep}, error_locations={_loc_rep(self.error_locations)}, 
            erasure_locations={_loc_rep(self.erasure_locations)})"""

class ErrorKernel(ErrorSet):
    """A local representation of errors, without referring the size of the lattice.

    This object is a generalization of a superoperator, because it may describe errors on arbitrary locations (such as measurement errors).
    """
    def __init__(self, probabilities, error_locations, condition=None, **kwargs):
        """Create a new 'kernel'. For a list of **kwargs, see ErrorSet.

        Args:
            condition (str, optional): A boolean expression returning True if the kernel should be applied at some lattice position.
                This statement is evaluated using the eval() built-in, using variables from the Lattice graph-dict and node-dict.
                These include e.g. the lattice sizes (X, Y, Z) in the graph-dict and lattice point positions (x, y, z) in the node-dict. 

                Examples:
                    * condition = 'x == 0 and y != 0' # Will apply kernel only to the first row (x) and all but the first column (y)
                    * condition = 'x != X - 1' # Will apply kernel to all but the last row
                    * condition = 'x == y' # Will apply kernel only along the diagonal slice where row and column are the same
                
                Defaults to None, applying the kernel for all lattice points.
        """
        self._condition = condition

        super().__init__(probabilities, error_locations, **kwargs)

    def build(self, lattice, node_index, **kwargs):
        """Build an ErrorSet from a kernel tiled along a lattice, taking into account the kernel condition.

        Args:
            lattice (Lattice): The Bravais lattice.

        Returns:
            ErrorSet: All possible error combinations of the error kernel across the lattice points.
        """
        error_locations, erasure_locations = [], []

        def add_location(locations, kernel_locations, index_func):
            for ker_locs in kernel_locations:
                locs = []

                for quotient_element in ker_locs:
                    element, *offset = re.split("[:,]", quotient_element)
                    offset = tuple(int(i) for i in offset) if offset else None
                    locs.append(index_func(element, offset))

                locations.append(locs)

        for ker_locs in self.error_locations:
            for quotient_element in ker_locs:
                element, *offset = re.split("[:,]", quotient_element)
                if offset:
                    lattice._load_offsets(tuple(int(i) for i in offset))

        # Is this really necessary?
        for ker_locs in self.erasure_locations:
            for quotient_element in ker_locs:
                element, *offset = re.split("[:,]", quotient_element)
                if offset:
                    lattice._load_offsets(tuple(int(i) for i in offset))

        for i, dd in lattice.nodes(data=True):
            if eval(self.condition, {"__builtins__": None}, dd | lattice.graph):
                neighbours = {offset: j for _, j, offset in lattice.edges(i, keys=True)}
                ix = lambda e, o: node_index[(e, neighbours[o] if o else i)]
                
                add_location(error_locations, self.error_locations, ix)
                add_location(erasure_locations, self.erasure_locations, ix)

        return ErrorSet(probabilities=self.probabilities, error_locations=error_locations, erasure_locations=erasure_locations, indices=self.indices, name=self.name)
    
    def sample(self, rng):
        raise NotImplementedError(f"{self.__class__.__name__} cannot be sampled")

    def mergeable(self, other):
        return (isinstance(other, ErrorKernel)
            and self.condition == other.condition 
            and self.num_error_bits == other.num_error_bits
            and self.num_erasure_bits == other.num_erasure_bits
            and np.array_equal(self.indices, other.indices)
            and np.array_equal(self.probabilities, other.probabilities)
        )

    def __iadd__(self, other):
        if not self.mergeable(other):
            raise ValueError(f"{other} cannot be added to {self}")

        if self.name and other.name:
            self.name = f"{self.name},{other.name}"  # Combine both names
        else:
            self.name = self.name or other.name  # Pick the non-empty name

        self.error_locations = np.append(self.error_locations, other.error_locations, axis=0)
        self.erasure_locations = np.append(self.erasure_locations, other.erasure_locations, axis=0)

    @property
    def condition(self):
        return self._condition or "True"

    def __repr__(self):
        return f"""{self.__class__.__name__}(name={self.name}, condition={self._condition}, parameters={self.parameters}, probabilities={self._prob_rep}, 
            error_locations={_loc_rep(self.error_locations)},  erasure_locations={_loc_rep(self.erasure_locations)})"""

class ErrorSampler(Parameterized):
    """A sampler is a collection of all error sets for a some Crystal."""
    def __init__(self, name=None, errors=None, **kwargs):
        """Create an error set directly.

        Args:
            name (str, optional): Name of this error set. Defaults to None.
            errors (list, optional): A list of all the ErrorSets in this sampler. Defaults to None.
        """
        super().__init__()  # For parameterized
        self.name = name or ""
        self.errors = errors or []

    @classmethod
    def from_json(cls, path, **kwargs):
        path = Path(path)

        with open(path) as f:
            attr = json.load(f)
            
            return cls(name=path.stem, errors=[ErrorSet(**e) for e in attr["errors"]])

    def to_json(self, path=None):
        path = path or f"{self.name}.json"
        with open(path, "w") as f:
            json.dump(self, f, cls=JSONEncoder)

    @property
    def parameters(self):
        return set.union(*(e.parameters for e in self.errors))

    def copy(self):
        return copy.deepcopy(self)

    def _bind_parameters(self, **parameters):
        for e in self.errors:
            e.bind_parameters(inplace=True, **parameters)  # Deepcopy already does copies of contained objects

    def _finalize_parameters(self):
        for e in self.errors:
            e.finalize_parameters()
        
    @parameters_bound
    def sample(self, rng):
        errors, erasures = zip(*(e.sample(rng) for e in self.errors))
        
        return _merge_samples(errors), _merge_samples(erasures)

    def __iter__(self):
        return iter(self.errors)

    def __len__(self):
        return len(self.errors)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, errors={self.errors})"

class ErrorModel(ErrorSampler):
    def __init__(self, name=None, errors=None, merge_errors=True, **kwargs):
        super().__init__(name, errors, **kwargs)

        if merge_errors:
            self.merge_errors()

    """An error model is a collection of error kernels"""
    def build(self, lattice, node_index, **kwargs):
        """Build a sampler from an error model and a Lattice object.

        This function builds all of the error sets from the kernels contained in the error model.

        Args:
            lattice (Lattice): The Bravais lattice.

        Returns:
            ErrorSampler: The complete description of errors in a crystal.
        """
        name = self.name + "_" + "_".join(str(i) for i in lattice.reps)
        errors = [kernel.build(lattice, node_index) for kernel in self.errors]
        
        return ErrorSampler(name=name, errors=errors, **kwargs)

    @classmethod
    def from_json(cls, path, **kwargs):
        path = Path(path)
        
        with open(path) as f:
            attr = json.load(f)
            
            return cls(name=path.stem, errors=[ErrorKernel(**e) for e in attr["errors"]])

    def __iadd__(self, other):
        if isinstance(other, ErrorModel):
            for kernel in other:
                self._append_kernel(kernel)
        elif isinstance(other, ErrorKernel):
            self._append_kernel(other)
        else:
            raise ValueError(f"cannot add type {type(other)} to {self}")

    def _append_kernel(self, other: ErrorKernel):
        for kernel in self:
            try:
                kernel += other
                break  # We managed to merge the kernel
            except ValueError:
                pass
        else:  # Executes if for loop completes normally
            self.errors.append(other)

    def merge_errors(self):
        current_errors = self.errors
        self.errors = []

        for kernel in current_errors:
            self._append_kernel(kernel)

class JSONEncoder(json.JSONEncoder):
    """An encoder for the above objects to JSON format"""
    def default(self, obj):
        if isinstance(obj, ErrorSampler):
            return dict(errors=obj.errors)
        if isinstance(obj, ErrorSet):
            out = dict(name=obj.name, probabilities=obj.probabilities, error_locations=obj.error_locations)
            if obj.erasure_locations.size > 0:
                out["erasure_locations"] = obj.erasure_locations
            if len(obj.indices) < 2**obj.num_bits:
                out["indices"] = obj.indices
            return out
        if isinstance(obj, sp.Float):
            return float(obj)
        if isinstance(obj, sp.Expr):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def _bit_mask(indices, **kwargs: int):
    """A helper function for creating bitmasks of natural numbers during error sampling.

    Bitmask are output as a structured numpy array, with individual columns indexed by keyword argument names.

    Args:
        indices (np.ndarray): Integer indices to convert to a binary mask.
        **kwargs (int): (name=i) pairs for each set of i columns in the bitmask. The ordering is important!

        Examples:
            * _bit_mask(np.arange(8), a=2, b=1) # Will return bitmask such that

                bitmask["a"] = array([[False, False],
                                    [False, False],
                                    [False,  True],
                                    [False,  True],
                                    [ True, False],
                                    [ True, False],
                                    [ True,  True],
                                    [ True,  True]])

                bitmask["b"] = array([[False],
                                    [ True],
                                    [False],
                                    [ True],
                                    [False],
                                    [ True],
                                    [False],
                                    [ True]])

                The 'full' mask may be recovered by placing mask["a"] and mask["b"] next to each other.

    Returns:
        np.ndarray: A bitmask where bitmask["name"] contains the corresponding i columns of a bitmask with row n
            equal to the binary representation of indices[n].
    """
    # Construct the raw mask by converting indices to binary representations
    n = sum(kwargs.values())
    raw_mask = (np.right_shift.outer(indices, np.arange(n-1, -1, -1)) & 1).astype(bool)

    # Split the raw mask column-wise as a structured array according to the kwargs
    dt = [(name, bool, (reps,)) for name, reps in kwargs.items()]
    return raw_mask.view(dt).squeeze()

def _merge_samples(samples):
    values, counts = np.unique(np.concatenate(samples), return_counts=True)
    return set(values[(counts & 1).astype(bool)])

def _validate_locations(name, locations):
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            arr = np.asarray(locations)
            if arr.ndim != 2:
                raise TypeError(f"{name} {locations} not 2-dimensional")
            return arr
        except np.VisibleDeprecationWarning:
            raise ValueError(f"Locations {locations} have different lengths")

def _loc_rep(locs):
    return f"[{locs[0]}{', ...' if len(locs) > 1 else ''}]"
