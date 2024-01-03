# Python
from __future__ import annotations
from typing import Any
import copy
from functools import lru_cache, reduce
from itertools import product, repeat
from collections import defaultdict
# from more_itertools import pairwise

# Math
import numpy as np
import sympy as sp
import networkx as nx
import galois as gl
GF2 = gl.GF(2)

# Manticore
from manticore.utils import Parameterized, parameters_bound

CARTESIAN_LABELS = "xyz"

@lru_cache(maxsize=None)
def binary_range(n):
    return (np.right_shift.outer(np.arange(2**n), np.arange(n-1, -1, -1)) & 1).astype(bool)

def dual_friendly(func):
    def wrapper(self, **attr):
        if self._dual and "dim" in attr:
            attr["dim"] = self.dim - attr["dim"]
        return func(self, **attr)
    return wrapper

class GraphUtils:
    """Utility tools for networkx Graphs.

    This class is meant to be subclassed.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Forwards to other super()
        self._dual = False

    @property
    def dim(self):
        return self.graph["dim"]
    
    @property
    def dual(self):
        G = copy.copy(self) # Shallow copy

        # G._node = self._node # Unnecessary when doing copy(self)
        G._succ, G._pred = self._pred, self._succ
        G._adj = self._pred

        G._dual = not self._dual
        return G

    def check_dim(self, dim):
        if dim < 0 or dim > self.dim:
            raise ValueError(f"Dimension {dim} not in {self.dim}")

    def check_len(self, vec):
        if len(vec) != self.dim:
            raise ValueError(f"{vec} incorrect dim {self.dim}")

    @dual_friendly
    def filter_nodes(self, **attr):
        """Return a generator over nodes whose data contain input attributes."""
        return (n for n, dd in self.nodes(data=True) if attr.items() <= dd.items())

    # @dual_friendly  # Would be doing it twice
    def filter_edges_from(self, **attr):
        if keys := attr.pop('keys', None):
            return self.edges(self.filter_nodes(**attr), keys=keys)
        else:
            return self.edges(self.filter_nodes(**attr))

    def incidence(self, nodes):
        """TODO might be super slow."""
        inc = set()
        for n in nodes:
            inc ^= set(self.predecessors(n))
        return inc

    def boundary(self, nodes):
        """TODO might be super slow."""
        bound = set()
        for n in nodes:
            bound ^= set(self.successors(n))
        return bound

    # def filter_edges(self, **attr):
    #     """Return a generator over edges whose data contain input attributes."""
    #     return ((u, v) for u, v, dd in self.edges(data=True) if attr.items() <= dd.items())

class UnitCell(GraphUtils, nx.MultiDiGraph):
    """A graph description of the unit cell chain complex from which the crystal complex is built.
    
    The unit cell is a labeled quotient graph of the structure to be built under equivalence of elements in different unit cells. The graph edges are labeled with the offset in the Bravais lattice.
    """

    def __init__(self, name, dim, **attr):
        super().__init__(name=name, dim=dim, **attr)

    def build(self, lattice, node_index, **attr):
        """Builds a crystal from a `UnitCell` and a Bravais `Lattice` object.
        
        The quotient edges of the unit cell are 'factored out' using the lattice according to the `offset` values on edges.
        That is, edges between elements in the crystal are connected only when the 'offset' data in the unit cell and
        the lattice graph match.

        Args:
            lattice (Lattice): Graph description of the Bravais lattice.

        Raises:
            ValueError: If dimensions between inputs don't match
        """
        if self.dim != lattice.dim:
            raise ValueError(f"Unit cell dim {self.dim} not same as lattice {lattice.dim}")

        crystal = Crystal(f"{self.name}_{lattice.name}", self.dim, **attr)

        # Adds the unit cell for each lattice point (a product of both graph's nodes)
        for u, x in product(self, lattice):
            i = node_index[(u, x)]
            crystal.add_node(i, uc=u, pos=x, **self.nodes[u], **lattice.nodes[x])

        # Add edges of the crystal according to the offset values
        for u, v, o, dd in self.edges(keys=True, data=True):
            for x, y in lattice.filter_offsets(o):
                i, j = node_index[(u, x)], node_index[(v, y)]

                if crystal.has_edge(i, j):
                    crystal.remove_edge(i, j)
                else:
                    crystal.add_edge(i, j, **dd)

        return crystal

    def add_elem(self, label, dim, **attr):
        """Add a new element in the chain complex.

        Elements are added as nodes in the Graph.

        Args:
            label (str): A name for the element.
            dim (int): Dimension where the element is added.
            **attr: Additional data added to the Graph node.

        Raises:
            ValueError: If `dim` not in the current complex.
            ValueError: If `label` already exists.
        """
        if self._dual:
            dim = self.dim - dim

        self.check_dim(dim)

        if self.has_node(label):
            raise ValueError(f"Node {label} already in {self.nodes()}")
            
        self.add_node(label, dim=dim, **attr)

    def add_bound(self, u, v, offset=None, **attr):
        """Add element `v` to the boundary of element `u`.

        Boundaries are added as directed edges u -> v in the Graph.

        Args:
            u (str): Name of the upper element getting a boundary.
            v (str): Name of the lower element added as a boundary.
            **attr: Additional data added to the Graph edge.

        Raises:
            ValueError: If `offset` does not have the same dimension as the complex.
            ValueError: If `u` or `v` do not exist.
            ValueError: If `u` is not 1 dimension above `v`.
        """
        if offset is None:
            offset = self.dim * (0,)
        self.check_len(offset)

        for n in (u, v):
            if not self.has_node(n):
                raise ValueError(f"Node {n} not in {self.nodes()}")

        udim, vdim = self.nodes[u]['dim'], self.nodes[v]['dim']
        if (2 * self._dual - 1) * (vdim - udim) != 1:
            raise ValueError(f"Node {u} not 1 dimension higher than {v}")
        
        self.add_edge(u, v, offset, **attr)

    def color(self, *args, **kwargs):
        raise NotImplementedError(f"{self} cannot be colored, please overwrite this method")

    def verify_coloring(self):
        attached_colors = defaultdict(set)
        colors = nx.get_edge_attributes(self, "color")

        for arc in self.filter_edges_from(dim=2, keys=True):
            face, edge, offset = arc
            try:
                color = colors[arc]
            except KeyError:
                raise ValueError(f"{self} not fully colored, missing color on {arc}")

            for element in (face, edge):
                if color in attached_colors[element]:
                    raise ValueError(f"{self} not properly colored, {element} has double attached color {color}")
                else:
                    attached_colors[element].add(color)

    # OLD For an n-split (vertex Nickerson splitting):
    #
    # 1. Choose n subsets S_i (i=1..n), where S_0 = S \ union_i(S_i) and S is all incident edges
    # 2. Create n new vertices v_i, with v_i having coboundaries S_i. The central (pivot) vertex v=v_0
    #    has coboundaries S_0
    # 3. Create n new edges e_i with boundaries v_0 and v_i (offset 0)
    # 4. Fix the remaining boundaries: for each v_i, calculate the coboundary delta_2 delta_1 v_i
    #    and connect the resulting {(f_j, offset_j)} tuples to e_i with offset_j.
    def n_split(self, element: Any, *args: tuple[Any, Any, tuple[Any, tuple[int, ...]]]):
        """Perform an n-split of element, with n = len(args).

        Args:
            element (Any): Name of the element being split (e.g. a face f).
            *args: Triplets of (new_element, new_connector, boundary) for each split that is performed.
                For more details on each individual triplet, see the other method simple_split.

        Raises:
            ValueError: If element does not exist.
            ValueError: If element has dimension zero.
        """
        if not self.has_node(element):
            raise ValueError(f"element {element} does not exist")

        dim = self.nodes[element]["dim"]
        if self._dual:
            dim = self.dim - dim

        if dim == 0:
            raise ValueError(f"cannot split 0-dimensional element {element}")

        for new_element, new_connector, boundary in args:
            parent = self.nodes[element].get("split_source", element)  # Flattens the split source pointer tree
            self.add_elem(new_element, dim, split_source=parent)
            self.add_elem(new_connector, dim - 1, split_connector=parent)

            self.add_bound(element, new_connector)
            self.add_bound(new_element, new_connector)

            fix = set()

            # Modify boundary_{dim} operator
            for boundary_element, offset in boundary:
                data = self.get_edge_data(element, boundary_element, offset)
                self.remove_edge(element, boundary_element, offset)
                self.add_bound(new_element, boundary_element, offset, **data)

                for _, superboundary_element, super_offset in self.edges(boundary_element, keys=True):
                    tot_offset = tuple(a + b for a, b in zip(offset, super_offset))
                    fix ^= {(superboundary_element, tot_offset)}

            # Modify boundary_{dim-1} operator
            for superboundary_element, offset in fix:
                self.add_bound(new_connector, superboundary_element, offset)

            # Modify boundary_{dim+1} operator
            for coboundary_element, _, offset in self.in_edges(element, keys=True):
                self.add_bound(coboundary_element, new_element, offset)

    def simple_split(self, element: Any, new_element: Any, new_connector: Any, *boundary: tuple[Any, tuple[int, ...]]):
        """Do a simple split of element, creating new_element and connecting both by new_connector.

        This method only makes sense for elements whose dimension > 0. If you want to do a vertex split
        (Nickerson), you should do a cell split of the dual complex with self.dual.simple_split(vertex, ...).

        Args:
            element (Any): Name of the element being split (e.g. a face f).
            new_element (Any): Name of the new element that is created (e.g. a face f_1).
            new_connector (Any): Name of the new connector between element and new_element (e.g. an edge e_1).
            *boundary (tuple[Any, tuple[int, ...]]): The boundary of element that should be transferred to new_element.
                Specified as pairs of (boundary_element, offset) that are currently the boundary of element.
        """
        self.n_split(element, (new_element, new_connector, boundary))

class Lattice(Parameterized, GraphUtils, nx.MultiDiGraph):
    """A graph description of the Bravais lattice from which the crystal complex is built.

    The edges of the graph are lazy-loaded depending on which offsets are present in the `UnitCell` that a `Crystal` is built with.

    Attributes:
        reps (tuple): The number of the lattice points along each direction.
        shear_matrix (np.ndarray): A dim x dim matrix where the i-th row represents the shear of the lattice along the boundary plane
            normal to the CARTESIAN_LABELS[i] direction. Naturally, all ii elements are zero.
    """
    def __init__(self, reps, **attr):
        """Create a new lattice with `reps` number of lattice points along each dimension"""

        name = "_".join(str(r) for r in reps)
        super().__init__(name=name, dim=len(reps), **attr)

        self.reps = np.asarray(sp.sympify(reps))
        self.shear_matrix = np.zeros((self.dim, self.dim), dtype=int)

        mask = ~np.eye(self.dim, dtype=bool)
        for i, label in zip(range(self.dim), CARTESIAN_LABELS):
            try:
                self.shear_matrix[i, mask[i]] = self.graph[f"{label}_shear"]
            except KeyError:
                continue
        
        # Annoying fix for when reps does not have parameters (crystal build will fail because lattice has no nodes)
        if not self.parameters:
            self.finalize_parameters()

    @property
    def parameters(self):
        return set() if self.reps.dtype == int else set.union(*(r.free_symbols for r in self.reps))

    def copy(self):
        return copy.deepcopy(self)

    def _bind_parameters(self, **parameters):
        self.reps = np.array([r.subs(parameters) for r in self.reps])

    def _finalize_parameters(self):
        self.reps = np.asarray(self.reps, dtype=int)

        if any(r <= 0 for r in self.reps):
            raise ValueError(f"Repetitions {self.reps} are not positive integers")
        
        for pos in np.ndindex(*self.reps):
            positions = dict(zip(CARTESIAN_LABELS, pos))
            self.add_node(pos, **positions)

        # Adds the dimensions of this lattice as X, Y, Z integers in the graph dict
        for label, rep in zip(CARTESIAN_LABELS.upper(), self.reps):
            self.graph[label] = rep

        self._loaded_offsets = set()

    @parameters_bound
    def filter_offsets(self, offset):
        self._load_offsets(offset)
        
        return ((u, v) for u, v, o in self.edges(keys=True) if o == offset)

    @parameters_bound
    def _load_offsets(self, offset):
        # Check if offset not already lazy-loaded, otherwise load now
        self.check_len(offset)
        
        if offset not in self._loaded_offsets:
            # I'm not sure if this works
            a, b = np.divmod(np.array(self.nodes) + offset, self.reps)
            a, b = np.divmod(b + a @ self.shear_matrix, self.reps)

            self.add_edges_from(zip(self.nodes, map(tuple, b + a @ self.shear_matrix), repeat(offset)))
            self._loaded_offsets.add(offset)

    @property
    @parameters_bound
    def nodes(self):
        return super().nodes

    @property
    @parameters_bound
    def edges(self):
        return super().edges

class Crystal(GraphUtils, nx.DiGraph):
    """Graph representation of a crystal given by its elements (faces, edges, etc.) and its boundary relations.

    This representation is basically a chain complex.
    """
    def __init__(self, name, dim, **attr):
        super().__init__(name=name, dim=dim, **attr)
