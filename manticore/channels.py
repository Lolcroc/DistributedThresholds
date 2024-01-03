# Python
import copy
from collections import Counter, defaultdict
from abc import ABC, abstractmethod

# Math
import networkx as nx
import numpy as np

# Manticore
from manticore.geometry import UnitCell, Lattice, Homology
from manticore.errors import ErrorModel
from manticore.decoders.uf import UFDecoder
from manticore.utils import Parameterized, parameters_bound

class Channel(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def sample(self, reps, rng):
        pass

class IsotropicChannel(Parameterized, Channel):
    """Contains primal and dual components."""
    def __init__(self, unit_cell: UnitCell, error_model: ErrorModel, lattice: Lattice):
        name = f"{unit_cell.name}_{lattice.name}_{error_model.name}"
        super().__init__(name)

        self.unit_cell = unit_cell
        self.error_model = error_model
        self.lattice = lattice

        self.unit_homology = Homology("primal", self.unit_cell.dual) + Homology("dual", self.unit_cell)

    @parameters_bound
    def sample(self, reps, rng):
        logical_errors = Counter()

        for _ in range(reps):
            errors, erasures = self.error_sampler.sample(rng)

            primal_errors = self.primal_error_locations & errors  # Faces
            dual_errors = self.dual_error_locations & errors  # Edges

            primal_erasures = self.primal_error_locations & erasures  # Faces
            dual_erasures = self.dual_error_locations & erasures  # Edges

            primal_syndromes = self.crystal.dual.boundary(primal_errors)  # Cells
            dual_syndromes = self.crystal.boundary(dual_errors)  # Vertices

            primal_correction = self.primal_decoder.decode(primal_syndromes, primal_erasures)  # Cells and faces
            dual_correction = self.dual_decoder.decode(dual_syndromes, dual_erasures)  # Vertices and edges

            net_errors = (primal_errors ^ primal_correction) | (dual_errors ^ dual_correction)
            net_logicals = self.homology.logical_errors(net_errors)
            logical_errors[net_logicals] += 1

        return logical_errors

    @property
    def parameters(self):
        return self.error_model.parameters | self.lattice.parameters
    
    def copy(self):
        return copy.deepcopy(self)

    def _bind_parameters(self, **parameters):
        self.error_model.bind_parameters(inplace=True, **parameters)  # Deepcopy already does copies of contained objects
        self.lattice.bind_parameters(inplace=True, **parameters)

    def _finalize_parameters(self):
        node_index = defaultdict(lambda: len(node_index))  # Easy method for unique IDs

        self.crystal = self.unit_cell.build(self.lattice, node_index)
        self.error_sampler = self.error_model.build(self.lattice, node_index)
        self.homology = self.unit_homology.build(self.lattice, node_index)

        self.primal_decoder = UFDecoder(self.crystal)  # Faces
        self.dual_decoder = UFDecoder(self.crystal.dual)  # Edges

        self.primal_error_locations = set(self.crystal.dual.filter_nodes(dim=1))  # Faces
        self.dual_error_locations = set(self.crystal.filter_nodes(dim=1))  # Edges

    def __repr__(self):
        return f"{self.__class__.__name__}(unit_cell={self.unit_cell}, error_model={self.error_model}, lattice={self.lattice})"

class BoundaryChannel(IsotropicChannel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_boundary"

        self.unit_primal_homology = Homology("primal", self.unit_cell.dual)  # Faces
        self.unit_dual_homology = Homology("dual", self.unit_cell)  # Edges
    
    def _finalize_parameters(self):
        node_index = defaultdict(lambda: len(node_index))  # Easy method for unique IDs

        crystal = self.unit_cell.build(self.lattice, node_index)

        # Introduce boundaries
        rough_boundary, smooth_boundary = set(), set()

        for name in self.unit_primal_homology:
            op = self.unit_primal_homology[name]
            if op.direction == "y":
                for elem in op.checks:
                    rough_boundary.update(crystal.filter_nodes(uc=elem, y=0))

        for name in self.unit_dual_homology:
            op = self.unit_dual_homology[name]
            if op.direction == "x":
                for elem in op.checks:
                    smooth_boundary.update(crystal.filter_nodes(uc=elem, x=0))

        print(self.unit_primal_homology)
        print(self.unit_homology)
        print("")

        removed_faces = _glue(crystal, rough_boundary)
        removed_edges = _glue(crystal.dual, smooth_boundary)
        removed_elements = removed_faces | removed_edges

        self.crystal = crystal
        self.error_sampler = self.error_model.build(self.lattice, node_index)

        # Remove any error sets that had overlap with removed elements
        for error in self.error_sampler:
            remove_loc_indices = []
            for i, (error_loc, erasure_loc) in enumerate(zip(error.error_locations, error.erasure_locations)):
                if any(e in removed_elements for e in error_loc) or any(e in removed_elements for e in erasure_loc):
                    remove_loc_indices.append(i)

            error.error_locations = np.delete(error.error_locations, remove_loc_indices, axis=0)
            error.erasure_locations = np.delete(error.erasure_locations, remove_loc_indices, axis=0)

        self.homology = self.unit_homology.build(self.lattice, node_index)  # This works regardless of removed elements

        self.primal_decoder = UFDecoder(self.crystal)  # Faces
        self.dual_decoder = UFDecoder(self.crystal.dual)  # Edges

        self.primal_error_locations = set(self.crystal.dual.filter_nodes(dim=1))  # Faces
        self.dual_error_locations = set(self.crystal.filter_nodes(dim=1))  # Edges

def _glue(crystal, logical):
    logical = logical & set(crystal.nodes) # Dirty fix if overlap with previous glues

    edges = set.union(*(set(crystal[n]) for n in logical))
    verts = list(set.union(*(set(crystal[n]) for n in edges)))

    to_remove = logical | edges
    crystal.remove_nodes_from(to_remove)

    base = verts[0]
    crystal.nodes[base]["boundary"] = True
    for node in verts[1:]:
        nx.contracted_nodes(crystal, base, node, self_loops=False, copy=False)

    return to_remove

if __name__ == "__main__":
    from manticore.geometry import Lattice
    from manticore.geometry.library import CubicCell
    from manticore.errors.library import Phenomenological

    uc = CubicCell()
    # raise ValueError
    em = Phenomenological(uc, 0.1)
    lat = Lattice((3,3,3))

    channel = BoundaryChannel(uc, em, lat)
    channel.finalize_parameters()