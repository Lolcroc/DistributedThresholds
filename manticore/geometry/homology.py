# Python
from __future__ import annotations
from itertools import compress
from functools import reduce

# Math
import numpy as np

# Manticore
from manticore.geometry import Lattice, UnitCell, CARTESIAN_LABELS
from manticore.geometry.linalg import boundary_maps, fundamental_subspaces, quotient, cartesian_transform

class Logical:
    def __init__(self, elements: set, checks: set, direction: str):
        self.elements = elements  
        self.checks = checks  # The associated logical in the cohomology group
        self.direction = direction  # e.g. "x"
    
    def build(self, lattice: Lattice, node_index):
        labels = CARTESIAN_LABELS[:lattice.dim]
        orthogonal_directions = labels.replace(self.direction, "")

        new_elements, new_checks = set(), set()

        for i, dd in lattice.nodes(data=True):
            if all(dd[direction] == 0 for direction in orthogonal_directions):  # Logical ops are along the line !self.direction == 0
                new_elements.update(node_index[(uc_element, i)] for uc_element in self.elements)

            if dd[self.direction] == 0:  # Checks are in the plane self.direction == 0
                new_checks.update(node_index[(uc_element, i)] for uc_element in self.checks)

        return Logical(new_elements, new_checks, self.direction)

    def __repr__(self):
        return f"{self.__class__.__name__}(direction={self.direction}, elements={self.elements}, checks={self.checks})"

class Homology:
    """Contains only the first homology group H_1 and its associated 'checks' in the corresponding cohomology group.
    
    E.g. for the 3D case, a logical operator in 'x' direction has a check consisting of the plane in 'yz' directions.
    The logical operator [c_1] is in H_1, whilst its check [dual(c_2)] is in dual(H_2).
    """
    def __init__(self, name, data: dict[str, Logical] | UnitCell = None):
        self.name = name
        self.operators = data if isinstance(data, dict) else _compute_homology(data)
    
    def build(self, lattice: Lattice, node_index):
        new_logicals = {}
        for name, logical in self.operators.items():
            new_logicals[name] = logical.build(lattice, node_index)
        
        return Homology(self.name, new_logicals)
        
    def logical_errors(self, errors: set) -> str:
        return "".join(str(len(errors & logical.checks) % 2) for logical in self.operators.values())

    def __len__(self):
        return len(self.operators)

    def __iter__(self):
        return iter(self.operators)

    def __getitem__(self, key):
        return self.operators[key]

    def __repr__(self):
        return f"{self.__class__.__name__}(operators={self.operators})"

    def __add__(self, other):
        operators = {}
        for hom in (self, other):
            for name, logical in hom.operators.items():
                operators[f"{hom.name}_{name}"] = logical

        return Homology(f"{self.name}_{other.name}", operators)

def _compute_homology(unit_cell: UnitCell):
    bounds, elements = boundary_maps(unit_cell)

    spaces = []
    for d in bounds:
        collapsed = reduce(lambda a, b: a ^ b, d.values())
        spaces.append(fundamental_subspaces(collapsed))

    # Primal H1 (primal_ops) and dual H2 (primal_checks) calculations
    hom = quotient(spaces[0]["ker"], spaces[1]["img"])
    cohom = quotient(spaces[1]["coker"], spaces[0]["coimg"])

    hom = cartesian_transform(hom, bounds[0])
    cohom = cohom @ np.linalg.inv(hom.T @ cohom)

    def from_dense(vector):
        return set(compress(elements[1], vector))
    
    logicals = {}

    for label, logical, check in zip(CARTESIAN_LABELS, hom.T, cohom.T):
        name = f"logical_{label}"
        logicals[name] = Logical(from_dense(logical), from_dense(check), label)

    return logicals
