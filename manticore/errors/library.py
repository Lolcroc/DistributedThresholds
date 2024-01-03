# Python
from __future__ import annotations
from typing import Any
from collections import defaultdict
from itertools import chain

# Math
import numpy as np

# Manticore
from manticore.geometry import UnitCell
from manticore.geometry.library import CubicCell, SixRingCubicCell, MaxSplitCubicCell, ThreeDiamondCell, TwoThreeDiamondCell, DoubleEdgeBellCubicCell, DoubleEdgeGHZCubicCell
from manticore.errors import ErrorKernel, ErrorModel
from manticore.simulator import build_kernel, get_frames, get_single_frame
from manticore.simulator.library import MonolithicCircuit, DistributedCircuit, FourGHZFusion, ThreeGHZFusion, DistributedCircuitErasure
from manticore.simulator.frame_errors import Depolarize, FrameError, DiagonalizeGHZ, ClassicalErase

from manticore.simulator.base import Frame
from manticore.simulator.circuit import QuantumOutRegister, OutClbit, OutQubit, Erasurebit
from manticore.simulator.linalg import bin_matrix, unbin_matrix
from qiskit.circuit import QuantumCircuit, QuantumRegister, Clbit, Qubit

import sympy as sp

def group_valencies(unit_cell, name=None):
    valencies = defaultdict(list)

    for edge in unit_cell.filter_nodes(dim=1):
        valency = len(unit_cell.in_edges(edge))
        valencies[valency].append([edge])   # Locations is a list-of-lists
    
    return valencies

def error_locations(unit_cell):
    locations = list([edge] for edge in unit_cell.filter_nodes(dim=1))  # Edges
    if unit_cell.dim > 2:
        locations.extend([edge] for edge in unit_cell.dual.filter_nodes(dim=1))  # Faces
    return locations

class PhenomKernel(ErrorKernel):
    """Standard phenomenological noise on corresponding locations."""
    def __init__(self, probability, locations: list[list[Any]] | UnitCell, name=None):
        if isinstance(locations, UnitCell):
            locations = error_locations(locations)
        
        super().__init__([1-probability, probability], locations, name=name)

class ErasureKernel(ErrorKernel):
    """Standard erasure kernel on corresponding locations.

    Erasure is always accompanied by uniform random noise (such as to create a mixed state).
    There are two ways to apply such a map:

        1. Erasure on a classical bit. The probabilities are [1-p, 0, p/2, p/2]
        2. Erasure on a qubit. The probabilities are [1-p, 0, 0, 0, p/4, p/4, p/4, p/4]

    The second case represents a type of 'correlated' erasure on two different edges in the
    primal and dual syndrome graphs. This type of kernel makes no assumptions about the nature
    of the bit (like a phenomenological error model), hence only the first type is applied.
    """

    def __init__(self, probability, locations: list[list[Any]] | UnitCell, name=None):
        if isinstance(locations, UnitCell):
            locations = error_locations(locations)
        
        super().__init__(
            [1-probability, 0, probability/2, probability/2], 
            error_locations=locations, 
            erasure_locations=locations, 
            name=name
        )

class PhenomenologicalErasureKernel(ErrorKernel):
    """A combination of both phenomenological and erasure noise.
    
    We can use the above models and combine them. Working the probabilities out on a piece of paper
    gives [(1-p)*(1-p_e), p(1-p_e), p_e/2, p_e/2], where p represents phenomenological error rate
    and p_e represents erasure error rate.
    """

    def __init__(self, phenom_probability, erasure_probability, locations: list[list[Any]] | UnitCell, name=None):
        if isinstance(locations, UnitCell):
            locations = error_locations(locations)

        super().__init__(
            [(1-phenom_probability)*(1-erasure_probability), phenom_probability*(1-erasure_probability), erasure_probability/2, erasure_probability/2],
            error_locations=locations, 
            erasure_locations=locations, 
            name=name
        )

class Phenomenological(ErrorModel):
    def __init__(self, unit_cell: UnitCell, probability):
        super().__init__(f"{unit_cell.name}_phenom_{probability}", errors=[PhenomKernel(probability, unit_cell)])

class Erasure(ErrorModel):
    def __init__(self, unit_cell: UnitCell, probability):
        super().__init__(f"{unit_cell.name}_erasure_{probability}", errors=[ErasureKernel(probability, unit_cell)])

class PhenomenologicalErasure(ErrorModel):
    def __init__(self, unit_cell: UnitCell, phenom_probability, erasure_probability):
        error_kernel = PhenomenologicalErasureKernel(phenom_probability, erasure_probability, unit_cell)
        super().__init__(f"{unit_cell.name}_phenom_{phenom_probability}_erasure_{erasure_probability}", errors=[error_kernel])

class WeightedPhenomenological(ErrorModel):
    def __init__(self, unit_cell: UnitCell, probability):
        error_kernels = []

        for valency, locations in group_valencies(unit_cell).items():
            error_kernels.append(PhenomKernel(valency * probability, locations, name=f"{valency}"))

        if unit_cell.dim > 2:
            for valency, locations in group_valencies(unit_cell.dual).items():
                error_kernels.append(PhenomKernel(valency * probability, locations, name=f"dual_{valency}"))

        super().__init__(f"{unit_cell.name}_weighted_phenom_{probability}", errors=error_kernels)

class WeightedErasure(ErrorModel):
    def __init__(self, unit_cell: UnitCell, probability):
        error_kernels = []

        for valency, locations in group_valencies(unit_cell).items():
            error_kernels.append(ErasureKernel(valency * probability, locations, name=f"{valency}"))

        if unit_cell.dim > 2:
            for valency, locations in group_valencies(unit_cell.dual).items():
                error_kernels.append(ErasureKernel(valency * probability, locations, name=f"dual_{valency}"))

        super().__init__(f"{unit_cell.name}_weighted_erasure_{probability}", errors=error_kernels)

class WeightedPhenomenologicalErasure(ErrorModel):
    def __init__(self, unit_cell: UnitCell, phenom_probability, erasure_probability):
        error_kernels = []

        for valency, locations in group_valencies(unit_cell).items():
            error_kernel = PhenomenologicalErasureKernel(valency * phenom_probability, valency * erasure_probability, locations, name=f"{valency}")
            error_kernels.append(error_kernel)

        if unit_cell.dim > 2:
            for valency, locations in group_valencies(unit_cell.dual).items():
                error_kernel = PhenomenologicalErasureKernel(valency * phenom_probability, valency * erasure_probability, locations, name=f"dual_{valency}")
                error_kernels.append(error_kernel)
        
        super().__init__(f"{unit_cell.name}_weighted_phenom_{phenom_probability}_erasure_{erasure_probability}", errors=error_kernels)

class Monolithic(ErrorModel):
    def __init__(self, unit_cell: UnitCell, probability, color_ordering=None, condition=None):
        error_kernels = []

        circuit = MonolithicCircuit(unit_cell, probability, color_ordering)

        for frame in get_frames(circuit):
            error_kernel = build_kernel(frame, condition)
            error_kernels.append(error_kernel)

        # Because initialization (Z) errors commute through CZ, the SPAM error model is twice a phenomenological
        # model with the same parameter p. Then the probability of error is 2*p*(1 - p).
        spam_kernel = PhenomKernel(2*probability*(1-probability), unit_cell, name=f"spam_{probability}")
        error_kernels.append(spam_kernel)

        super().__init__(f"{unit_cell.name}_monolithic_{probability}", errors=error_kernels)

class SixRing(ErrorModel):
    def __init__(self, p, p_i, condition=None):
        error_kernels = []

        unit_cell = SixRingCubicCell()
        circuit = DistributedCircuit(unit_cell, p, {2: DiagonalizeGHZ(2, p_i)})

        for frame in get_frames(circuit):
            error_kernel = build_kernel(frame, condition)
            error_kernels.append(error_kernel)

        super().__init__(f"cubic_six_ring_{p}_{p_i}", errors=error_kernels)

class CubicGHZFusion(ErrorModel):
    def __init__(self, p, p_i, condition=None):
        error_kernels = []

        ghz_circuit = FourGHZFusion(p, p_i)
        ghz_frame = get_single_frame(ghz_circuit).error  # We do not 'twirle'

        unit_cell = MaxSplitCubicCell()
        circuit = DistributedCircuit(unit_cell, p, {4: ghz_frame})

        for frame in get_frames(circuit):
            error_kernel = build_kernel(frame, condition)
            error_kernels.append(error_kernel)

        super().__init__(f"cubic_ghz_fusion_{p}_{p_i}", errors=error_kernels)

class ThreeDiamondFusion(ErrorModel):
    def __init__(self, p, p_i, condition=None):
        error_kernels = []

        ghz_circuit = ThreeGHZFusion(p, p_i)
        ghz_frame = get_single_frame(ghz_circuit).error  # We do not 'twirle'

        unit_cell = ThreeDiamondCell()
        circuit = DistributedCircuit(unit_cell, p, {3: ghz_frame})
        
        for frame in get_frames(circuit):
            error_kernel = build_kernel(frame, condition)
            error_kernels.append(error_kernel)

        super().__init__(f"three_diamond_fusion_{p}_{p_i}", errors=error_kernels)

class TwoThreeDiamondFusion(ErrorModel):
    def __init__(self, p, p_i, condition=None):
        error_kernels = []

        ghz_circuit = ThreeGHZFusion(p, p_i)
        ghz_frame = get_single_frame(ghz_circuit).error  # We do not 'twirle'

        unit_cell = TwoThreeDiamondCell()
        circuit = DistributedCircuit(unit_cell, p, {3: ghz_frame, 2: DiagonalizeGHZ(2, p_i)})

        for frame in get_frames(circuit):
            error_kernel = build_kernel(frame, condition)
            error_kernels.append(error_kernel)

        super().__init__(f"23_diamond_fusion_{p}_{p_i}", errors=error_kernels)

class DoubleEdgeBell(ErrorModel):
    def __init__(self, p, p_i, condition=None):
        error_kernels = []

        unit_cell = DoubleEdgeBellCubicCell()
        circuit = DistributedCircuit(unit_cell, p, {2: DiagonalizeGHZ(2, p_i)})

        for frame in get_frames(circuit):
            error_kernel = build_kernel(frame, condition)
            error_kernels.append(error_kernel)

        super().__init__(f"double_edge_bell_{p}_{p_i}", errors=error_kernels)

class DoubleEdgeGHZFusion(ErrorModel):
    def __init__(self, p, p_i, condition=None):
        error_kernels = []

        ghz_circuit = FourGHZFusion(p, p_i)
        ghz_frame = get_single_frame(ghz_circuit).error  # We do not 'twirle'
        # ghz_frame.label = "GHZ_frame"

        unit_cell = DoubleEdgeGHZCubicCell()
        circuit = DistributedCircuit(unit_cell, p, {4: ghz_frame})

        for frame in get_frames(circuit):
            # if frame.error.label == "GHZ_frame":
            #     print(frame.error.name, frame.error.label)
            #     print(frame.probabilities)
            #     print(frame.indices)
            #     print([(getattr(q, "xlabel", str(q)), getattr(q, "zlabel", str(q))) for q in frame.qubits])
            #     print([getattr(q, "label", str(q)) for q in frame.clbits])
            error_kernel = build_kernel(frame, condition)
            error_kernels.append(error_kernel)

        super().__init__(f"double_edge_ghz_{p}_{p_i}", errors=error_kernels)

class SixRingErasure(ErrorModel):
    def __init__(self, p, p_i, condition=None):
        error_kernels = []

        unit_cell = SixRingCubicCell()

        q0 = Qubit()
        q1 = OutQubit()
        q2 = Qubit()
        q3 = OutQubit()

        ca = OutClbit()
        cb = Clbit()

        single_select = QuantumCircuit([q0, q1, q2, q3, ca, cb])
        single_select.create_bell(q0, q2, p_i)
        single_select.create_bell(q1, q3, p_i)

        single_select.cx(q0, q1)
        single_select.cx(q2, q3)

        single_select.depolarize(p, [q0, q1])
        single_select.depolarize(p, [q2, q3])

        single_select.measure(q0, ca)
        single_select.measure(q2, cb)

        single_select.measure_flip(p, ca)
        single_select.measure_flip(p, cb)

        single_select.clcx(cb, ca)

        frame = get_single_frame(single_select)
        new_frame = frame.condition_clbit(ca, 0)

        circuit = DistributedCircuit(unit_cell, p, {2: new_frame.error})

        for frame in get_frames(circuit):
            prob_norm = sp.simplify(sum(frame.probabilities))
            if prob_norm != 1:
                print("Applying erasure")
                parent_bit = frame.data[0]
                erasure_bit = Erasurebit(source=parent_bit, label=parent_bit.label)
                erasure_frame = Frame(ClassicalErase(), [], [erasure_bit, parent_bit]) * (1-prob_norm)
                frame = frame + erasure_frame
            error_kernel = build_kernel(frame, condition)
            error_kernels.append(error_kernel)

        super().__init__(f"cubic_six_ring_single_{p}_{p_i}", errors=error_kernels)

class SixRingErasureDouble(ErrorModel):
    def __init__(self, p, p_i, condition=None):
        error_kernels = []

        unit_cell = SixRingCubicCell()

        q0 = Qubit()
        q1 = Qubit()
        q2 = OutQubit()
        q3 = Qubit()
        q4 = Qubit()
        q5 = OutQubit()

        ca = OutClbit()
        cb = Clbit()
        cc = Clbit()
        cd = Clbit()

        double_select = QuantumCircuit([q0, q1, q2, q3, q4, q5, ca, cb, cc, cd])
        double_select.create_bell(q0, q3, p_i)
        double_select.create_bell(q1, q4, p_i)
        double_select.create_bell(q2, q5, p_i)

        double_select.cx(q1, q2)
        double_select.cx(q4, q5)

        double_select.depolarize(p, [q1, q2])
        double_select.depolarize(p, [q4, q5])

        double_select.cz(q0, q1)
        double_select.cz(q3, q4)

        double_select.depolarize(p, [q0, q1])
        double_select.depolarize(p, [q3, q4])

        double_select.measure(q0, ca)
        double_select.measure(q1, cb)
        double_select.measure(q3, cc)
        double_select.measure(q4, cd)

        double_select.measure_flip(p, ca)
        double_select.measure_flip(p, cb)
        double_select.measure_flip(p, cc)
        double_select.measure_flip(p, cd)

        double_select.clcx(cb, ca)
        double_select.clcx(cc, ca)
        double_select.clcx(cd, ca)

        frame = get_single_frame(double_select)
        new_frame = frame.condition_clbit(ca, 0)

        circuit = DistributedCircuit(unit_cell, p, {2: new_frame.error})

        for frame in get_frames(circuit):
            prob_norm = sp.simplify(sum(frame.probabilities))
            if prob_norm != 1:
                print("Applying erasure")
                parent_bit = frame.data[0]
                erasure_bit = Erasurebit(source=parent_bit, label=parent_bit.label)
                erasure_frame = Frame(ClassicalErase(), [], [erasure_bit, parent_bit]) * (1-prob_norm)
                frame = frame + erasure_frame
            error_kernel = build_kernel(frame, condition)
            error_kernels.append(error_kernel)

        super().__init__(f"cubic_six_ring_double_{p}_{p_i}", errors=error_kernels)

class SixRingErasureNew(ErrorModel):
    def __init__(self, p, p_i, condition=None):
        error_kernels = []

        unit_cell = SixRingCubicCell()
        circuit = DistributedCircuitErasure(unit_cell, p, {2: DiagonalizeGHZ(2, p_i)})

        for frame in get_frames(circuit):
            error_kernel = build_kernel(frame, condition)
            error_kernels.append(error_kernel)

        super().__init__(f"cubic_six_ring_single_{p}_{p_i}", errors=error_kernels)

class SurfaceCodeMonolithic(ErrorModel):
    def __init__(self, p):

        # Z (plaq) stab, going anticlockwise
        z0 = OutQubit(xlabel="f_y:0,0,0", zlabel="e_x:0,0,0")
        z1 = OutQubit(xlabel="f_x:0,0,0", zlabel="e_y:0,0,0")
        z2 = OutQubit(xlabel="f_y:0,1,0", zlabel="e_x:0,1,0")
        z3 = OutQubit(xlabel="f_x:1,0,0", zlabel="e_y:1,0,0")
        za = Qubit()
        zm = OutClbit(label="f_z:0,0,0")

        # X (star) stab, going anticlockwise
        x0 = OutQubit(xlabel="f_y:-1,0,-1", zlabel="e_x:-1,0,0")
        x1 = OutQubit(xlabel="f_x:0,-1,-1", zlabel="e_y:0,-1,0")
        x2 = OutQubit(xlabel="f_y:0,0,-1", zlabel="e_x:0,0,0")
        x3 = OutQubit(xlabel="f_x:0,0,-1", zlabel="e_y:0,0,0")
        xa = Qubit()
        xm = OutClbit(label="e_z:0,0,-1")

        z_circuit = QuantumCircuit([z0, z1, z2, z3, za, zm])
        x_circuit = QuantumCircuit([x0, x1, x2, x3, xa, xm])

        # Init error
        z_circuit.phase_flip(p, za)
        x_circuit.phase_flip(p, xa)

        # Zigzag CZ: z0 - z2 - z1 - z3
        for qubit in (z0, z2, z1, z3):
            z_circuit.cz(za, qubit)
            z_circuit.depolarize(p, [za, qubit])

        # Zigzag CX: x0 - x2 - x1 - x3
        for qubit in (x0, x2, x1, x3):
            x_circuit.cx(xa, qubit)
            x_circuit.depolarize(p, [xa, qubit])

        # Measure
        z_circuit.h(za)
        z_circuit.measure(za, zm)
        z_circuit.measure_flip(p, zm)
        x_circuit.h(xa)
        x_circuit.measure(xa, xm)
        x_circuit.measure_flip(p, xm)

        z_kernel = build_kernel(get_single_frame(z_circuit))
        x_kernel = build_kernel(get_single_frame(x_circuit))

        super().__init__(f"cbcq_monolithic_{p}", errors=[x_kernel, z_kernel])

class SurfaceCodeGHZ(ErrorModel):
    def __init__(self, p, p_i):

        # Z (plaq) stab, going anticlockwise
        z0 = OutQubit(xlabel="f_y:0,0,0", zlabel="e_x:0,0,0")
        z1 = OutQubit(xlabel="f_x:0,0,0", zlabel="e_y:0,0,0")
        z2 = OutQubit(xlabel="f_y:0,1,0", zlabel="e_x:0,1,0")
        z3 = OutQubit(xlabel="f_x:1,0,0", zlabel="e_y:1,0,0")
        z_data = (z0, z1, z2, z3)
        z_anc = za0, za1, za2, za3 = tuple(Qubit() for _ in range(4))
        zm_anc = zm0, zm1, zm2, zm3 = tuple(Clbit() for _ in range(4))
        zm = OutClbit(label="f_z:0,0,0")

        # X (star) stab, going anticlockwise
        x0 = OutQubit(xlabel="f_y:-1,0,-1", zlabel="e_x:-1,0,0")
        x1 = OutQubit(xlabel="f_x:0,-1,-1", zlabel="e_y:0,-1,0")
        x2 = OutQubit(xlabel="f_y:0,0,-1", zlabel="e_x:0,0,0")
        x3 = OutQubit(xlabel="f_x:0,0,-1", zlabel="e_y:0,0,0")
        x_data = (x0, x1, x2, x3)
        x_anc = xa0, xa1, xa2, xa3 = tuple(Qubit() for _ in range(4))
        xm_anc = xm0, xm1, xm2, xm3 = tuple(Clbit() for _ in range(4))
        xm = OutClbit(label="e_z:0,0,-1")

        fusion_circuit = FourGHZFusion(p, p_i)
        frame = get_single_frame(fusion_circuit)
        ghz_error = frame.error
        
        z_circuit = QuantumCircuit([z0, z1, z2, z3, *z_anc, *zm_anc, zm])
        x_circuit = QuantumCircuit([x0, x1, x2, x3, *x_anc, *xm_anc, xm])

        # GHZ error
        z_circuit.append(ghz_error.copy(), qargs=z_anc)
        x_circuit.append(ghz_error.copy(), qargs=x_anc)

        # Z circuit indirect measurement
        for data, anc, manc in zip(z_data, z_anc, zm_anc):
            z_circuit.cz(anc, data)
            z_circuit.depolarize(p, [anc, data])

            z_circuit.h(anc)
            z_circuit.measure(anc, manc)
            z_circuit.measure_flip(p, manc)
            z_circuit.clcx(manc, zm)

        # X circuit indirect measurement
        for data, anc, manc in zip(x_data, x_anc, xm_anc):
            x_circuit.cx(anc, data)
            x_circuit.depolarize(p, [anc, data])

            x_circuit.h(anc)
            x_circuit.measure(anc, manc)
            x_circuit.measure_flip(p, manc)
            x_circuit.clcx(manc, xm)

        z_frame = get_single_frame(z_circuit)
        x_frame = get_single_frame(x_circuit)

        # Nickerson subrounds - commute the superop through the projector by its commutation relation
        z_preframe = z_frame.copy()
        x_preframe = x_frame.copy()

        # Add the sum of all x bits to the measurement bit
        inds = bin_matrix(z_frame.indices, z_frame.num_bits, galois=True)
        inds[:, 0] += np.sum(inds[:, z_frame.num_clbits:(z_frame.num_clbits + z_frame.num_qubits)], axis=1)
        z_preframe.indices = unbin_matrix(inds)
        # Moves the X errors one layer back
        z_preframe.qubits = [OutQubit(xlabel=qubit.xlabel[:-1] + "-1", zlabel=qubit.zlabel) for qubit in z_preframe.qubits]

        # Add the sum of all z bits to the measurement bit
        inds = bin_matrix(x_frame.indices, x_frame.num_bits, galois=True)
        inds[:, 0] += np.sum(inds[:, (x_frame.num_clbits + x_frame.num_qubits):], axis=1)
        x_preframe.indices = unbin_matrix(inds)
        # Moves the Z errors one layer back
        x_preframe.qubits = [OutQubit(xlabel=qubit.xlabel, zlabel=qubit.zlabel[:-1] + "-1") for qubit in x_preframe.qubits]
        
        # Checkerboard condition
        z_kernel = build_kernel(z_frame, condition="(x + y) % 2 == 0")
        z_prekernel = build_kernel(z_preframe, condition="(x + y) % 2 == 1")

        x_kernel = build_kernel(x_frame, condition="(x + y) % 2 == 0")
        x_prekernel = build_kernel(x_preframe, condition="(x + y) % 2 == 1")

        super().__init__(f"cbcq_ghz_{p}_{p_i}", errors=[x_kernel, z_kernel, x_prekernel, z_prekernel])

if __name__ == "__main__":
    import sympy as sp

    p, p_i = sp.symbols(("p", "p_i"))

    # error_model = DoubleEdgeGHZFusion(p, p_i)

    # error_model = SurfaceCodeMonolithic(p)
    # print(error_model)

    error_model = SurfaceCodeGHZ(p, p_i)
    print(error_model)

    # for error in error_model:
    #     print(error.name)
    #     print(error.probabilities)
    #     print(error.error_locations)

    # error_model.to_json("H:/onzin.json")
    # print(error_model)