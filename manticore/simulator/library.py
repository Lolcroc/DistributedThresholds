# Python
from collections import defaultdict
from itertools import combinations

# Math
import numpy as np
import sympy as sp
import networkx as nx

# Qiskit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, Qubit, Clbit

# Manticore
from manticore.simulator.base import Frame
from manticore.simulator.frame_errors import FrameError, QuantumErase, ClassicalErase
from manticore.simulator.circuit import InQubit, OutQubit, IOQubit, InClbit, OutClbit, IOClbit, LabelledClbit, Erasurebit
from manticore.simulator.circuit import QuantumInRegister, QuantumOutRegister, QuantumIORegister, ClassicalInRegister, ClassicalOutRegister, ClassicalIORegister

class CustomGates:
    """Various custom gates

    U_A maps 0 -> -i, 1 -> +i = first h then sdg
    U_B maps 0 -> +i, 1 -> -i = first h then s

    TODO Are these above forms correct?
    """
    ua = QuantumCircuit(1, name='ua')
    ua.h(0)
    ua.sdg(0)
    ua = ua.to_gate()

    ub = QuantumCircuit(1, name='ub')
    ub.h(0)
    ub.s(0)
    ub = ub.to_gate()

class DiagonalBell(QuantumCircuit):

    def __init__(self, p_i):
        qreg = QuantumOutRegister(2, "q")
        super().__init__(qreg)

        self.create_bell(*qreg, p_i)

class ThreeGHZFusion(QuantumCircuit):
    def __init__(self, p, p_i):
        qa, qb0, qc = (OutQubit() for _ in range(3))
        qb1 = Qubit()
        mbit = Clbit()

        super().__init__([qa, qb0, qb1, qc, mbit])

        self.create_bell(qa, qb0, p_i)
        self.create_bell(qc, qb1, p_i)

        self.cx(qb0, qb1)
        self.depolarize(p, [qb0, qb1])
        self.measure(qb1, mbit)
        self.measure_flip(p, mbit)

        self.x(qc).c_if(mbit, 1)

class FourGHZFusion(QuantumCircuit):
    def __init__(self, p, p_i):
        qa, qb, qc, qd = (OutQubit() for _ in range(4))
        qb1, qc1 = Qubit(), Qubit()
        cb1, cc1 = Clbit(), Clbit()

        super().__init__([qa, qb, qc, qd, qb1, qc1, cb1, cc1])

        self.create_bell(qa, qb1, p_i)
        self.create_bell(qb, qc, p_i)
        self.create_bell(qc1, qd, p_i)

        self.cx(qb, qb1)
        self.cx(qc, qc1)
        self.depolarize(p, [qb, qb1], [qc, qc1])

        self.measure(qb1, cb1)
        self.measure(qc1, cc1)
        self.measure_flip(p, cb1)
        self.measure_flip(p, cc1)

        self.x(qa).c_if(cb1, 1)
        self.x(qd).c_if(cc1, 1)

class BBPSSW(QuantumCircuit):
    def __init__(self, p=None):
        pass

class DEJMPSCircuit(QuantumCircuit):
    """Standard DEJMPS circuit

    Note: the coefficients of bell-diagonal terms has some ideal ordering that influence the 'best' local rotations of Alice and Bob. So the ideal circuit depends on the input state.
    This is not explained in DEJMPS paper but in Appendix B1 of "Optimizing practical entanglement distillation" of Rozpędek et al.
    """
    def __init__(self, p=None):
        qreg_io = QuantumIORegister(2, "qio")
        qreg_anc = AncillaRegister(2, "anc")
        creg_out = ClassicalOutRegister(1, "ca")
        creg = ClassicalRegister(1, "cb")
        super().__init__(qreg_io, qreg_anc, creg_out, creg)

        self.ua(0)
        self.ua(2)
        self.ub(1)
        self.ub(3)
        self.depolarize(p, 0, 1, 2, 3)

        self.cx(0, 2)
        self.cx(1, 3)
        self.depolarize(p, [0, 2], [1, 3])
        
        self.measure((2, 3), (0, 1))
        self.measure_flip(p, 0, 1)

        self.clcx(1, 0)
    
    def ua(self, q):
        self.append(CustomGates.ua, [q])

    def ub(self, q):
        self.append(CustomGates.ub, [q])

class Modicum(QuantumCircuit):

    def __init__(self, p=None):

        a0, b0, c0, d0 = qreg_in = QuantumInRegister(4, "qbot")
        a1, b1, c1, d1 = qreg_io = QuantumIORegister(4, "qtop")
        creg_a = ClassicalOutRegister(1, "cr_a")
        ma = creg_a[0]
        mb, mc, md = creg_bcd = ClassicalRegister(3, "cr_bcd")

        super().__init__(qreg_in, qreg_io, creg_a, creg_bcd)

        # # AB and CD Bell pairs
        self.create_bell(a1, b1)
        self.create_bell(c1, d1)

        # # AC something?
        self.create_bell(a0, c0)
        self.cx(a1, a0)
        self.cx(c1, c0)
        self.depolarize(p, [a0, a1], [c0, c1])
        self.measure(a0, ma)
        self.measure(c0, mc)
        self.measure_flip(p, ma, mc)

        # BD distillation
        self.create_bell(b0, d0)
        self.cz(b0, b1)
        self.cz(d0, d1)
        self.depolarize(p, [b0, b1], [d0, d1])
        self.h(b0)
        self.h(d0)
        self.measure(b0, mb)
        self.measure(d0, md)
        self.measure_flip(p, mb, md)

        # Measurement parities and X correction
        self.clcx(mc, ma) # AC parity
        self.x(c1).c_if(ma, 1)
        self.x(d1).c_if(ma, 1)

        self.clcx(md, mb) # BD parity
        self.clcx(mb, ma) # ABCD parity

class CZMeasure(QuantumCircuit):

    def __init__(self, size):
        qreg_top = QuantumInRegister(size, "qtop")
        qreg_bot = QuantumRegister(size, "qbot")
        creg_top = ClassicalRegister(size, "ctop")
        creg_bot = ClassicalOutRegister(size, "cbot")
        top_cout = ClassicalOutRegister(1, "cout")
        super().__init__(qreg_top, qreg_bot, creg_top, creg_bot, top_cout)

        self.cz(qreg_top, qreg_bot)

        self.h(self.qubits)
        self.measure(qreg_top, creg_top)
        self.measure(qreg_bot, creg_bot)

        for cbit in creg_top:
            self.clcx(cbit, top_cout)

class MonolithicCircuit(QuantumCircuit):
    """Does not include SPAM errors"""
    def __init__(self, unit_cell, probability, color_ordering: list[int] = None):
        unit_cell.verify_coloring()

        if color_ordering is None:
            max_color = max(c for _, _, c in unit_cell.edges(data="color", default=0))
            color_ordering = {i: i for i in range(max_color + 1)}
        else:
            color_ordering = {c: i for i, c in enumerate(color_ordering)}
        
        # Initialize CZ graph
        G = nx.Graph()
        zero_offset = unit_cell.dim * (0,)
        colors = nx.get_edge_attributes(unit_cell, "color")

        edge_locations = set()
        
        # Loop over inner edges (CZ gates with errors)
        for arc in unit_cell.filter_edges_from(dim=2, keys=True):
            face, edge, offset = arc
            edge_loc = (edge, offset)

            edge_locations.add(edge_loc)
            G.add_edge((face, zero_offset), edge_loc, inner=True, color=colors[arc])

        # Loop over outer edges (CZ gates without errors)
        for edge, offset in edge_locations:
            for _, face, offset2 in unit_cell.dual.edges(edge, keys=True):
                total_offset = tuple(i - j for i, j in zip(offset, offset2))
                new_arc = ((edge, offset), (face, total_offset))

                if not G.has_edge(*new_arc):
                    arc = (face, edge, offset2)
                    G.add_edge(*new_arc, inner=False, color=colors[arc])

        # Create qubits and clbitss
        for node in G.nodes:
            label = f"{node[0]}:{','.join(str(o) for o in node[1])}"
            G.nodes[node]["qubit"] = InQubit(xlabel=label + "_x", zlabel=label + "_z")
            G.nodes[node]["clbit"] = OutClbit(label=label)

        qubits = nx.get_node_attributes(G, "qubit")
        clbits = nx.get_node_attributes(G, "clbit")
        colors = nx.get_edge_attributes(G, "color")
        inners = nx.get_edge_attributes(G, "inner")

        # Create the monolithic circuit
        qreg = QuantumRegister(name="q", bits=[qubits[node] for node in sorted(qubits)])
        creg = ClassicalRegister(name="c", bits=[clbits[node] for node in sorted(clbits)])
        super().__init__(qreg, creg)

        # Add CZ gates and depolarizing noise for 'inner' gates
        for edge in sorted(G.edges, key=lambda arc: color_ordering[colors[arc]]):
            qubit_pair = list(map(qubits.get, edge))
            self.cz(*qubit_pair)

            if inners[edge]:
                self.depolarize(probability, qubit_pair)

        # Add X-basis measurements for all qubits
        for node in G.nodes:
            self.h(qubits[node])
            self.measure(qubits[node], clbits[node])

class DistributedCircuitErasure(QuantumCircuit):
    """Includes SPAM errors"""
    def __init__(self, unit_cell, probability, ghz_errors: dict[int, FrameError] = None):
        if ghz_errors is None:
            ghz_errors = {}
        # unit_cell.verify_coloring()  # This doesn't work for split cells. We have to rely on the caller

        # Initialize CZ graph
        G = nx.Graph()
        zero_offset = unit_cell.dim * (0,)
        colors = defaultdict(lambda: -1, nx.get_edge_attributes(unit_cell, "color"))
        
        # Loop over inner edges (CZ gates with errors)
        edge_locations = set()
        for arc in unit_cell.filter_edges_from(dim=2, keys=True):
            face, edge, offset = arc
            edge_loc = (edge, offset)

            edge_locations.add(edge_loc)
            G.add_edge((face, zero_offset), edge_loc, inner=True, color=colors[arc])

        # Loop over outer edges (CZ gates without errors)
        for edge, offset in edge_locations:
            for _, face, offset2 in unit_cell.dual.edges(edge, keys=True):
                total_offset = tuple(i - j for i, j in zip(offset, offset2))
                new_arc = ((edge, offset), (face, total_offset))

                if not G.has_edge(*new_arc):
                    arc = (face, edge, offset2)
                    G.add_edge(*new_arc, inner=False, color=colors[arc])

        # Extract locations where to prepare GHZ states
        ghz_locations = set()

        # Messy. Sorry
        edges_faces = set(list(unit_cell.dual.filter_nodes(dim=1)) + list(unit_cell.filter_nodes(dim=1)))
        split_sources = nx.get_node_attributes(unit_cell, "split_source")
        split_connectors = nx.get_node_attributes(unit_cell, "split_connector")
        split_sources = {k: v for k, v in split_sources.items() if k in edges_faces and v in edges_faces}
        split_connectors = {k: v for k, v in split_connectors.items() if k in edges_faces and v in edges_faces}

        for node in list(G.nodes):
            element, offset = node
            connected_split = split_connectors.get(element, None)

            if connected_split:
                G.remove_node(node)  # Removes the bivalent edge (see Thesis for details)
                ghz_locations.add((connected_split, offset))
        
        # IMPORTANT set is not order-safe. Frames may be applied in a different order unless we
        # sort while looping over later on
        # Entangled partners always have the same offset as the parent (i.e. zero offset)
        entangled_partners = defaultdict(set)
        for child, parent in split_sources.items():
            entangled_partners[parent].add(child)

        # Add new qubit locations for GHZ states that on the outer rims of the unit cell
        # These states are e.g. the 'half Bell pairs' on the outside of the six-ring architecture
        for node in ghz_locations:
            parent, offset = node

            if node not in G:
                G.add_node(node)

            for child in entangled_partners[parent]:
                child_node = (child, offset)
                if child_node not in G:
                    G.add_node(child_node)

        # Create qubits and clbits
        for node in G.nodes:
            element, offset = node

            label = f"{element}:{','.join(str(o) for o in offset)}"
            G.nodes[node]["qubit"] = InQubit(xlabel=label + "_x", zlabel=label + "_z")

            if element in split_sources:
                # This bit was split from another bit. We do not add it to the error kernel,
                # but combine its measurement outcome with the parent in the circuit
                clbit = LabelledClbit(label=label)
            else:
                clbit = OutClbit(label=label)
            G.nodes[node]["clbit"] = clbit

        qubits = nx.get_node_attributes(G, "qubit")
        clbits = nx.get_node_attributes(G, "clbit")
        colors = nx.get_edge_attributes(G, "color")
        inners = nx.get_edge_attributes(G, "inner")

        # Create the distributed circuit
        qreg = QuantumRegister(name="q", bits=[qubits[node] for node in sorted(qubits)])
        creg = ClassicalRegister(name="c", bits=[clbits[node] for node in sorted(clbits)])
        super().__init__(qreg, creg)

        # Add GHZ states
        for node in ghz_locations:
            parent, offset = node

            if any(o != 0 for o in offset):
                continue

            ghz_qubits = [qubits[(parent, zero_offset)]]
            for child in sorted(entangled_partners[parent]):
                ghz_qubits.append(qubits[(child, zero_offset)])

            # Appends the corresponding N-valent frame error
            valency = len(ghz_qubits)
            try:
                ghz_frame_error = ghz_errors[valency].copy()
                ghz_frame_error.data = [clbits[(parent, zero_offset)]]

                
                ghz_frame = Frame(ghz_frame_error, ghz_qubits, [])
                parent_bit = clbits[(parent, zero_offset)]
                erasure_bit = Erasurebit(source=parent_bit, label=parent_bit.label)
                erasure_frame = Frame(ClassicalErase(), [], [erasure_bit, parent_bit])
                f1 = (ghz_frame * (1-probability)) + (erasure_frame * (probability))
                self.add_bits((erasure_bit,))
                self.append(f1.error, qargs=f1.qubits, cargs=f1.clbits)
            except KeyError as e:
                raise ValueError(f"{valency}-valent GHZ protocol not provided, aborting") from e

        # Add CZ gates
        for edge in sorted(G.edges, key=colors.get):
            qubit_pair = list(map(qubits.get, edge))
            self.cz(*qubit_pair)

        # Add X-basis measurements for all qubits
        for node in G.nodes:
            qubit, clbit = qubits[node], clbits[node]

            self.h(qubit)
            self.measure(qubit, clbit)

        # Combine measurements from split faces and edges
        for node in ghz_locations:
            parent, offset = node

            parent_bit = clbits[(parent, zero_offset)]
            for child in entangled_partners[parent]:
                child_bit = clbits[(child, zero_offset)]
                self.clcx(child_bit, parent_bit)

class DistributedCircuit(QuantumCircuit):
    """Includes SPAM errors"""
    def __init__(self, unit_cell, probability, ghz_errors: dict[int, FrameError] = None):
        if ghz_errors is None:
            ghz_errors = {}
        # unit_cell.verify_coloring()  # This doesn't work for split cells. We have to rely on the caller

        # Initialize CZ graph
        G = nx.Graph()
        zero_offset = unit_cell.dim * (0,)
        colors = defaultdict(lambda: -1, nx.get_edge_attributes(unit_cell, "color"))
        
        # Loop over inner edges (CZ gates with errors)
        edge_locations = set()
        for arc in unit_cell.filter_edges_from(dim=2, keys=True):
            face, edge, offset = arc
            edge_loc = (edge, offset)

            edge_locations.add(edge_loc)
            G.add_edge((face, zero_offset), edge_loc, inner=True, color=colors[arc])

        # Loop over outer edges (CZ gates without errors)
        for edge, offset in edge_locations:
            for _, face, offset2 in unit_cell.dual.edges(edge, keys=True):
                total_offset = tuple(i - j for i, j in zip(offset, offset2))
                new_arc = ((edge, offset), (face, total_offset))

                if not G.has_edge(*new_arc):
                    arc = (face, edge, offset2)
                    G.add_edge(*new_arc, inner=False, color=colors[arc])

        # Extract locations where to prepare GHZ states
        ghz_locations = set()

        # Messy. Sorry
        edges_faces = set(list(unit_cell.dual.filter_nodes(dim=1)) + list(unit_cell.filter_nodes(dim=1)))
        split_sources = nx.get_node_attributes(unit_cell, "split_source")
        split_connectors = nx.get_node_attributes(unit_cell, "split_connector")
        split_sources = {k: v for k, v in split_sources.items() if k in edges_faces and v in edges_faces}
        split_connectors = {k: v for k, v in split_connectors.items() if k in edges_faces and v in edges_faces}

        for node in list(G.nodes):
            element, offset = node
            connected_split = split_connectors.get(element, None)

            if connected_split:
                G.remove_node(node)  # Removes the bivalent edge (see Thesis for details)
                ghz_locations.add((connected_split, offset))
        
        # IMPORTANT set is not order-safe. Frames may be applied in a different order unless we
        # sort while looping over later on
        # Entangled partners always have the same offset as the parent (i.e. zero offset)
        entangled_partners = defaultdict(set)
        for child, parent in split_sources.items():
            entangled_partners[parent].add(child)

        # Add new qubit locations for GHZ states that on the outer rims of the unit cell
        # These states are e.g. the 'half Bell pairs' on the outside of the six-ring architecture
        for node in ghz_locations:
            parent, offset = node

            if node not in G:
                G.add_node(node)

            for child in entangled_partners[parent]:
                child_node = (child, offset)
                if child_node not in G:
                    G.add_node(child_node)

        # Create qubits and clbits
        for node in G.nodes:
            element, offset = node

            label = f"{element}:{','.join(str(o) for o in offset)}"
            G.nodes[node]["qubit"] = InQubit(xlabel=label + "_x", zlabel=label + "_z")

            if element in split_sources:
                # This bit was split from another bit. We do not add it to the error kernel,
                # but combine its measurement outcome with the parent in the circuit
                clbit = LabelledClbit(label=label)
            else:
                clbit = OutClbit(label=label)
            G.nodes[node]["clbit"] = clbit

        qubits = nx.get_node_attributes(G, "qubit")
        clbits = nx.get_node_attributes(G, "clbit")
        colors = nx.get_edge_attributes(G, "color")
        inners = nx.get_edge_attributes(G, "inner")

        # Create the distributed circuit
        qreg = QuantumRegister(name="q", bits=[qubits[node] for node in sorted(qubits)])
        creg = ClassicalRegister(name="c", bits=[clbits[node] for node in sorted(clbits)])
        super().__init__(qreg, creg)

        # Add GHZ states
        for node in ghz_locations:
            parent, offset = node

            if any(o != 0 for o in offset):
                continue

            ghz_qubits = [qubits[(parent, zero_offset)]]
            for child in sorted(entangled_partners[parent]):
                ghz_qubits.append(qubits[(child, zero_offset)])

            # Appends the corresponding N-valent frame error
            valency = len(ghz_qubits)
            try:
                ghz_frame_error = ghz_errors[valency].copy()
                ghz_frame_error.data = [clbits[(parent, zero_offset)]]
                # prob_norm = sp.simplify(sum(ghz_frame_error.probabilities))
                # if prob_norm != 1:
                #     print("Applying erasure")
                #     ghz_frame = Frame(ghz_frame_error, ghz_qubits, [])
                #     parent_bit = clbits[(parent, zero_offset)]
                #     erasure_bit = Erasurebit(source=parent_bit)
                #     erasure_frame = Frame(ClassicalErase(), [], [erasure_bit, parent_bit]) * (1-prob_norm)
                #     f1 = ghz_frame + erasure_frame
                #     self.add_bits((erasure_bit,))
                #     self.append(f1.error, qargs=f1.qubits, cargs=f1.clbits)
                # else:
                self.append(ghz_frame_error, qargs=ghz_qubits)
            except KeyError as e:
                raise ValueError(f"{valency}-valent GHZ protocol not provided, aborting") from e

        # Add state-prep noise for 'inner' non-GHZ qubits
        for node in G.nodes:
            element, offset = node

            # Somewhat messy. We are checking if this qubit is (A) inner, (B) not a parent of a split
            # and (C) not a child of a split
            if any(o != 0 for o in offset) or element in entangled_partners or element in split_sources:
                continue

            # We are preparing a |+> state, so state prep noise is given by a mixture of |+> and |->
            self.phase_flip(probability, qubits[node])

        # Add CZ gates and depolarizing noise for 'inner' gates
        for edge in sorted(G.edges, key=colors.get):
            qubit_pair = list(map(qubits.get, edge))
            self.cz(*qubit_pair)

            if inners[edge]:
                self.depolarize(probability, qubit_pair)

        # Add X-basis measurements for all qubits and flips for 'inner' qubits
        for node in G.nodes:
            qubit, clbit = qubits[node], clbits[node]

            self.h(qubit)
            self.measure(qubit, clbit)

            _, offset = node
            if all(o == 0 for o in offset):
                self.measure_flip(probability, clbit)

        # Combine measurements from split faces and edges
        for node in ghz_locations:
            parent, offset = node

            parent_bit = clbits[(parent, zero_offset)]
            for child in entangled_partners[parent]:
                child_bit = clbits[(child, zero_offset)]
                self.clcx(child_bit, parent_bit)


        ### Plotting

        # from manticore.geometry.plotting import element_coordinates, plot_coloring
        # import matplotlib.pyplot as plt

        # fig = plt.figure(figsize=(6, 6), dpi=100)
        # ax = fig.add_subplot(projection='3d', proj_type='ortho')

        # positions = element_coordinates(unit_cell)

        # # Plot coloring
        # cmap = plt.get_cmap("tab10")

        # # coordinate_offsets = dict(e_x=(0, -0.1, -0.1), e_x0=(0, 0.1, 0.1), e_y=(-0.1, 0, -0.1), e_y0=(0.1, 0, 0.1), e_z=(-0.1, -0.1, 0), e_z0=(0.1, 0.1, 0))
        
        # # Cubic six-ring
        # # offset_val = 1/6
        # # coordinate_offsets = dict(
        # #     e_x=(offset_val, 0, 0), e_x_inner=(-offset_val, 0, 0),
        # #     e_y=(0, offset_val, 0), e_y_inner=(0, -offset_val, 0),
        # #     e_z=(0, 0, offset_val), e_z_inner=(0, 0, -offset_val),
        # # )

        # # Cubic max-split
        # # offset_val = 1/3
        # # offset_val2 = 1/6
        # # coordinate_offsets = dict(
        # #     e_x=(0, -offset_val2, 0), e_y=(0, 0, -offset_val2), e_z=(-offset_val2, 0, 0),
        # #     f_x=(0, 0, offset_val2), f_y=(offset_val2, 0, 0), f_z=(0, offset_val2, 0),

        # #     f_xy=(0, 0, offset_val), f_yz=(offset_val, 0, 0), f_zx=(0, offset_val, 0),
        # #     f_yx=(0, 0, offset_val), f_zy=(offset_val, 0, 0), f_xz=(0, offset_val, 0),
        # #     f_yx_1=(0, 0, -offset_val), f_zy_1=(-offset_val, 0, 0), f_xz_1=(0, -offset_val, 0),

        # #     e_yx=(0, 0, offset_val2), e_zy=(offset_val2, 0, 0), e_xz=(0, offset_val2, 0),
        # #     e_xy=(0, 0, offset_val2), e_yz=(offset_val2, 0, 0), e_zx=(0, offset_val2, 0),
        # #     e_xy_1=(0, 0, -offset_val2), e_yz_1=(-offset_val2, 0, 0), e_zx_1=(0, -offset_val2, 0),
        # # )

        # # coordinate_offsets = dict(
        # #     f_x=(1, 0, 0), f_y=(0, 1, 0), f_z=(0, 0, 1), f_xx=(-1, 0, 0), f_yy=(0, -1, 0), f_zz=(0, 0, -1),
        # #     f_x1=(1, 0, 0), f_y1=(0, 1, 0), f_z1=(0, 0, 1), f_xx1=(-1, 0, 0), f_yy1=(0, -1, 0), f_zz1=(0, 0, -1),
        # #     e_x=(1, 0, 0), e_x1=(-1, 0, 0),
        # #     e_y=(0, 1, 0), e_y1=(0, -1, 0),
        # #     e_z=(0, 0, 1), e_z1=(0, 0, -1),
        # #     e_xx=(1, 0, 0), e_xx1=(-1, 0, 0),
        # #     e_yy=(0, 1, 0), e_yy1=(0, -1, 0),
        # #     e_zz=(0, 0, 1), e_zz1=(0, 0, -1),
        # # )
        # # coordinate_offsets = {k: tuple(0.1 * i for i in v) for k, v in coordinate_offsets.items()}

        # # coordinate_offsets = {}

        # # Plot unit cell contours
        # for edge in unit_cell.filter_nodes(dim=1):
        #     (v1, o1), (v2, o2) = ((v, offset) for _, v, offset in unit_cell.edges(edge, keys=True))
        #     p1 = positions.loc[v1] + o1 + coordinate_offsets.get(v1, 0)
        #     p2 = positions.loc[v2] + o2 + coordinate_offsets.get(v2, 0)

        #     ax.plot(*np.array([p1, p2]).T, color='k', alpha=0.1)
        
        # # Plot CZ gates
        # for (u, o1), (v, o2), dd in G.edges(data=True):
        #     p1 = positions.loc[u] + o1 + coordinate_offsets.get(u, 0)
        #     p2 = positions.loc[v] + o2 + coordinate_offsets.get(v, 0)

        #     alpha = 1 if dd["inner"] else 0
        #     ax.plot(*np.array([p1, p2]).T, color=cmap(dd["color"]), alpha=alpha)

        # # Plot entanglement
        # for l, offset in ghz_locations:
        #     for u, v in combinations(entangled_partners[l] | {l}, 2):
        #         p1 = positions.loc[u] + offset + coordinate_offsets.get(u, 0)
        #         p2 = positions.loc[v] + offset + coordinate_offsets.get(v, 0)

        #         alpha = 0.3 if any(o != 0 for o in offset) else 1
        #         ax.plot(*np.array([p1, p2]).T, color='darkturquoise', ls=(0, (2, 2)), alpha=alpha)

        # # Plot qubits
        # for u, offset in G.nodes:
        #     pos = positions.loc[u] + offset + coordinate_offsets.get(u, 0)
        #     alpha = 0 if any(o != 0 for o in offset) else 1
        #     ax.scatter(*pos, c='k', alpha=alpha)

        # # plot_coloring(unit_cell, ax, color, **coordinate_offsets)
        # ax.set_xlim(-0.5, 1.5)
        # ax.set_ylim(-0.5, 1.5)
        # ax.set_zlim(-0.5, 1.5)

        # ax.view_init(elev=30, azim=120)
        # ax.set_axis_off()
        # plt.show()

if __name__ == "__main__":
    # import sympy as sp
    # p = sp.symbols("p")
    # qc = Modicum(p)
    from manticore.simulator.base import get_single_frame, build_kernel
    # frame = get_single_frame(qc)
    # print(f"Conditioning on {qc.cregs[0][0]}")
    # conditioned_frame = frame.condition_clbit(qc.cregs[0][0])
    # kernel = build_kernel(conditioned_frame)
    # print(kernel)

    # qc = MonolithicCircuit(uc, 0.1)

    # from qiskit.transpiler.passes import ALAPSchedule
    # from qiskit.transpiler import PassManager
    # from manticore.simulator.transpiler.default_durations import UnityDurations

    # # Initialize a pass for an ALAP schedule
    # durations = UnityDurations()
    # alap = ALAPSchedule(durations)
    # pm_alap = PassManager(alap)

    # # Run ALAP on test circuit
    # qc_alap = pm_alap.run(qc)

    # print(qc_alap)

    # from manticore.geometry.library import CubicCell, DiamondCell
    # from manticore.simulator.frame_errors import DiagonalizeGHZ

    # unit_cell = DiamondCell()
    # unit_cell.color()
    # unit_cell.verify_coloring()

    # for i, j, k in ("xyz", "yzx", "zxy"):
    #     face = f"f_{i}"
    #     new_face = f"f_{i}_inner"
    #     new_edge = f"e_{j}{k}"
    #     unit_cell.simple_split(face, new_face, new_edge, (f"e_{j}", (0, 0, 0)), (f"e_{k}", (0, 0, 0)))

    # for i, j, k in ("xyz", "yzx", "zxy"):
    #     edge = f"e_{i}"
    #     new_edge = f"e_{i}_inner"
    #     new_face = f"f_{j}{k}"
    #     unit_cell.dual.simple_split(edge, new_edge, new_face, (f"f_{j}_inner", (0, 0, 0)), (f"f_{k}_inner", (0, 0, 0)))

    # unit_cell.simple_split("f_1", "f_11", "e_f1", ("e_y", (0, 0, 1)), ("e_z", (0, 1, 0)), ("e_x", (0, 1, 0)))
    # unit_cell.simple_split("f_x", "f_x1", "e_fx", ("e_y", (0, 0, 0)), ("e_z", (0, 1, 0)), ("e_1", (0, 1, 0)))
    # unit_cell.simple_split("f_y", "f_y1", "e_fy", ("e_x", (0, 0, 0)), ("e_z", (1, 0, 0)), ("e_1", (1, 0, 0)))
    # unit_cell.simple_split("f_z", "f_z1", "e_fz", ("e_y", (0, 0, 0)), ("e_1", (0, 1, 0)), ("e_x", (0, 1, 0)))

    # unit_cell.dual.simple_split("e_1", "e_11", "f_e1", ("f_x", (0, 0, 1)), ("f_z", (1, 0, 0)), ("f_y1", (1, 0, 0)))
    # unit_cell.dual.simple_split("e_x", "e_x1", "f_ex", ("f_z1", (0, 1, 0)), ("f_11", (0, 1, 0)), ("f_y1", (0, 0, 0)))
    # unit_cell.dual.simple_split("e_y", "e_y1", "f_ey", ("f_z", (1, 0, 0)), ("f_1", (1, 0, 0)), ("f_x1", (0, 0, 0)))
    # unit_cell.dual.simple_split("e_z", "e_z1", "f_ez", ("f_x", (0, 0, 0)), ("f_1", (1, 0, 0)), ("f_y1", (1, 0, 0)))
    # units = dict(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))

    # from manticore.geometry.library import DoubleEdgeGHZCubicCell
    # unit_cell = DoubleEdgeGHZCubicCell()

    # circuit = DistributedCircuit(unit_cell, 0.1, {4: DiagonalizeGHZ(4, 0)})

    # circuit = DistributedCircuit(unit_cell, 0.1, {2: DiagonalizeGHZ(2, 0)})
    # circuit = DistributedCircuit(unit_cell, 0.1, {4: DiagonalizeGHZ(4, 0)})

    