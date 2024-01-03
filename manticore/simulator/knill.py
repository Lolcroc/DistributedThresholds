# Python
from itertools import chain
from more_itertools import unique_everseen

# Math
import numpy as np
import galois as gl
GF2 = gl.GF(2)

# Qiskit
from qiskit.circuit import Instruction, Qubit, Clbit
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGInNode, DAGOutNode, DAGOpNode
dummy = DAGCircuit()
_bits_in_condition = dummy._bits_in_condition # Can call without DAG object

# Manticore
from manticore.simulator.circuit import InQubit, OutQubit, IOQubit, InClbit, OutClbit, IOClbit

class Knill:

    def __init__(self):
        self.in_qubits = []
        self.in_clbits = []
        self.out_qubits = []
        self.out_clbits = []

        self.table = GF2.Identity(0)

    @property
    def num_in_qubits(self):
        return len(self.in_qubits)

    @property
    def num_in_clbits(self):
        return len(self.in_clbits)
    
    @property
    def num_in_bits(self):
        return 2 * self.num_in_qubits + self.num_in_clbits

    @property
    def num_out_qubits(self):
        return len(self.out_qubits)

    @property
    def num_out_clbits(self):
        return len(self.out_clbits)
    
    @property
    def num_out_bits(self):
        return 2 * self.num_out_qubits + self.num_out_clbits
    
    # Input table
    def from_in(self, qubits, clbits):
        qubit_indices = list(map(self.in_qubits.index, qubits))
        clbit_indices = list(map(self.in_clbits.index, clbits))
        
        cl_table_indices = clbit_indices
        x_table_indices = [self.num_in_clbits + i for i in qubit_indices]
        z_table_indices = [self.num_in_clbits + self.num_in_qubits + i for i in qubit_indices]
        return self.table[cl_table_indices + x_table_indices + z_table_indices]

    @property
    def classical_in(self):
        return self.table[:self.num_in_clbits]

    @classical_in.setter
    def classical_in(self, val):
        self.table[:self.num_in_clbits] = val

    @property
    def X_in(self):
        return self.table[self.num_in_clbits:(self.num_in_clbits + self.num_in_qubits)]

    @X_in.setter
    def X_in(self, val):
        self.table[self.num_in_clbits:(self.num_in_clbits + self.num_in_qubits)] = val

    @property
    def Z_in(self):
        return self.table[(self.num_in_clbits + self.num_in_qubits):]

    @Z_in.setter
    def Z_in(self, val):
        self.table[(self.num_in_clbits + self.num_in_qubits):] = val

    # Output table

    @property
    def classical_out(self):
        return self.table[:, :self.num_out_clbits]

    @classical_out.setter
    def classical_out(self, val):
        self.table[:, :self.num_out_clbits] = val

    @property
    def X_out(self):
        return self.table[:, self.num_out_clbits:(self.num_out_clbits + self.num_out_qubits)]

    @X_out.setter
    def X_out(self, val):
        self.table[:, self.num_out_clbits:(self.num_out_clbits + self.num_out_qubits)] = val

    @property
    def Z_out(self):
        return self.table[:, (self.num_out_clbits + self.num_out_qubits):]

    @Z_out.setter
    def Z_out(self, val):
        self.table[:, (self.num_out_clbits + self.num_out_qubits):] = val

    # In-out tables
    
    @property
    def classical(self):
        return self.table[:self.num_in_clbits, :self.num_out_clbits]

    @classical.setter
    def classical(self, val):
        self.table[:self.num_in_clbits, :self.num_out_clbits] = val

    @property
    def X(self):
        return self.table[self.num_in_clbits:(self.num_in_clbits + self.num_in_qubits), self.num_out_clbits:(self.num_out_clbits + self.num_out_qubits)]

    @X.setter
    def X(self, val):
        self.table[self.num_in_clbits:(self.num_in_clbits + self.num_in_qubits), self.num_out_clbits:(self.num_out_clbits + self.num_out_qubits)] = val

    @property
    def Z(self):
        return self.table[(self.num_in_clbits + self.num_in_qubits):, (self.num_out_clbits + self.num_out_qubits):]

    @Z.setter
    def Z(self, val):
        self.table[(self.num_in_clbits + self.num_in_qubits):, (self.num_out_clbits + self.num_out_qubits):] = val

    def add_in_qubit(self, qubit):
        self.table = np.insert(self.table, [self.num_in_clbits + self.num_in_qubits, self.num_in_bits], 0, axis=0)
        self.in_qubits.append(qubit)

    def add_out_qubit(self, qubit):
        self.table = np.insert(self.table, [self.num_out_clbits + self.num_out_qubits, self.num_out_bits], 0, axis=1)
        self.out_qubits.append(qubit)

    def add_in_clbit(self, clbit):
        self.table = np.insert(self.table, [self.num_in_clbits], 0, axis=0)
        self.in_clbits.append(clbit)

    def add_out_clbit(self, clbit):
        self.table = np.insert(self.table, [self.num_out_clbits], 0, axis=1)
        self.out_clbits.append(clbit)

    def remove_in_qubit(self, qubit):
        qubit_index = self.in_qubits.index(qubit)
        self.table = np.delete(self.table, [self.num_in_clbits + qubit_index, self.num_in_clbits + self.num_in_qubits + qubit_index], axis=0)
        self.in_qubits.remove(qubit)

    def remove_out_qubit(self, qubit):
        qubit_index = self.out_qubits.index(qubit)
        self.table = np.delete(self.table, [self.num_out_clbits + qubit_index, self.num_out_clbits + self.num_out_qubits + qubit_index], axis=1)
        self.out_qubits.remove(qubit)

    def remove_in_clbit(self, clbit):
        clbit_index = self.in_clbits.index(clbit)
        self.table = np.delete(self.table, [clbit_index], axis=0)
        self.in_clbits.remove(clbit)

    def remove_out_clbit(self, clbit):
        clbit_index = self.out_clbits.index(clbit)
        self.table = np.delete(self.table, [clbit_index], axis=1)
        self.out_clbits.remove(clbit)

    def prepend(self, node: DAGNode):
        if isinstance(node, DAGInNode):
            bit = node.wire
            if isinstance(bit, InQubit):
                _prepend_in_qubit(self, bit)
            elif isinstance(bit, Qubit):
                _prepend_start_qubit(self, bit)
            elif isinstance(bit, InClbit):
                _prepend_in_clbit(self, bit)
            else:
                _prepend_start_clbit(self, bit)
        elif isinstance(node, DAGOutNode):
            bit = node.wire
            if isinstance(bit, OutQubit):
                _prepend_out_qubit(self, bit)
            elif isinstance(bit, Qubit):
                _prepend_finish_qubit(self, bit)
            elif isinstance(bit, OutClbit):
                _prepend_out_clbit(self, bit)
            else:
                _prepend_finish_clbit(self, bit)
        else:
            op, qargs, cargs = node.op, node.qargs, node.cargs
            
            # Correct condition
            cargs = list(unique_everseen(cargs + _bits_in_condition(op.condition)))

            qubit_indices = map(self.in_qubits.index, qargs)
            clbit_indices = map(self.in_clbits.index, cargs)

            _PREPEND_OPS[op.name](self, *chain(qubit_indices, clbit_indices))

    def __repr__(self):
        return f"{self.__class__.__name__}(table=\n{self.table}, in_clbits={self.in_clbits}, in_qubits={self.in_qubits}, out_clbits={self.out_clbits}, out_qubits={self.out_qubits})"

# %%

"""
Time-reversed operations
"""

def _prepend_in_qubit(knill, qubit):
    pass

def _prepend_in_clbit(knill, clbit):
    pass

def _prepend_start_qubit(knill, qubit):  # Time-reversed means we remove the qubit
    knill.remove_in_qubit(qubit)

def _prepend_start_clbit(knill, clbit):  # Time-reversed means we remove the clbit
    knill.remove_in_clbit(clbit)

def _prepend_out_qubit(knill, qubit):
    knill.add_in_qubit(qubit)
    knill.add_out_qubit(qubit)
    knill.X[-1, -1] = 1
    knill.Z[-1, -1] = 1

def _prepend_out_clbit(knill, clbit):
    knill.add_in_clbit(clbit)
    knill.add_out_clbit(clbit)
    knill.classical[-1, -1] = 1
    
def _prepend_finish_qubit(knill, qubit):  # Time-reversed means we add the qubit
    knill.add_in_qubit(qubit)

def _prepend_finish_clbit(knill, clbit):  # Time-reversed means we add the clbit
    knill.add_in_clbit(clbit)

# Methods below indexed by integer

def _prepend_measure(knill, qubit, clbit):
    knill.X_in[qubit] ^= knill.classical_in[clbit]

def _prepend_reset(knill, *qubits):
    for qubit in qubits:
        knill.X_in[qubit] = knill.Z_in[qubit] = 0

def _prepend_i(knill, *args):
    pass

def _prepend_x(knill, qubit, condition=None):
    if condition:
        knill.classical_in[condition] ^= knill.Z_in[qubit]

def _prepend_y(knill, qubit, condition=None):
    if condition:
        knill.classical_in[condition] ^= knill.X_in[qubit]
        knill.classical_in[condition] ^= knill.Z_in[qubit]

def _prepend_z(knill, qubit, condition=None):
    if condition:
        knill.classical_in[condition] ^= knill.X_in[qubit]

def _prepend_h(knill, qubit):
    x, z = knill.X_in[qubit], knill.Z_in[qubit]
    tmp = x.copy()
    knill.X_in[qubit], knill.Z_in[qubit] = z, tmp

def _prepend_s(knill, qubit):
    knill.X_in[qubit] ^= knill.Z_in[qubit]

def _prepend_sdg(knill, qubit):
    knill.X_in[qubit] ^= knill.Z_in[qubit]

def _prepend_cx(knill, control, target):
    knill.X_in[control] ^= knill.X_in[target]
    knill.Z_in[target] ^= knill.Z_in[control]

def _prepend_cz(knill, control, target):
    knill.X_in[control] ^= knill.Z_in[target]
    knill.X_in[target] ^= knill.Z_in[control]

def _prepend_swap(knill, qubit0, qubit1):
    x0, z0 = knill.X_in[qubit0].copy(), knill.Z_in[qubit0].copy()
    knill.X_in[qubit0], knill.Z_in[qubit0] = knill.X_in[qubit1], knill.Z_in[qubit1]
    knill.X_in[qubit1], knill.Z_in[qubit1] = x0, z0

def _prepend_clcx(knill, control, target):
    knill.classical_in[control] ^= knill.classical_in[target]

_PREPEND_OPS = {
    "measure": _prepend_measure,
    "reset": _prepend_reset,
    "set_stabilizer": _prepend_reset,
    "i": _prepend_i,
    "id": _prepend_i,
    "iden": _prepend_i,
    "x": _prepend_x,
    "y": _prepend_y,
    "z": _prepend_z,
    "h": _prepend_h,
    "s": _prepend_s,
    "sdg": _prepend_sdg,
    "sinv": _prepend_sdg,
    "cx": _prepend_cx, 
    "cz": _prepend_cz, 
    "swap": _prepend_swap,
    "clcx": _prepend_clcx,  # Custom. Classical CX
    "frame_error": _prepend_i  # Custom. Do nothing
}
