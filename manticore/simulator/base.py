# Python
from __future__ import annotations
from itertools import permutations
from more_itertools import unique_everseen
from functools import reduce
import operator

# Math
import numpy as np
import sympy as sp
import pandas as pd
import galois as gl
GF2 = gl.GF(2)

# Qiskit
from qiskit.circuit import QuantumCircuit, Qubit, Clbit, QuantumRegister, ClassicalRegister
from qiskit.dagcircuit import DAGOpNode, DAGInNode, DAGOutNode
from qiskit.converters import circuit_to_dag

# Custom
from manticore.simulator.frame_errors import FrameError
from manticore.simulator.circuit import Erasurebit, OutClbit, OutQubit
from manticore.simulator.linalg import twiddle_indices, unbin_matrix, bin_matrix
from manticore.simulator.knill import Knill

from manticore.errors import ErrorKernel, ErrorModel

class Frame:
    """A representation of a Clifford-friendly stochastic error string, as a combination of one or more of:
        1. Pauli errors on qubits (e.g. depolarizing noise)
        2. Classical bit-flips on clbits (e.g. noisy measurements)
    
    Erasure errors are captured through by a special 'flag' classical bit with a bit-flip representing erasure.
    The corresponding qubit(s) will be maximally depolarized.
    """
    __slots__ = ("error", "qubits", "clbits")

    def __init__(self, error: FrameError, qubits: list[Qubit], clbits: list[Clbit]):
        """Create a new Frame over a certain subset of qubits and clbits (classical bits).

        Args:
            error (FrameError): A Qiskit Instruction representing the error. See FrameError for more details.
            qubits (list[Qubit]): The qubits over which to define the Frame.
            clbits (list[Clbit]): The classical bits over which to define the Frame.

        Raises:
            ValueError: If the number of qubits in error does not agree with the given lists of qubits and clbits.
        """
        if len(qubits) != error.num_qubits or len(clbits) != error.num_clbits:
            raise ValueError(f"Wrong amount of qubits {qubits} and/or clbits {clbits}, expected {error.num_qubits} and {error.num_clbits}")

        self.error = error
        self.qubits = qubits
        self.clbits = clbits

    def __mul__(self, other):
        if isinstance(other, Frame):
            return Frame.merge(self, other, _outer_xor, _outer_mul)
        elif isinstance(other, Knill):
            # Multiply error with knill matrix on relevant rows
            bin_indices = bin_matrix(self.indices, self.num_bits, galois=True)
            new_bin_indices = bin_indices @ other.from_in(self.qubits, self.clbits)

            # Remove the zero columns (qubits and clbits that are not relevant)
            to_drop = []
            qubits, clbits = [], []
            M, N = len(other.out_clbits), len(other.out_qubits)

            # Check for clbits to remove
            nonzero_columns = np.bitwise_or.reduce(new_bin_indices, axis=0)
            for i, clbit in enumerate(other.out_clbits):
                if nonzero_columns[i]:
                    clbits.append(clbit)
                else:
                    to_drop.append(i)

            # Check for qubits to remove
            for i, qubit in enumerate(other.out_qubits):
                x_ind = M + i
                z_ind = M + N + i
                if nonzero_columns[x_ind] or nonzero_columns[z_ind]:  # If at least one nonzero
                    qubits.append(qubit)
                else:
                    to_drop.extend((x_ind, z_ind))

            # Remove columns and convert the matrix back to integers
            new_bin_indices = np.delete(new_bin_indices, to_drop, axis=1)
            indices = unbin_matrix(new_bin_indices)   # May still need to merge duplicate indices; FrameError __init__ takes care of this

            return Frame(FrameError(len(qubits), len(clbits), self.probabilities, indices, label=self.label, check_normalized=False, data=self.data), qubits, clbits)
        elif isinstance(other, (sp.Expr, float)):
            new_probabilities = [other * p for p in self.probabilities]
            qubits = self.qubits.copy()
            clbits = self.clbits.copy()
            return Frame(FrameError(len(qubits), len(clbits), new_probabilities, self.indices, label=self.label, check_normalized=False, data=self.data), qubits, clbits)
        else:
            raise ValueError(f"Cannot multiply {self} with {other}")

    def __imul__(self, other):
        frame = self * other

        self.error = frame.error
        self.qubits = frame.qubits
        self.clbits = frame.clbits

        return self

    def __add__(self, other):
        if isinstance(other, Frame):
            return Frame.merge(self, other, _concat, _concat)
        else:
            raise ValueError(f"Cannot add {self} with {other}")

    @classmethod
    def merge(cls, frame1, frame2, index_merger, prob_merger):
        # Get the shared qubits and clbits
        qubits = list(unique_everseen(frame1.qubits + frame2.qubits))
        clbits = list(unique_everseen(frame1.clbits + frame2.clbits))

        # Transform indices to agree with the new shared qubits and clbits
        new_indices1 = twiddle_indices(frame1.indices, frame1.qubits, frame1.clbits, qubits, clbits)
        new_indices2 = twiddle_indices(frame2.indices, frame2.qubits, frame2.clbits, qubits, clbits)

        # Combine the indices and probabilities of both frames
        indices = index_merger(new_indices1, new_indices2)
        probabilities = prob_merger(frame1.probabilities, frame2.probabilities)

        return cls(FrameError(len(qubits), len(clbits), probabilities, indices, check_normalized=False, data=frame1.error.data + frame2.error.data), qubits, clbits)

    @property
    def num_bits(self):
        return self.error.num_bits

    @property
    def num_qubits(self):
        return self.error.num_qubits

    @property
    def num_clbits(self):
        return self.error.num_clbits

    @classmethod
    def empty(cls):
        return cls(FrameError(0, 0, [1]), [], [])

    @property
    def probabilities(self):
        return self.error.probabilities

    @probabilities.setter
    def probabilities(self, probs):
        self.error.probabilities = probs

    @property
    def indices(self):
        return self.error.indices

    @indices.setter
    def indices(self, inds):
        self.error.indices = inds

    @property
    def data(self):
        return self.error.data

    def twiddle_qubits(self):
        qubit_permutations = list(permutations(self.qubits))
        N = len(qubit_permutations)

        all_probabilities = np.tile(self.probabilities, N) / N
        all_indices = []

        for perm in qubit_permutations:
            perm_indices = twiddle_indices(self.indices, self.qubits, self.clbits, perm, self.clbits)
            all_indices.extend(list(perm_indices))

        return Frame(
            FrameError(len(self.qubits), len(self.clbits), all_probabilities, all_indices, label=self.label, check_normalized=False, data=self.data),
            self.qubits, self.clbits
        )

    def condition_clbit(self, clbit: Clbit, flip: bool = False, drop_bit: bool = True) -> Frame:
        """Calculate a new frame conditioned on a classical bit having value `flip`.

        This operations basically drops those rows from the frame where the classical bit has value `flip`.
        The remaining rows will not be normalized.

        Args:
            clbit (Clbit): The classical bit whose value should be `flip`.

        Raises:
            ValueError: If clbit is not in this Frame.

        Returns:
            Frame: The newly conditioned frame.
        """
        if clbit not in self.clbits:
            raise ValueError(f"Cannot condition {clbit} that is not in {self}")

        mask = 1 << (self.num_bits - 1 - self.clbits.index(clbit))

        new_indices = []
        new_probabilities = []
        for i, p in zip(self.indices, self.probabilities):
            if flip == (i & mask > 0):
                new_indices.append(i)
                new_probabilities.append(p)

        new_clbits = self.clbits.copy()
        new_qubits = self.qubits.copy()

        if drop_bit:
            new_clbits.remove(clbit)
            new_indices = twiddle_indices(new_indices, self.qubits, self.clbits, new_qubits, new_clbits)

        return Frame(FrameError(len(new_qubits), len(new_clbits), new_probabilities, new_indices, check_normalized=False, data=self.data), new_qubits, new_clbits)

    @property
    def label(self):
        return self.error.label

    def copy(self):
        return Frame(self.error.copy(), self.qubits.copy(), self.clbits.copy())

    def __repr__(self):
        return f"{self.__class__.__name__}(errors={self.error.errors}, qubits={self.qubits}, clbits={self.clbits})"

def _outer_xor(a, b):
    return np.bitwise_xor.outer(a, b).flatten()

def _outer_mul(a, b):
    return np.outer(a, b).flatten()

def _concat(a, b):
    return np.concatenate((a, b))

def pandas_indices(qubits, clbits):
    # The order is important. FrameError has clbits up front, then X-part of qubits and then the Z-part.
    column_tuples = []
    for clbit in clbits:
        column_tuples.append((clbit, ''))  # Needs an extra empty string to create MultiIndex
    for label in "xz":
        for qubit in qubits:  # Inner loop means first all X then all Z labels
            column_tuples.append((qubit, label))

    return pd.MultiIndex.from_tuples(column_tuples)

def to_pandas(frame: Frame) -> pd.DataFrame:
    binary_indices = bin_matrix(frame.indices, frame.num_bits)
    columns = pandas_indices(frame.qubits, frame.clbits)

    return pd.DataFrame(binary_indices, columns=columns, index=frame.indices, dtype=bool)

def pandas_indices_flat(qubits, clbits):
    # The order is important. FrameError has clbits up front, then X-part of qubits and then the Z-part.
    columns = []
    for clbit in clbits:
        columns.append(f"{clbit}")  # Needs an extra empty string to create MultiIndex
    for label in "xz":
        for qubit in qubits:  # Inner loop means first all X then all Z labels
            columns.append(f"{qubit}_{label}")

    return columns

def to_pandas_float(frame: Frame) -> pd.DataFrame:
    binary_indices = bin_matrix(frame.indices, frame.num_bits)
    columns = pandas_indices_flat(frame.qubits, frame.clbits)

    return pd.DataFrame(binary_indices, columns=columns, index=frame.indices, dtype=bool)

def get_frames(circuit: QuantumCircuit) -> Frame:
    dag = circuit_to_dag(circuit)
    knill = Knill()

    # Ensures reproducable topological ordering on the same circuit
    def sort_node(node):
        idx = ""

        if isinstance(node, (DAGInNode, DAGOutNode)):
            idx += str(circuit.find_bit(node.wire).index)
        else:
            for bit in node.qargs + node.cargs:
                idx += str(circuit.find_bit(bit).index)

        return idx

    # Loop over circuit instructions in reversed order
    for node in reversed(list(dag.topological_nodes(key=sort_node))):
        knill.prepend(node)  # Add instructions to the Knill operator

        # Yield if instruction is an error
        if isinstance(node, DAGOpNode) and isinstance(node.op, FrameError):
            frame = Frame(node.op, node.qargs, node.cargs)
            # dag.remove_op_node(node)  # Not really necessary

            new_frame = frame * knill  # Multiplies the frame through the knill operator

            if new_frame.num_bits > 0:
                yield new_frame

# Combines frames from get_frames
# This method can be slow depending on how big the frames are
def get_single_frame(circuit: QuantumCircuit, expand=False) -> Frame:
    frame = reduce(operator.mul, get_frames(circuit), Frame.empty())

    if expand:
        to_qubits = [q for q in circuit.qubits if isinstance(q, OutQubit)]
        to_clbits = [c for c in circuit.clbits if isinstance(c, OutClbit)]

        indices = twiddle_indices(frame.indices, frame.qubits, frame.clbits, to_qubits, to_clbits)
        return Frame(FrameError(len(to_qubits), len(to_clbits), frame.probabilities, indices, check_normalized=False), to_qubits, to_clbits)

    return frame

def build_kernel(frame: Frame, condition=None) -> ErrorKernel:
    df = to_pandas(frame)

    cl_erasures = []  # Labels for classical erasures
    qu_erasures = []  # Labels for quantum erasures
    error_support = []  # Labels for the rest of the frame

    # This loop does two things:
    # 1. Create lists of erasure bits to move to the front of the frame (since this is what ErrorSet needs)
    # 2. Duplicate every single quantum erasure bit to two bits, one for each of the X/Z parts
    for bit, label in df.columns:
        if isinstance(bit, Erasurebit):
            source_bit = bit.source

            if isinstance(source_bit, Qubit):
                df[bit, 'x'] = df[bit, 'z'] = df[bit]
                qu_erasures.append((bit, 'x'))
                qu_erasures.append((bit, 'z'))
            else:
                cl_erasures.append((bit, label))
        else:
            error_support.append((bit, label))

    # Combine classical and quantum erasures. Classical goes in front
    erasure_support = cl_erasures + qu_erasures

    # Reindex the erasures in front, then the rest
    df = df.reindex(columns=erasure_support + error_support)
    indices = unbin_matrix(df.values)
    
    # Convert probabilities
    probabilities = np.asarray(frame.probabilities)
    # if probabilities.dtype != float:
    #     probabilities = np.array([p._symbol_expr for p in frame.probabilities])

    # Map qubits and clbits to str locations
    error_locations = _map_bits(error_support)
    erasure_locations = _map_bits(erasure_support)

    return ErrorKernel(
        probabilities, 
        error_locations=error_locations, 
        erasure_locations=erasure_locations, 
        indices=indices, 
        name=frame.label, 
        condition=condition, 
        check_normalized=False
    )

def _map_bits(support):
    return [[getattr(bit, f"{label}label", f"{label}{bit}") for bit, label in support]]
