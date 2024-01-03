# Python
from itertools import product
from more_itertools import powerset
from functools import reduce
import operator
import warnings

# Math
import numpy as np
import sympy as sp
import pandas as pd

# Qiskit
from qiskit.circuit import Instruction, QuantumCircuit, Qubit, Clbit
from qiskit.circuit.bit import Bit
from qiskit.circuit.exceptions import CircuitError

# Me
from manticore.simulator.circuit import Erasurebit

def merge_numpy(indices, probabilities):
    summed_probabilities = np.bincount(indices, weights=probabilities)
    indices = np.flatnonzero(summed_probabilities)
    probabilities = summed_probabilities[indices]
    return indices, probabilities

def merge_sympy(indices, probabilities):
    summed_probabilities = (max(indices) + 1) * [0]
    nonzero_indices = set()

    for i, p in zip(indices, probabilities):
        summed_probabilities[i] += sp.expand(p)
        nonzero_indices.add(i)
    
    indices = sorted(nonzero_indices)
    probabilities = list(summed_probabilities[i] for i in indices)
    return indices, probabilities

class FrameError(Instruction):
    """Parameterized instruction amortizes noise calculations for different parameter sets."""

    def __init__(self, num_qubits: int, num_clbits: int, probabilities: list, indices: list = None, label : str = None, check_normalized : bool = True, float_tol: float = 1e-6, data = None):
        """Create a new FrameError (a combination of quantum bit/phaseflips and classical bitflips).

        Args:
            num_qubits (int): The number of qubits that this error supports.
            num_clbits (int): The number of classical bits that this error supports.
                Because qubits support X/Z errors (2 bits), and classical bits support bit-flips (1 bit), the total number
                of bits of this frame error is S = 2 * num_qubits + num_clbits.
            probabilities (list): The probabilities for each of the error configurations on the qubits and classical bits.
            indices (list, optional): Integers such that the binary representation bin(indices[i]) corresponds to a bitmask
                for the locations of errors with probability probabilities[i]. Indices and probabilities should have the same 
                length. More details below. Defaults to None, i.e. indices are implicitly given as a list(range(2**S))).
            label (str, optional): Instruction label for this error. Defaults to None.
            check_normalized (bool, optional): Check if probabilities sum to unity. Defaults to True.
            float_tol (float, optional): Absolute error tolerance for checking if probabilities sum to unity.

            The ordering of indices is subtle. Given num_qubits = N and num_clbits = M, the index i should be interpreted
            as a binary representation bin(i) = b_{2*N + M - 1} ... b_0 (b_0 is the LSB), such that:
                * b_{2*N + M - 1} ... b_{2*N}   : M bits corresponding to classical bit flips (e.g. from measurement/erasure)
                * b_{2*N - 1} ... b_{N}         : N bits corresponding to X flips
                * b_{N-1} ... b_0               : N bits corresponding to Z flips

            Within each substring, the ordering of lowest-to-highest bit is left-to-right (MSB to LSB). This ordering is
            consistent with https://arxiv.org/pdf/quant-ph/0406196.pdf, except that we have prepended M bits for the
            classical part of the frame. This means that with a frame supported on e.g. clbits=[A, B], the index i = 1
            corresponds to a bitmask 01 on top of AB, which means that there is an error on B (and not A!). See below
            for examples.

        Examples:
            * FrameError(num_qubits=2, num_clbits=0, probabilities=[p, 1-p], indices=[1, 8]) # Corresponds to

                bin(1) = 0b0001 = IZ = Z_1 with probability p
                bin(8) = 0b1000 = XI = X_0 with probability 1-p

            * FrameError(num_qubits=1, num_clbits=1, probabilities=[p/4, p/4, p/4, p/4], indices=[0, 1, 4, 5]) # Corresponds to

                bin(0) = 0b000 = 0I = no error with probability p/4
                bin(1) = 0b001 = 0Z = Z error with probability p/4
                bin(4) = 0b100 = 1I = classical bit-flip with probability p/4
                bin(5) = 0b101 = 1Z = classical bit-flip and Z error with probability p/4
        """
        size = 2 * num_qubits + num_clbits
        if size > 32:
            raise RuntimeError(f"cannot create FrameError on more than 32 bits")

        # If no indices given, assume the list of probabilities is full and indices are implicit
        if indices is None:
            indices = range(2**size)

        if len(probabilities) != len(indices):
            raise ValueError(f"Probs {probabilities} has wrong len {len(probabilities)}, expected {len(indices)}")

        # Sum probabilities that share the same index and reconstruct a sparse representation of the indices and probabilities
        # indices, probabilities = merge_numpy(indices, probabilities)
        indices, probabilities = merge_sympy(indices, probabilities)

        if len(probabilities) > 2**size:
            raise ValueError(f"Probs {probabilities} has wrong len {len(probabilities)}, expected at most {2**size}")
        if check_normalized and abs((s := sp.simplify(sum(probabilities))) - 1) > float_tol:
            raise ValueError(f"Probabilities {probabilities} not normalized, found sum={s}")

        params = list(indices) + list(probabilities)  # Order is important
        super().__init__("frame_error", num_qubits, num_clbits, params=params, duration=0, label=label)

        self.data = data or []

    @property
    def num_bits(self):
        return 2 * self.num_qubits + self.num_clbits

    @property
    def size(self):
        return len(self.params) // 2  # First half indices, second half probabilities

    @property
    def indices(self):
        return self.params[:self.size]

    @indices.setter
    def indices(self, inds):
        if len(inds) != self.size:
            raise ValueError(f"Cannot set indices {inds} on existing size {self.size}")
        self.params[:self.size] = inds

    # TODO lifehack - we can create a custom class that inherits both an array type (eg np.ndarray or sympy/symengine array)
    # AND the ParameterExpression type, and Qiskit will recognize those parameters accordingly
    @property
    def probabilities(self):
        return self.params[self.size:]

    @probabilities.setter
    def probabilities(self, probs):
        if len(probs) != self.size:
            raise ValueError(f"Cannot set probabilities {probs} on existing size {self.size}")
        self.params[self.size:] = probs

    @property
    def errors(self):
        return f"[{', '.join(self._to_label(i) for i in self.indices)}]"

    def __repr__(self):
        return f"{self.__class__.__name__}(num_qubits={self.num_qubits}, num_clbits={self.num_clbits}, errors={self.errors})"

    def _to_label(self, i):
        paulis = {'00': 'I', '01': 'Z', '10': 'X', '11': 'Y'}

        M, N = self.num_clbits, self.num_qubits
        bin_rep = f"{i:0{2*N + M}b}"

        # The bit-qubit ordering is vital here. See the class description for an explanation.
        cl_bits, x_bits, z_bits = bin_rep[:M], bin_rep[M:(M+N)], bin_rep[(M+N):]

        # The 0-th qubit is printed first
        return f"{cl_bits}{''.join(paulis[x_bits[i] + z_bits[i]] for i in range(N))}"

class MeasureFlip(FrameError):
    """Standard classical measurement error."""

    def __init__(self, prob):
        super().__init__(0, 1, [1 - prob, prob])

    def broadcast_arguments(self, qargs, cargs):
        for carg in cargs:
            yield [], carg

    @property
    def label(self):
        return f"m_flip({self.params[-1]})"

class Depolarize(FrameError):
    """Standard depolarizing channel. The probability of the identity term is 1 - prob."""

    def __init__(self, num_qubits, prob):
        num_errors = 4**num_qubits - 1
        p = prob / num_errors
        p_iden = 1 - prob

        super().__init__(num_qubits, 0, [p_iden] + num_errors * [p])

    def broadcast_arguments(self, qargs, cargs):
        for qarg in qargs:
            yield qarg, []

    @property
    def label(self):
        return f"dp_{self.num_qubits}" # ({self.params[-1]})

class BitFlip(FrameError):
    """Quantum bit-flip channel."""

    def __init__(self, prob):
        super().__init__(1, 0, [1 - prob, prob], [0, 2])

class PhaseFlip(FrameError):
    """Quantum phase-flip channel."""

    def __init__(self, prob):
        super().__init__(1, 0, [1 - prob, prob], [0, 1])

class Dephase(PhaseFlip):
    """Simple dephasing channel."""

    def __init__(self, duration, T2):
        sq = np.exp(-duration/(2 * T2))  # sqrt(1-gamma)

        super().__init__((1 - sq)/2)  # Sanity check: if T2 = inf, then sq = 1 and so p_phaseflip = 0

    @property
    def label(self):
        return f"dephase({self.params[-1]})"

class Relax(FrameError):
    """This channel has two interpretations.
    
    1.  Same-temperature amplitude damping (p=1/2), i.e. the stationary state is maximally mixed.
        The Kraus operators of this channel are in fact Pauli without approximation.

    2.  Pauli-twirled 0 Kelvin amplitude damping (p=1), i.e. the stationary state is |0>.
        Because this channel is not Pauli, twirling provides an approximation of actual amplitude damping!
    """

    def __init__(self, duration, T1):
        sq = np.exp(-duration/(2 * T1))  # sqrt(1-gamma)
        g4 = (1 - np.exp(-duration/T1))/4  # gamma/4 prefactor

        super().__init__(1, 0, [(1+sq)/2 - g4, (1-sq)/2 - g4, g4, g4])

    @property
    def label(self):
        return f"relax({self.params[-1]})"

class Decohere(FrameError):
    """A combination of both dephasing and relaxation, where relaxation is towards I/2 (see above).
    
    For T1 = T2, use the simpler form below.
    """

    def __init__(self, duration, T1, T2):
        if T1 == T2:
            warnings.warn(f"Decoherence channel with T1 = T2 = {T1} used. Use the simpler channel UnaryDecohere instead of Decohere.")

        sq = np.exp((-duration/2) * (T1 + T2) / (T1 * T2))  # sqrt(1-gamma)*sqrt(1-lambda), where gamma is for T1 and lambda for T2
        g4 = (1 - np.exp(-duration/T1))/4  # gamma/4 prefactor

        super().__init__(1, 0, [(1+sq)/2 - g4, (1-sq)/2 - g4, g4, g4])

    @property
    def label(self):
        return f"decohere({self.params[-1]})"

class UnaryDecohere(Depolarize):
    """A decoherence channel with T1 = T2 = T. This channel is a depolarizing channel with prob = 3 * gamma / 4."""

    def __init__(self, duration, T):
        super().__init__(1, 0.75 * (1 - np.exp(-duration/T)))

    @property
    def label(self):
        return f"simple_d({self.params[-1]})"

class DiagonalizeGHZ(FrameError):
    def __init__(self, num_qubits, prob):
        # Probabilities are all equally divided over diagonal entries
        num_errors = 2**num_qubits - 1
        probabilities = [1 - prob] + num_errors * [prob / num_errors]
        
        # 1 is the Z error, rest are X errors
        generators = [1] + list(1 << (np.arange(num_qubits + 1, 2*num_qubits)))

        indices = []
        for errors in powerset(generators):
            indices.append(reduce(operator.xor, errors, 0))

        super().__init__(num_qubits, 0, probabilities=probabilities, indices=indices)

    @property
    def label(self):
        return f"dghz_{self.num_qubits}" # ({self.params[-1]})

class QuantumErase(FrameError):
    """A simple erasure channel over 1 qubit and 1 erasure bit (classical).

    If rho is the qubit state, and |0>/|1> is the erasure bit state, then
    E(rho) = (1 - p)*rho ⊗ |0X0| + p*(I/2) ⊗ |1X1|,

    where I/2 = 1/4 * (rho + X rho X + Y rho Y + Z rho Z) is a depolarizing channel with p=1/4.

    The erasure bit is 1 iff erasure took place - programmers should take care that
    this bit does not interact with the circuit, since the erasure bit counts as a flag
    and not as a measurement outcome.
    """

    def __init__(self, probability=1):
        super().__init__(1, 1, [1-probability] + 4 * [probability/4], [0, 4, 5, 6, 7])

    def broadcast_arguments(self, qargs, cargs):
        for qarg, carg in zip(qargs, cargs):
            yield qarg, carg

    @property
    def label(self):
        return f"qerase_({self.params[-1]})"

class ClassicalErase(FrameError):
    """A simple erasure channel over 1 classical bit and 1 erasure bit (classical).

    If |m>=|0>/|1> is the classical bit state, and |0>/|1> is the erasure bit state, then
    E(|m>) = (1 - p)*|m> ⊗ |0X0| + p*(I/2) ⊗ |1X1|,
    
    where I/2 = 1/2(|0X0| + |1X1|) is a classical bit-flip channel with p=1/2.

    The erasure bit is 1 iff erasure took place - programmers should take care that
    this bit does not interact with the circuit, since the erasure bit counts as a flag
    and not as a measurement outcome.
    """

    def __init__(self, probability=1):
        super().__init__(0, 2, [1-probability] + 2 * [probability/2], [0, 2, 3])

    def broadcast_arguments(self, qargs, cargs):
        for qarg, carg in zip(qargs, cargs):
            yield qarg, carg

    @property
    def label(self):
        return f"cerase_({self.params[-1]})"

def measure_flip(self, probability, *clbits):
    if probability == 0:
        return

    return self.append(MeasureFlip(probability), cargs=list(clbits))

def bit_flip(self, probability, qubit):
    if probability == 0:
        return

    return self.append(BitFlip(probability), qargs=[qubit])

def phase_flip(self, probability, qubit):
    if probability == 0:
        return

    return self.append(PhaseFlip(probability), qargs=[qubit])

def depolarize(self, probability: float, *qubits: int):
    """Apply a depolarizing channel on qubits

    Args:
        probability (float): Depolarizing probability. The first element for I is (1-probability)
        *qubits (int): A variable number of qubits to apply this channel for. See examples below.

        Examples:
            depolarize(p, 0)                # 1-qubit depol on qubit 0
            depolarize(p, 0, 1)             # 1-qubit depol on qubit 0 and 1
            depolarize(p, [0, 1])           # 2-qubit depol on qubits 0, 1
            depolarize(p, [0, 1], [2, 3])   # 2-qubit depol on qubits 0, 1 and 2, 3

    Returns:
        qiskit.circuit.InstructionSet: handle to the added instruction.
    """
    if probability == 0:
        return

    try:
        num_qubits = len(qubits[0])
    except TypeError:
        num_qubits = 1

    return self.append(Depolarize(num_qubits, probability), qargs=list(qubits))

def dephase(self, duration, T2, qubit):
    if duration == 0:
        return

    return self.append(Dephase(duration, T2), qargs=[qubit])

def relax(self, duration, T1, qubit):
    if duration == 0:
        return

    return self.append(Relax(duration, T1), qargs=[qubit])

def decohere(self, duration, T1, T2, qubit):
    if duration == 0:
        return

    return self.append(Decohere(duration, T1, T2), qargs=[qubit])

def unary_decohere(self, duration, T, qubit):
    if duration == 0:
        return

    return self.append(UnaryDecohere(duration, T), qargs=[qubit])

def diagonalize_ghz(self, probability: float, qubits):
    if probability == 0:
        return

    return self.append(DiagonalizeGHZ(len(qubits), probability), qargs=qubits)

def erase(self, probability, bit: Bit):
    """bit can be Qubit or Clbit - the correct erasure channel is applied"""
    if not isinstance(bit, Bit):
        raise ValueError(f"please provide a bit, not {bit}")

    if probability == 0:
        return

    # Locate existing erasure bit or create a new one
    for clbit in self.clbits:
        if isinstance(clbit, Erasurebit) and clbit.source == bit:
            erasure_bit = clbit.source
            break
    else:  # Executes if for loop didn't break
        erasure_bit = Erasurebit(source=bit)
        self.add_bits((erasure_bit,))

    if isinstance(bit, Qubit):
        return self.append(QuantumErase(probability), qargs=[bit], cargs=[erasure_bit])
    else:
        return self.append(ClassicalErase(probability), cargs=[erasure_bit, bit])  # Im pretty sure it should be this order

QuantumCircuit.measure_flip = measure_flip
QuantumCircuit.bit_flip = bit_flip
QuantumCircuit.phase_flip = phase_flip
QuantumCircuit.depolarize = depolarize
QuantumCircuit.dephase = dephase
QuantumCircuit.relax = relax
QuantumCircuit.decohere = decohere
QuantumCircuit.unary_decohere = unary_decohere
QuantumCircuit.diagonalize_ghz = diagonalize_ghz
QuantumCircuit.erase = erase

def test_noise(self, qubits):
    num_qubits = len(qubits)

    probs = np.arange(4**num_qubits, dtype=float)
    probs = probs/sum(probs)

    return self.append(FrameError(num_qubits, 0, [p for p in probs], label="test_noise"), qargs=qubits)

QuantumCircuit.test_noise = test_noise
