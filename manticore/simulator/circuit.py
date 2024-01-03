# Python
from itertools import count

# Math
import numpy as np
import pandas as pd

# Qiskit
from qiskit.circuit import QuantumCircuit, Instruction, Qubit, Clbit, AncillaQubit, QuantumRegister, ClassicalRegister
from qiskit.circuit.bit import Bit
from qiskit.circuit.register import Register
from qiskit.providers.aer.library import SetStabilizer

def patched__repr__(self):
    """Return the official string representing the bit."""
    if (self._register, self._index) == (None, None):
        # Similar to __hash__, use default repr method for new-style Bits.
        return f"{self.__class__.__name__}_{id(self)}"
    return f"{self._register.name}_{self._index}"

Bit.__repr__ = patched__repr__

class LabelledQubit(Qubit):
    __slots__ = ("xlabel","zlabel")

    def __init__(self, register=None, index=None, xlabel=None, zlabel=None):
        super().__init__(register, index)
        self.xlabel = xlabel or f"{self}_x"
        self.zlabel = zlabel or f"{self}_z"

class LabelledClbit(Clbit):
    __slots__ = ("label",)

    def __init__(self, register=None, index=None, label=None):
        super().__init__(register, index)
        self.label = label or f"{self}"

class InQubit(LabelledQubit):
    __slots__ = ()

class OutQubit(LabelledQubit):
    __slots__ = ()

class IOQubit(InQubit, OutQubit):
    __slots__ = ()

class InClbit(LabelledClbit):
    __slots__ = ()

class OutClbit(LabelledClbit):
    __slots__ = ()

class IOClbit(InClbit, OutClbit):
    __slots__ = ()

class Erasurebit(OutClbit):
    __slots__ = ("source",)

    def __init__(self, source: Bit, register=None, index=None, label=None):
        super().__init__(register, index, label)
        self.source = source

class LabelledRegister(Register):
    __slots__ = ()

    def __init__(self, size=None, name=None, bits=None):
        super().__init__(size, name, bits)

        # for i, bit in enumerate(self):
        #     if isinstance(bit, LabelledQubit):
        #         bit.xlabel = f"{self.name}_{i}_x"
        #         bit.zlabel = f"{self.name}_{i}_z"
        #     elif isinstance(bit, LabelledClbit):
        #         bit.label = f"{self.name}_{i}"

class QuantumInRegister(LabelledRegister, QuantumRegister):
    __slots__ = ()

    instances_counter = count()
    prefix = "qi"
    bit_type = InQubit

class QuantumOutRegister(LabelledRegister, QuantumRegister):
    __slots__ = ()

    instances_counter = count()
    prefix = "qo"
    bit_type = OutQubit

class QuantumIORegister(LabelledRegister, QuantumRegister):
    __slots__ = ()

    instances_counter = count()
    prefix = "qio"
    bit_type = IOQubit

class ClassicalInRegister(LabelledRegister, ClassicalRegister):
    __slots__ = ()

    instances_counter = count()
    prefix = "ci"
    bit_type = InClbit

class ClassicalOutRegister(LabelledRegister, ClassicalRegister):
    __slots__ = ()

    instances_counter = count()
    prefix = "co"
    bit_type = OutClbit

class ClassicalIORegister(LabelledRegister, ClassicalRegister):
    __slots__ = ()

    instances_counter = count()
    prefix = "cio"
    bit_type = IOClbit

class CreateBell(SetStabilizer):

    def __init__(self):
        qc = QuantumCircuit(2, name='bell')
        qc.h(0)
        qc.cx(0, 1)
        super().__init__(qc)

class ClassicalCX(Instruction):

    def __init__(self, label=None):
        super().__init__("clcx", 0, 2, [], label=label)

def clcx(self, control_bit, target_bit):
    self.append(ClassicalCX(), cargs=[control_bit, target_bit])

def fake_clcx(self, control_bit: Clbit, target_bit: Clbit):
    """Qiskit Simulator-friendly version of a classical CNOT

    Does the following:
        1.  Creates a new qubit in |0>
        2.  Does a conditional X-gate on this qubit for both classical bits. The qubit now has
            a state |m0 ⊕ m1>.
        3.  Measures this qubit back to the classical bit m1 -> m0 ⊕ m1.

    Args:
        control_bit (Clbit): Control of the classical CNOT operation (m0).
        target_bit (Clbit): Target of the classical CNOT operation (m1).
    """
    fake_qreg = self._create_qreg(1, "fake")
    self.add_register(fake_qreg)

    clbits = [control_bit, target_bit]
    clbits = self.cbit_argument_conversion(clbits)

    for cbit in clbits:
        self.x(fake_qreg).c_if(cbit, True)

    self.measure(fake_qreg, clbits[-1])

def create_bell(self, qubit0, qubit1, p_i=0):
    self.append(CreateBell(), [qubit0, qubit1])
    self.diagonalize_ghz(p_i, [qubit0, qubit1])

QuantumCircuit.clcx = clcx
QuantumCircuit.fake_clcx = fake_clcx
QuantumCircuit.create_bell = create_bell
