# Python
from itertools import groupby

# Math
import numpy as np
from retworkx import collect_runs
from sympy.logic.boolalg import anf_coeffs

# Qiskit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler import TransformationPass
from qiskit.circuit import Instruction

class PairConditionalPaulis(TransformationPass):
    """Merge consecutive Pauli operators conditioned on measurement outcomes into one.

    Merging is only supported for Pauli operators that are conditioned on the parity
    of one or more outcomes. For example, two X operators conditioned on 0=0b00 and 
    3=0b11 will be merged into one X operator conditioned on the parity of both
    outcomes being zero.
    """

    def __init__(self):
        super().__init__()

    def run(self, dag):
        """Run this pass on `dag`.

        Args:
            dag (DAGCircuit): The DAG to be transformed.

        Raises:
            TranspilerError: If trying to merge consecutive Pauli operators whose
            conditional outcomes cannot be expressed as a parity

        Returns:
            DAGCircuit: The new DAG with merged Paulis
        """
        # Collect consecutive Pauli gates ("runs")
        paulis = collect_runs(dag._multi_graph, _is_cond_pauli)
        # Split each run by its subtype (X, Y, or Z)
        paulis = _split_by_type(paulis)
        
        for run in paulis:
            first_op = run[0].op
            reg_size = first_op.condition[0].size

            # Fill a truth table; the bool at index i is True iff the Pauli is
            # conditioned on this value of the classical register
            truth_table = np.zeros(2**reg_size, dtype=bool)
            for node in run:
                instr = node.op
                reg, val = instr.condition

                truth_table[val] = True

            # Convert to truth table to Algebraic Normal Form:
            # https://en.wikipedia.org/wiki/Algebraic_normal_form#Formal_representation
            anf_table = anf_coeffs(truth_table)

            # Single-variate indices of the ANF form
            single_inds = 2**np.arange(reg_size)

            # Populate coefficients of the parity string from ANF form
            # These are exactly the single-variate coefficients
            parity_coefficients = []
            for i, coeff in enumerate(anf_table):
                # If there is single-variate condition, save its parity coefficient
                if i in single_inds:
                    parity_coefficients.append(bool(coeff)) # Bool casting to prevent error in parameter check
                # If there is multi-variate condition, throw an error
                elif i > 0 and coeff:
                    vals = ", ".join(f"{node.op.condition[1]}" for node in run)
                    raise TranspilerError(f"Pauli {first_op.name} conditioned on {first_op.condition[0]}", \
                        f"with values ({vals}) cannot be written as a parity")

            # Create new instruction
            # The value for the condition is either even (0) or odd (1) parity
            new_gate = first_op.copy()
            new_gate = new_gate.c_if(first_op.condition[0], 1-anf_table[0])
            new_gate.params = parity_coefficients

            # Substitute in DAG
            dag.substitute_node(run[0], new_gate, inplace=True)

            # Remove the rest of the Paulis
            for node in run[1:]:
                dag.remove_op_node(node)
        
        return dag

def _split_by_type(runs):
    """Split a consecutive run of Pauli operators into chunks based on their type

    For example, a run of [XGate, XGate, YGate, YGate] will be split into
    [XGate, XGate], [YGate, YGate] runs.

    Args:
        runs (list[list[DAGNode]]): A nested list of consecutive Pauli operators
        as DAGNode objects

    Returns:
        list[list[DAGNode]]: Each run split into one type only
    """
    out = []
    for run in runs:
        groups = groupby(run, lambda x: x.op.name)

        for _, gates in groups:
            out.append(list(gates))

    return out

def _is_cond_pauli(node):
    """Test if input `node` is a conditional Pauli

    Args:
        node (DAGNode): The input to test

    Returns:
        bool: True iff the input `node` is a conditional Pauli
    """
    return node.type == "op" and node.name in ('x', 'y', 'z') and node.op.condition is not None