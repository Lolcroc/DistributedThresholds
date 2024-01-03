# Math
import numpy as np

# Qiskit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler import TransformationPass
from qiskit.dagcircuit import DAGCircuit

from qiskit.quantum_info import Clifford
from qiskit.providers.aer.library import SetStabilizer

class ExpandInitialStabilizers(TransformationPass):
    """Expand input stabilizers with Z-stabilizers on uninitialized qubits and merge into one instruction
    
    This operation is necessary to do Clifford simulation, since Qiskit
    only allows for SetStabilizer instructions that span across all qubits
    """

    def __init__(self):
        super().__init__()

    def run(self, dag):
        """Run this pass on `dag`.

        Args:
            dag (DAGCircuit): The DAG to be transformed.

        Returns:
            DAGCircuit: The new DAG with an initial stabilizer across all qubits
        """
        # initial_stabs = set()

        # # Put all SetStabilizer instructions in the first 
        # for qubit in dag.qubits:
        #     node = dag.input_map[qubit]
            
        #     first_node = next(dag.successors(node))
        #     if isinstance(first_node.op, SetStabilizer):
        #         initial_stabs.add(first_node)

        # Stabilizer tableau across all qubits
        clifford = Clifford(np.eye(2 * dag.num_qubits(), dtype=bool))

        # Qubit to integer index in stabilizer tableau
        q2i = {q: i for i, q in enumerate(dag.qubits)}

        # True iff the DAGNode represents a SetStabilizer instruction
        def sets_stab(node):
            return isinstance(node.op, SetStabilizer)

        # Loop over all SetStabilizer nodes in the initial layer
        for stab_node in filter(sets_stab, dag.front_layer()):
            # Reconstruct stabilizer tableau from operation
            stab_cliff = Clifford.from_dict(stab_node.op.params[0])

            # Compute indices in stabilizer tableau
            x_indices = [q2i[q] for q in stab_node.qargs]
            z_indices = [q2i[q] + dag.num_qubits() for q in stab_node.qargs]
            tot_indices = x_indices + z_indices

            # Overwrite the corresponding tableau entries
            clifford.table._array[np.ix_(tot_indices, tot_indices)] = \
                stab_cliff.table._array # Submatrix idx should use np.ix_
            clifford.table._phase[tot_indices] = \
                stab_cliff.table._phase

            # Remove the current node
            dag.remove_op_node(stab_node)

        # Create new stabilizer instruction and prepend to circuit
        new_op = SetStabilizer(clifford)
        dag.apply_operation_front(new_op, dag.qubits, [])

        return dag