# Python
from collections import defaultdict
from itertools import product
from more_itertools import unique_everseen

# Math
import numpy as np

# Qiskit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.passes import ALAPSchedule, ASAPSchedule
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit, Instruction, Gate, Measure, Parameter, Qubit, Clbit

# Custom
from superop.errors.standard_errors import FrameError, Depolarize, MeasureFlip, Decohere
from .default_durations import UnityDurations

class AddDecoherence(TransformationPass):
    
    def __init__(self, durations=None, T1=10):
        super().__init__()
        self.T1 = T1

        if durations is None:
            durations = UnityDurations()
        self.requires.append(ALAPSchedule(durations))

    def run(self, dag):
        new_dag = dag._copy_circuit_metadata()

        for node in dag.topological_op_nodes():
            op = node.op
            new_dag.apply_operation_back(op, node.qargs, node.cargs)

            if op.duration > 0:
                for qarg in node.qargs:
                    noise = Decohere(op.duration, self.T1)
                    new_dag.apply_operation_back(noise, [qarg])

        return new_dag

class InsertPauliNoise(TransformationPass):

    def __init__(self):
        super().__init__()

    def run(self, dag):
        new_dag = dag._copy_circuit_metadata()

        # prob = Parameter('p')
        prob = 0.1

        for node in dag.topological_op_nodes():
            op = node.op
            new_dag.apply_operation_back(op, node.qargs, node.cargs)

            noise = None
            if isinstance(op, Gate) and op.name not in ('x', 'y', 'z'):
                noise = Depolarize(op.num_qubits, prob)
            elif isinstance(op, Measure):
                noise = MeasureFlip(prob)

            if noise:
                new_dag.apply_operation_back(noise, node.qargs, node.cargs)

        return new_dag

class SqueezePauliNoise(TransformationPass):

    def __init__(self):
        super().__init__()

    def run(self, dag):
        reversed_nodes = reversed(list(dag.topological_op_nodes()))

        q2i = {q: i for i, q in enumerate(dag.qubits)}
        c2i = {c: i for i, c in enumerate(dag.clbits)}

        num_qubits, num_clbits = dag.num_qubits(), dag.num_clbits()

        knill_op = KnillOperator(QuantumCircuit(num_qubits, num_clbits)) # Temp

        errors = []
        for node in reversed_nodes:
            op, qargs, cargs = node.op, node.qargs, node.cargs
            condition_bits = dag._bits_in_condition(op.condition)
            cargs = list(dict.fromkeys(cargs + condition_bits))

            x_qubits = [q2i[q] for q in qargs]
            z_qubits = [q2i[q] + num_qubits for q in qargs]
            clbits = [c2i[c] + 2 * num_qubits for c in cargs]
            cols = x_qubits + z_qubits + clbits

            if isinstance(op, FrameError):
                # Propagate error through knill
                op.table = xor_dot(knill_op.table[:, cols], op.table)
                op.num_qubits = knill_op.num_qubits
                op.num_clbits = knill_op.num_clbits

                dag.remove_op_node(node)
                errors.append(op)
            else:
                # Update Knill operator
                table = KnillOperator(op).table
                knill_op.table[:, cols] = xor_dot(knill_op.table[:, cols], table)

        errors.sort(key=lambda e: len(e.params), reverse=True)

        err_len = 2 * num_qubits + num_clbits

        tot_table = errors[0].table
        tot_probs = errors[0].probabilities
        for err in errors[1:]:
            # Multiply in the new error
            new_probs = (a * b for a, b in product(tot_probs, err.probabilities))
            combined_error = np.reshape(tot_table[:, :, None] ^ err.table[:, None, :], (err_len, -1))

            # Convert error strings to binary (int)
            mask = np.arange(err_len)
            as_ints = np.sum(combined_error.T << mask, axis=1)
            
            # Aggregate errors with same error strings
            d = defaultdict(lambda: 0)
            for i, val in zip(as_ints, new_probs):
                d[i] += val

            as_ints = np.fromiter(d.keys(), dtype=int)

            # Set new errors and probabilities
            tot_table = (as_ints[None, :] >> mask[:, None]) % 2 == 1
            tot_probs = list(d.values())

        tot_error = FrameError(num_qubits, num_clbits, tot_table, tot_probs)
        dag.apply_operation_back(tot_error, dag.qubits, dag.clbits)

        return dag

def xor_dot(a, b):
    out = np.dot(a.astype(int), b.astype(int)) % 2
    return out.astype(bool)