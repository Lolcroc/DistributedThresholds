# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from collections import defaultdict
from typing import List
from qiskit.transpiler.passes.scheduling.time_unit_conversion import TimeUnitConversion
from qiskit.circuit.delay import Delay
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

"""
See https://github.com/Qiskit/qiskit-terra/issues/7006

Qiskit ALAP/ASAP schedulers do not take into account topological order of
classical bit lines. This means *any* instruction that depends on classical
registers is treated as if it doesn't interact with classical bits. Then it
is possible for conditional gates to be scheduled *before* the measurements
that produce the outcome they are conditioned on. This is clearly a bug.

Funnily enough, this is actually what we want for Pauli operators . Since our
circuits are Clifford, the conditional Paulis may be tracked offline (in some
classical software). This means these gates and their classical registry
dependencies may be completely ignored by the ALAP/ASAP schedulers. In an
actual experiment, the conditional Paulis are then post-processed by (a)
applying them based on measurement results, (b) pulling Paulis through all
Cliffords until hitting another measurement, and (c) flipping those outcomes
based on the Pauli in front.

Relying on this bug is obviously not future proof. The correct strategy
would then be to run the ALAP/ASAP scheduler in such a way as to completely
ignore *only* Pauli operators with classical dependencies. Any non-Pauli
Instruction cannot be ignored and needs to be treated as a dependency.
Setting a duration to zero is not the same as ignoring it! The dependency
will still be handled in order to prevent other gates from sliding through
the zero-duration gate.
"""

def patched_alap_run(self, dag):
    """A patch for ALAPSchedule.run that takes into account classical bits
    """
    # Why this is here? I don't know
    # if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
    #     raise TranspilerError("ALAP schedule runs on physical circuits only")

    time_unit = self.property_set["time_unit"]
    new_dag = DAGCircuit()
    for qreg in dag.qregs.values():
        new_dag.add_qreg(qreg)
    for creg in dag.cregs.values():
        new_dag.add_creg(creg)

    qubit_time_available = defaultdict(int)

    def pad_with_delays(qubits: List[int], until, unit) -> None:
        """Pad idle time-slots in ``qubits`` with delays in ``unit`` until ``until``."""
        for q in qubits:
            if qubit_time_available[q] < until:
                idle_duration = until - qubit_time_available[q]
                new_dag.apply_operation_front(Delay(idle_duration, unit), [q], [])

    bit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
    for node in reversed(list(dag.topological_op_nodes())):
        """PATCH HERE IN FOR LOOP"""
        start_time = max(qubit_time_available[q] for _, _, q in dag.edges(node))
        pad_with_delays(node.qargs, until=start_time, unit=time_unit)

        new_dag.apply_operation_front(node.op, node.qargs, node.cargs)

        # validate node.op.duration
        if node.op.duration is None:
            indices = [bit_indices[qarg] for qarg in node.qargs]
            raise TranspilerError(
                f"Duration of {node.op.name} on qubits " f"{indices} is not found."
            )
        if isinstance(node.op.duration, ParameterExpression):
            indices = [bit_indices[qarg] for qarg in node.qargs]
            raise TranspilerError(
                f"Parameterized duration ({node.op.duration}) "
                f"of {node.op.name} on qubits {indices} is not bounded."
            )

        stop_time = start_time + node.op.duration
        # update time table
        """PATCH HERE IN FOR LOOP"""
        for _, _, q in dag.edges(node):
            qubit_time_available[q] = stop_time

    working_qubits = qubit_time_available.keys()
    circuit_duration = max(qubit_time_available[q] for q in working_qubits)
    pad_with_delays(new_dag.qubits, until=circuit_duration, unit=time_unit)

    new_dag.name = dag.name
    new_dag.metadata = dag.metadata
    # set circuit duration and unit to indicate it is scheduled
    new_dag.duration = circuit_duration
    new_dag.unit = time_unit
    return new_dag

# from qiskit.transpiler.passes import ALAPSchedule
# ALAPSchedule.run = patched_alap_run