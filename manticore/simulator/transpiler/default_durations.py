from qiskit.transpiler import InstructionDurations

import warnings

class UnityDurations(InstructionDurations):

    def __init__(self, dt: float = None):
        super().__init__(dt=dt)
        self.instant_instructions = ["x", "y", "z", "frame_error", "clcx"]

    def __str__(self):
        return f"*: 1 dt (unity)"

    def update(self, inst_durations, dt):
        warnings.warn("Updating unity durations is disabled",
            DeprecationWarning, stacklevel=2)
        pass

    def get(self, inst, qubits, unit="dt"):
        if len(qubits) == 0: # TODO msmsnt flips cause IndexError in super
            return 0
        return super().get(inst, qubits, unit)

    def _get(self, name, qubits, to_unit) -> float:
        if name in self.instant_instructions:
            return 0

        return self._convert_unit(1, "dt", to_unit)

    def units_used(self):
        return {"dt"}
