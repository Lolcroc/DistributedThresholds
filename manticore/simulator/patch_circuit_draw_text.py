def _node_to_gate(self, node, layer):
    """Convert a dag op node into its corresponding Gate object, and establish
    any connections it introduces between qubits"""
    op = node.op
    current_cons = []
    connection_label = None
    conditional = False
    base_gate = getattr(op, "base_gate", None)

    params = get_param_str(op, "text", ndigits=5)
    if not isinstance(op, (Measure, SwapGate, Reset)) and not op._directive:
        gate_text, ctrl_text, _ = get_gate_ctrl_text(op, "text")
        gate_text = TextDrawing.special_label(op) or gate_text
        gate_text = gate_text  # Patch: do not draw + params

    if op.condition is not None:
        # conditional
        layer.set_cl_multibox(op.condition, top_connect="╨")
        conditional = True

    # add in a gate that operates over multiple qubits
    def add_connected_gate(node, gates, layer, current_cons):
        for i, gate in enumerate(gates):
            actual_index = self.qubits.index(node.qargs[i])
            if actual_index not in [i for i, j in current_cons]:
                layer.set_qubit(node.qargs[i], gate)
                current_cons.append((actual_index, gate))

    if isinstance(op, Measure):
        gate = MeasureFrom()
        layer.set_qubit(node.qargs[0], gate)
        if self.cregbundle and self.bit_locations[node.cargs[0]]["register"] is not None:
            layer.set_clbit(
                node.cargs[0],
                MeasureTo(str(self.bit_locations[node.cargs[0]]["index"])),
            )
        else:
            layer.set_clbit(node.cargs[0], MeasureTo())

    elif op._directive:
        # barrier
        if not self.plotbarriers:
            return layer, current_cons, connection_label

        for qubit in node.qargs:
            if qubit in self.qubits:
                layer.set_qubit(qubit, Barrier())

    elif isinstance(op, SwapGate):
        # swap
        gates = [Ex(conditional=conditional) for _ in range(len(node.qargs))]
        add_connected_gate(node, gates, layer, current_cons)

    elif isinstance(op, Reset):
        # reset
        layer.set_qubit(node.qargs[0], ResetDisplay(conditional=conditional))

    elif isinstance(op, RZZGate):
        # rzz
        connection_label = "ZZ%s" % params
        gates = [Bullet(conditional=conditional), Bullet(conditional=conditional)]
        add_connected_gate(node, gates, layer, current_cons)

    elif len(node.qargs) == 1 and not node.cargs:
        # unitary gate
        layer.set_qubit(node.qargs[0], BoxOnQuWire(gate_text, conditional=conditional))

    elif isinstance(op, ControlledGate):
        params_array = TextDrawing.controlled_wires(node, layer)
        controlled_top, controlled_bot, controlled_edge, rest = params_array
        gates = self._set_ctrl_state(node, conditional, ctrl_text, bool(controlled_bot))
        if base_gate.name == "z":
            # cz
            gates.append(Bullet(conditional=conditional))
        elif base_gate.name in ["u1", "p"]:
            # cu1
            connection_label = f"{base_gate.name.upper()}{params}"
            gates.append(Bullet(conditional=conditional))
        elif base_gate.name == "swap":
            # cswap
            gates += [Ex(conditional=conditional), Ex(conditional=conditional)]
            add_connected_gate(node, gates, layer, current_cons)
        elif base_gate.name == "rzz":
            # crzz
            connection_label = "ZZ%s" % params
            gates += [Bullet(conditional=conditional), Bullet(conditional=conditional)]
        elif len(rest) > 1:
            top_connect = "┴" if controlled_top else None
            bot_connect = "┬" if controlled_bot else None
            indexes = layer.set_qu_multibox(
                rest,
                gate_text,
                conditional=conditional,
                controlled_edge=controlled_edge,
                top_connect=top_connect,
                bot_connect=bot_connect,
            )
            for index in range(min(indexes), max(indexes) + 1):
                # Dummy element to connect the multibox with the bullets
                current_cons.append((index, DrawElement("")))
        else:
            gates.append(BoxOnQuWire(gate_text, conditional=conditional))
        add_connected_gate(node, gates, layer, current_cons)

    elif len(node.qargs) >= 2 and not node.cargs:
        layer.set_qu_multibox(node.qargs, gate_text, conditional=conditional)

    elif node.qargs and node.cargs:
        if self.cregbundle and node.cargs:
            raise TextDrawerCregBundle("TODO")
        layer._set_multibox(
            gate_text,
            qubits=node.qargs,
            clbits=node.cargs,
            conditional=conditional,
        )
    else:
        raise VisualizationError(
            "Text visualizer does not know how to handle this node: ", op.name
        )

    # sort into the order they were declared in, to ensure that connected boxes have
    # lines in the right direction
    current_cons.sort(key=lambda tup: tup[0])
    current_cons = [g for q, g in current_cons]

    return layer, current_cons, connection_label

from qiskit.visualization.text import *
from qiskit.visualization.text import TextDrawing
TextDrawing._node_to_gate = _node_to_gate