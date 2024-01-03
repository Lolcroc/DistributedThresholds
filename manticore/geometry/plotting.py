# Math
import numpy as np
import pandas as pd

# Manticore
from manticore.geometry import CARTESIAN_LABELS

def element_coordinates(self, base=None):
    labels = tuple(CARTESIAN_LABELS[:self.dim])
    
    vertices = list(self.filter_nodes(dim=0))
    base = base if base in vertices else vertices[0]

    # A linear system of eqs. s.t. Ax = b, where x are the positions of vertices
    A = pd.DataFrame(0, index=vertices, columns=vertices) # M x M matrix
    b = pd.DataFrame(0, index=vertices, columns=labels) # M x dim matrix (rows are xyz coordinates)
    positions = pd.DataFrame(0, index=self.nodes, columns=labels) # N x dim matrix, where N = len(self.nodes)

    for u in vertices:
        # Skip calculating the base (since it is located at the origin)
        if u == base:
            continue

        # Do a ping-pong from all vertices to elements and back
        for _, e, o1 in self.dual.edges(u, keys=True):
            for _, v, o2 in self.edges(e, keys=True):
                same = u == v and o1 == o2 # True iff the ping-pong returns to sender
                diff = tuple(j - i for i, j in zip(o1, o2))

                # +1 is added for each incoming boundary
                # -1 is added for each distinct neighbour
                A.loc[u, v] += 2*same - 1
                b.loc[u] += diff

    # Set the equation: base = [0, 0]
    A.loc[base, base] = 1
    A_inv = pd.DataFrame(np.linalg.inv(A), index=A.columns, columns=A.index)
    positions.loc[vertices] = A_inv @ b # Solve positions of vertices

    # Every higher dimensional element is placed at the average of its boundary
    for dim in range(1, self.dim + 1):
        for e in self.filter_nodes(dim=dim):
            neighbours = self.edges(e, keys=True)
            N = len(neighbours)

            for _, u, offset in neighbours:
                positions.loc[e] += (positions.loc[u] + offset) / N

    return positions

def plot_coloring(unit_cell, ax, cmap, total_offset=0, positions=None, **coordinate_offsets):
    # unit_cell.verify_coloring()
    if positions is None:
        positions = element_coordinates(unit_cell)

    plot_edges = set()
    for _, edge, offset in unit_cell.filter_edges_from(dim=2, keys=True):
        plot_edges.add((edge, offset))

    # transform_test = np.array([
    #     [1/np.sqrt(2),  0,              1/np.sqrt(3)],
    #     [0,             1/np.sqrt(2),   1/np.sqrt(3)],
    #     [-1/np.sqrt(2), -1/np.sqrt(2),  1/np.sqrt(3)]
    # ], dtype=float) 
    # transform_test = np.linalg.inv(transform_test)

    # trans2 = np.array([[1, 0, 0], [1/np.sqrt(5), 2/np.sqrt(5), 0], [0, 0, 3]], dtype=float)
    # transform_test = trans2 @ transform_test

    # transform_test = np.eye(3) + lattice.skew_matrix
    # transform_test = np.eye(3) + lattice.skew_matrix.T / lattice.reps
    # transform_test = transform_test / np.linalg.norm(transform_test, axis=0)
    # transform_test = transform_test / np.linalg.det(transform_test)
    # print(transform_test)

    # Plot the lattice edges with low alpha
    for edge, offset in plot_edges:
        (v1, o1), (v2, o2) = ((v, o) for _, v, o in unit_cell.edges(edge, keys=True))
        p1 = positions.loc[v1] + o1 + offset + coordinate_offsets.get(v1, 0) + total_offset
        p2 = positions.loc[v2] + o2 + offset + coordinate_offsets.get(v2, 0) + total_offset

        coords = np.array([p1, p2]).T
        # alpha = 0.2 if any(o != 0 for o in offset) else 0.5
        ax.plot(*coords, color='k', alpha=0.2)

    # Plot the lattice edges with low alpha
    # for edge in unit_cell.filter_nodes(dim=1):
    #     (v1, o1), (v2, o2) = ((v, offset) for _, v, offset in unit_cell.edges(edge, keys=True))
    #     p1 = positions.loc[v1] + o1 + coordinate_offsets.get(v1, 0) + total_offset
    #     p2 = positions.loc[v2] + o2 + coordinate_offsets.get(v2, 0) + total_offset

    #     coords = np.array([p1, p2]).T

    #     ax.plot(*coords, color='k', alpha=0.2)

    # Plot the colorings from faces to edges
    for arc in unit_cell.filter_edges_from(dim=2, keys=True):
        face, edge, offset = arc

        p1 = positions.loc[face] + coordinate_offsets.get(face, 0) + total_offset
        p2 = positions.loc[edge] + offset + coordinate_offsets.get(edge, 0) + total_offset

        color = unit_cell.edges[arc].get("color", -1)
        coords = np.array([p1, p2]).T
        ax.plot(*coords, color=cmap(color), alpha=1)

# trans = np.array([[1, 0], [-1/np.sqrt(5), 2/np.sqrt(5)]], dtype=float)

def plot_edge(crystal, positions, ax, edge, color, offset=0, alpha=1):
    edata = crystal.nodes[edge]
    edge_pos = positions.loc[edata["uc"]] + edata["pos"]

    for vertex in crystal[edge]:
        if crystal.nodes[vertex].get("boundary", False):
            continue
        vdata = crystal.nodes[vertex]
        vertex_pos = positions.loc[vdata["uc"]] + vdata["pos"]

        coords = np.array([edge_pos, vertex_pos])
        jump = np.abs(coords[1] - coords[0])
        coords[0] -= (jump > 0.999) * (jump + 0.5)  # TODO Dirty only works for cubic

        # coords = trans @ coords.T
        coords = coords.T

        ax.plot(*coords, color=color, alpha=alpha)
