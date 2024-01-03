from collections import defaultdict
from more_itertools import pairwise
from itertools import compress, groupby

import numpy as np
import networkx as nx
import galois as gl
GF2 = gl.GF(2)

from manticore.geometry import Lattice, Crystal
from manticore.geometry.library import SquareCell, CubicCell
from manticore.geometry.linalg import fundamental_subspaces, quotient
from collections import defaultdict

import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt
from plotting import element_coordinates, plot_edge


def get_logicals(crystal):
    def boundary_maps(uc):
        elements = []
        for d in range(uc.dim + 1):
            elements.append(list(uc.filter_nodes(dim=d)))

        bounds = []
        for rows, cols in pairwise(elements):
            shape = (len(rows), len(cols))
            matrix = GF2.Zeros(shape)

            for u, v in uc.edges():
                if u in cols and v in rows:
                    matrix[rows.index(v), cols.index(u)] ^= 1

            bounds.append(matrix)

        return bounds, elements

    (d1, d2, d3), elements = boundary_maps(crystal)

    d1_spaces = fundamental_subspaces(d1)
    d2_spaces = fundamental_subspaces(d2)
    d3_spaces = fundamental_subspaces(d3)

    primal_logicals = quotient(d1_spaces["ker"], d2_spaces["img"])
    primal_checks = quotient(d2_spaces["coker"], d1_spaces["coimg"])
    dual_logicals = quotient(d3_spaces["coker"], d2_spaces["coimg"])
    dual_checks = quotient(d2_spaces["ker"], d3_spaces["img"])

    return primal_logicals, primal_checks, dual_logicals, dual_checks, elements

def print_sparse(a, crystal, labels="xyz", op=False):
    a, b = np.nonzero(a)

    for k, g in groupby(zip(a, b), key=lambda a: a[0]):
        _, ps = zip(*g)
        # elems = list(crystal.nodes[p] for p in ps)
        print(f"Logical {'op' if op else 'check'} {labels[k]}: {ps}")

# Glueeeeeeeee
def glue(crystal, logical):
    print("\nCUTTING AWAY")
    logical = logical & set(crystal.nodes) # Dirty fix if overlap with previous glues
    for node in logical:
        print(crystal.nodes[node])
    edges = list(set.union(*(set(crystal[n]) for n in logical)))
    verts = list(set.union(*(set(crystal[n]) for n in edges)))

    # print("CLOSURE:")
    # print(edges)
    # print(verts)
    crystal.remove_nodes_from(logical)
    crystal.remove_nodes_from(edges)

    base = verts[0]
    crystal.nodes[base]["boundary"] = True
    for node in verts[1:]:
        nx.contracted_nodes(crystal, base, node, self_loops=False, copy=False)


#%%
def merge_nodes(G,nodes, new_node, attr_dict=None, **attr):
    """
    Merges the selected `nodes` of the graph G into one `new_node`,
    meaning that all the edges that pointed to or from one of these
    `nodes` will point to or from the `new_node`.
    attr_dict and **attr are defined as in `G.add_node`.
    """
    
    G.add_node(new_node, attr_dict, **attr) # Add the 'merged' node
    
    for n1,n2,data in G.edges(data=True):
        # For all edges related to one of the nodes to merge,
        # make an edge going to or coming from the `new gene`.
        if n1 in nodes:
            G.add_edge(new_node,n2,data)
        elif n2 in nodes:
            G.add_edge(n1,new_node,data)
    
    for n in nodes: # remove the merged nodes
        G.remove_node(n)

def plot_crystal(crystal, unit_cell):
    from manticore.geometry.plotting import element_coordinates, plot_coloring
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.use('TkAgg')

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(projection='3d', proj_type='ortho')

    positions = element_coordinates(unit_cell)

    # Plot coloring
    cmap = plt.get_cmap("tab10")
    # Plot unit cell contours

    for edge in crystal.filter_nodes(dim=1):
        v1, v2 = (v for _, v in crystal.edges(edge))
        vdata1 = crystal.nodes[v1]
        vdata2 = crystal.nodes[v2]

        l1_pos = vdata1["pos"]
        l2_pos = vdata2["pos"]

        p1 = positions.loc[vdata1["uc"]]  # + coordinate_offsets.get(v1, 0)
        p2 = positions.loc[vdata2["uc"]] # + coordinate_offsets.get(v2, 0)

        coords = np.array([l1_pos, l2_pos])
        # jump = np.abs(coords[1] - coords[0])
        # coords[0] += (jump > 1) * (jump + 1)

        plot_coords = np.array([p1, p2]) + coords
        ax.plot(*plot_coords.T, color='k', alpha=0.1)

    ax.view_init(elev=30, azim=120)
    ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
    for _ in range(1):
        uc = CubicCell()
        sizes = (3, 3, 3)
        lat = Lattice(sizes)

        node_index = defaultdict(lambda: len(node_index))
        crystal = uc.build(lat, node_index)

        for a in crystal.filter_nodes():
            if a == 0:
                print(f"Main object is: {crystal.nodes[a]}.")
                b = [v for (_, v) in crystal.edges(a)]
                for c in b:
                    print(f"Boundary object is: {crystal.nodes[c]}.")

            # if a == 0:
            #     print(f"Main object is: {crystal.nodes[a]}.")
            #     for b in crystal.edges(a):
            #         print(f"Smaller objects are: {crystal.nodes[b]}.")


        positions = element_coordinates(uc)  # Used for plotting

        break

        primal_logicals, primal_checks, dual_logicals, dual_checks, elements = get_logicals(crystal)
        print("BEFORE")
        print_sparse(primal_logicals.T, crystal)
        # print_sparse(primal_checks.T, crystal)
        print_sparse(dual_logicals.T, crystal)
        # print_sparse(dual_checks.T, crystal)
        print(f"elements = {elements}.")

        # primal_faces_glue = set(compress(elements[2], dual_checks[:, 0]))
        # dual_faces_glue = set(compress(elements[1], primal_checks[:, 2]))
        # dual_faces_glue_y = set(compress(elements[1], primal_checks[:, 1]))

        # print("First primal logical AFTER")
        # logical = set(compress(elements[1], primal_logicals[:, 0]))

        rough_x = set(crystal.filter_nodes(uc="f_x", x=0))  # Creates a rough boundary
        smooth_y = set(crystal.filter_nodes(uc="e_y", y=0))  # Creates a smooth boundary
        smooth_z = set(crystal.filter_nodes(uc="e_z", z=0))  # Creates a smooth boundary
        rough_z = set(crystal.filter_nodes(uc="f_z", z=0))  # Creates a smooth boundary

        first_z_faces = set(crystal.filter_nodes(uc="f_z", z=0))  # Creates a rough boundary
        last_z_faces = set(crystal.filter_nodes(uc="f_z", z=2))  # Creates a rough boundary
        first_z_edges = set(crystal.filter_nodes(uc="e_z", z=0))  # Creates a rough boundary
        last_z_edges = set(crystal.filter_nodes(uc="e_z", z=2))  # Creates a rough boundary

        x_logical = set(crystal.filter_nodes(uc="f_x", x=0))  # Creates rough boundary
        y_logical = set(crystal.dual.filter_nodes(uc="e_y", y=0))  # Creates smooth boundary

        # glue(crystal, rough_x)
        # glue(crystal.dual, smooth_y)
        # glue(crystal.dual, smooth_z)
        # glue(crystal, rough_z)

        # crystal.remove_nodes_from(first_z_faces)
        # crystal.remove_nodes_from(last_z_edges)

        # primal_logicals, primal_checks, dual_logicals, dual_checks, elements = get_logicals(crystal)
        # print(elements)
        # print("\nAFTER GLUE PRIMAL LOGICALS")
        # print_sparse(primal_logicals.T, crystal, op=True)
        # print_sparse(primal_checks.T, crystal)
        # print("\nAFTER GLUE DUAL LOGICALS")
        # print_sparse(dual_logicals.T, crystal, op=True)
        # print_sparse(dual_checks.T, crystal)

        print(crystal)
        print(x_logical)
        print(len(x_logical))

        # glue(crystal, x_logical)
        # glue(crystal.dual, y_logical)

        # plot_crystal(crystal, uc)

        # Plotting


        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(projection='3d', proj_type='ortho')

        for edge in crystal.filter_nodes(dim=1):
            plot_edge(crystal, positions, ax, edge, 'gray', alpha=0.1)



        logical = set(compress(elements[1], primal_logicals[:, 0]))
        for i, node in enumerate(logical):
            print(i, crystal.nodes[node])

        for edge in logical:
            plot_edge(crystal, positions, ax, edge, 'blue', alpha=0.8)

        for node in logical:
            print(crystal.nodes[node])

        ax.set_xlim(-0.5, sizes[0]-0.5)
        ax.set_ylim(-0.5, sizes[1]-0.5)
        ax.set_zlim(-0.5, sizes[2]-0.5)

        ax.view_init(elev=30, azim=120)
        ax.set_axis_off()
        # plt.show()
