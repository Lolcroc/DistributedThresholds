"""Prebuilt unit cells."""

# Python
from itertools import permutations

# Custom
from manticore.geometry import UnitCell

class CubicCell(UnitCell):
    """A 3D cubic unit cell.

    Contains 1 cube with 6 surrounding half-faces.
    """
    def __init__(self, **attr):
        super().__init__("cubic", 3, **attr)

        cube, vertex = "q", "v"
        self.add_elem(cube, 3)
        self.add_elem(vertex, 0)

        units = dict(x=(1,0,0), y=(0,1,0), z=(0,0,1))

        for i, u in units.items():
            face, edge = "f_" + i, "e_" + i

            self.add_elem(face, 2)
            self.add_elem(edge, 1)

            self.add_bound(cube, face, offset=u)
            self.add_bound(edge, vertex, offset=u)

        for i, j, k in permutations(units):
            self.add_bound("f_" + i, "e_" + j, offset=units[k])

        # Adds intra-cell boundaries - needs list() call to prevent changes during loop
        for (u, v) in list(self.edges()):
            self.add_bound(u, v)

    def color(self, *args, **kwargs):
        indices = {}
        for i, label in enumerate("xyz"):
            indices["f_" + label] = indices["e_" + label] = i

        for arc in self.filter_edges_from(dim=2, keys=True):
            face, edge, offset = arc
            self.edges[arc]["color"] = 2 * sum(offset) - 1 + (indices[face] - indices[edge]) % 3

class SquareCell(UnitCell):
    """A 2D square unit cell. 

    Contains 1 square with surrounding 4 half-edges.
    """
    def __init__(self, **attr):
        super().__init__("square", 2, **attr)

        face, vertex = "f", "v"
        self.add_elem(face, 2)
        self.add_elem(vertex, 0)

        units = dict(x=(1,0), y=(0,1))

        for i in units:
            self.add_elem("e_" + i, 1)

        for i, j in permutations(units):
            edge = "e_" + i

            self.add_bound(face, edge, offset=units[j])
            self.add_bound(edge, vertex, offset=units[i])

        # Adds intra-cell boundaries - needs list() call to prevent changes during loop
        for (u, v) in list(self.edges()):
            self.add_bound(u, v)

class DiamondCell(CubicCell):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.name = "diamond"

        self.dual.simple_split("v", "v_1", "e_1", 
            ("e_x", (0, 0, 0)), ("e_y", (0, 0, 0)), ("e_z", (0, 0, 0)))
        self.simple_split("q", "q_1", "f_1", 
            ("f_x", (0, 0, 0)), ("f_y", (0, 0, 0)), ("f_z", (0, 0, 0)))

    # def color(self, *args, **kwargs):
    #     units = dict(x=(1,0,0), y=(0,1,0), z=(0,0,1))
    #     indices = {label: i for i, label in enumerate("xyz")}

    #     for a, b, c in permutations(units):
    #         i = indices[a]

    #         self.edges[(f"f_{a}", f"e_{b}", units[c])]["color"] = (2 * (i + indices[b])) % 6
    #         self.edges[(f"f_{a}", f"e_{b}", (0, 0, 0))]["color"] = (2 * (i + indices[b]) + 1) % 6

    #         if (indices[c] - indices[b]) % 3 != 1:
    #             continue

    #         index_f_1 = [0, 2, 1]
    #         self.edges[("f_1", f"e_{a}", units[b])]["color"] = 2 * index_f_1[i]
    #         self.edges[("f_1", f"e_{a}", units[c])]["color"] = 2 * index_f_1[i] + 1
    #         self.edges[(f"f_{a}", f"e_{1}", units[b])]["color"] = (4 * i) % 6
    #         self.edges[(f"f_{a}", f"e_{1}", units[c])]["color"] = (4 * i + 1) % 6

    def color(self, *args, **kwargs):
        zero_offset = (0, 0, 0)
        x_offset = (1, 0, 0)
        y_offset = (0, 1, 0)
        z_offset = (0, 0, 1)

        self.edges[("f_x", "e_1", y_offset)]["color"] = 0
        self.edges[("f_x", "e_y", zero_offset)]["color"] = 1
        self.edges[("f_x", "e_z", zero_offset)]["color"] = 2
        self.edges[("f_x", "e_1", z_offset)]["color"] = 3
        self.edges[("f_x", "e_y", z_offset)]["color"] = 4
        self.edges[("f_x", "e_z", y_offset)]["color"] = 5

        self.edges[("f_y", "e_z", x_offset)]["color"] = 0
        self.edges[("f_y", "e_1", x_offset)]["color"] = 1
        self.edges[("f_y", "e_x", zero_offset)]["color"] = 2
        self.edges[("f_y", "e_z", zero_offset)]["color"] = 3
        self.edges[("f_y", "e_1", z_offset)]["color"] = 4
        self.edges[("f_y", "e_x", z_offset)]["color"] = 5

        self.edges[("f_z", "e_y", x_offset)]["color"] = 0
        self.edges[("f_z", "e_x", y_offset)]["color"] = 1
        self.edges[("f_z", "e_1", y_offset)]["color"] = 2
        self.edges[("f_z", "e_y", zero_offset)]["color"] = 3
        self.edges[("f_z", "e_x", zero_offset)]["color"] = 4
        self.edges[("f_z", "e_1", x_offset)]["color"] = 5

        self.edges[("f_1", "e_x", y_offset)]["color"] = 0
        self.edges[("f_1", "e_z", y_offset)]["color"] = 1
        self.edges[("f_1", "e_y", z_offset)]["color"] = 2
        self.edges[("f_1", "e_x", z_offset)]["color"] = 3
        self.edges[("f_1", "e_z", x_offset)]["color"] = 4
        self.edges[("f_1", "e_y", x_offset)]["color"] = 5

class TriamondCell(CubicCell):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.name = "triamond"

        s1 = [("e_x", (0, 0, 0)), ("e_y", (0, 0, 0))]
        s2 = [("e_z", (0, 0, 1)), ("e_x", (1, 0, 0))]
        s3 = [("e_y", (0, 1, 0)), ("e_z", (0, 0, 0))]

        r1 = [("f_x", (0, 0, 0)), ("f_y", (0, 0, 0))]
        r2 = [("f_z", (0, 0, 1)), ("f_x", (1, 0, 0))]
        r3 = [("f_y", (0, 1, 0)), ("f_z", (0, 0, 0))]

        self.dual.n_split("v", ("v_xy", "e_xy", s1), ("v_xz", "e_xz", s2), ("v_yz", "e_yz", s3))
        self.n_split("q", ("q_xy", "f_xy", r1), ("q_xz", "f_xz", r2), ("q_yz", "f_yz", r3))

    def color(self, *args, **kwargs):
        labels = ['x', 'y', 'z', 'xy', 'yz', 'xz']
        for i, r in enumerate(labels):
            if r == 'yz':
                labels_used = ['z', 'x', 'xy', 'y', 'xz']
            elif r == 'xz':
                labels_used = ['xy', 'y', 'yz', 'z', 'x']
            elif r == "xy":
                labels_used = ['x', 'y', 'z', 'yz', 'xz']
            else:
                labels_used = labels.copy()
                labels_used.remove(r)
            
            for i_label, label in enumerate(labels_used):
                offsets = [edge[2] for edge in self.edges(f'f_{r}', keys=True) if edge[1] == f"e_{label}"]
                self.edges[(f"f_{r}", f"e_{label}", offsets[0])]["color"] = (i + i_label) % 10
                self.edges[(f"f_{r}", f"e_{label}", offsets[1])]["color"] = (i + i_label + 5) % 10

    # def color(self, *args, **kwargs):
    #     units = dict(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))
    #     zero_offset = (0, 0, 0)

    #     for i, j, k in ("xyz", "zxy", "yzx"):
    #         self.edges[(f"f_{i}", f"e_{")]

class DoubleEdgeCubicCell(CubicCell):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.name = "double_edge_cubic"

        s1 = [("e_x", (0, 0, 0))]
        s2 = [("e_y", (0, 0, 0))]
        s3 = [("e_z", (0, 0, 0))]

        r1 = [("f_x", (0, 0, 0))]
        r2 = [("f_y", (0, 0, 0))]
        r3 = [("f_z", (0, 0, 0))]

        self.dual.n_split("v", ("v_x", "e_xx", s1), ("v_y", "e_yy", s2), ("v_z", "e_zz", s3))
        self.n_split("q", ("q_x", "f_xx", r1), ("q_y", "f_yy", r2), ("q_z", "f_zz", r3))

    # def color(self, *args, **kwargs):
    #     indices, offsets = {}, {}
    #     for i, label in enumerate("xyz"):
    #         indices[f"f_{label}"] = indices[f"e_{label}"] = indices[f"f_{label}{label}"] = indices[f"e_{label}{label}"] = i
    #         offsets[f"f_{label}"] = offsets[f"e_{label}"] = 0
    #         offsets[f"f_{label}{label}"] = offsets[f"e_{label}{label}"] = 4

    #     for arc in self.filter_edges_from(dim=2, keys=True):
    #         face, edge, offset = arc
    #         self.edges[arc]["color"] = abs(offsets[face] - offsets[edge]) + 2 * sum(offset) - 1 + (indices[face] - indices[edge]) % 3

    def color(self, *args, **kwargs):
        units = dict(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))
        zero_offset = (0, 0, 0)

        for i, j, k in ("xyz", "zxy", "yzx"):
            self.edges[(f"f_{i}", f"e_{k}", zero_offset)]["color"] = 0
            self.edges[(f"f_{i}", f"e_{k}{k}", zero_offset)]["color"] = 1
            self.edges[(f"f_{i}", f"e_{j}{j}", zero_offset)]["color"] = 2
            self.edges[(f"f_{i}", f"e_{j}", zero_offset)]["color"] = 3
            self.edges[(f"f_{i}", f"e_{k}{k}", units[j])]["color"] = 4
            self.edges[(f"f_{i}", f"e_{k}", units[j])]["color"] = 5
            self.edges[(f"f_{i}", f"e_{j}", units[k])]["color"] = 6
            self.edges[(f"f_{i}", f"e_{j}{j}", units[k])]["color"] = 7

            self.edges[(f"f_{i}{i}", f"e_{k}", zero_offset)]["color"] = 4
            self.edges[(f"f_{i}{i}", f"e_{k}{k}", zero_offset)]["color"] = 5
            self.edges[(f"f_{i}{i}", f"e_{j}{j}", zero_offset)]["color"] = 6
            self.edges[(f"f_{i}{i}", f"e_{j}", zero_offset)]["color"] = 7
            self.edges[(f"f_{i}{i}", f"e_{k}{k}", units[j])]["color"] = 0
            self.edges[(f"f_{i}{i}", f"e_{k}", units[j])]["color"] = 1
            self.edges[(f"f_{i}{i}", f"e_{j}", units[k])]["color"] = 2
            self.edges[(f"f_{i}{i}", f"e_{j}{j}", units[k])]["color"] = 3

class HexCell(UnitCell):
    """A hexagonal unit cell. Mostly used for testing (because less trivial than above)

    Contains 1 hexagon with surrounding 6 half-edges.
    """
    def __init__(self, **attr):
        super().__init__("hex", 2, **attr)

        face, edge = "f", "e_xy"
        self.add_elem(face, 2)
        self.add_elem(edge, 1)

        self.add_bound(face, edge)
        self.add_bound(face, edge, offset=(1,1))

        units = dict(x=(1,0), y=(0,1))

        for i in units:
            edge, vertex = "e_" + i, "v_" + i
            self.add_elem(edge, 1)
            self.add_elem(vertex, 0)

            self.add_bound(face, edge)
            self.add_bound(edge, vertex)
            self.add_bound("e_xy", vertex)

        for i, j in permutations(units):
            edge, vertex = "e_" + i, "v_" + j

            self.add_bound(face, edge, offset=units[j])
            self.add_bound(edge, vertex, offset=units[i])

# Distributed versions

class SixRingCubicCell(CubicCell):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.name = "sixring_cubic"

        self.color()
        self.verify_coloring()

        for i, j, k in ("xyz", "yzx", "zxy"):
            face = f"f_{i}"
            new_face = f"f_{i}_inner"
            new_edge = f"e_{j}{k}"
            self.simple_split(face, new_face, new_edge, (f"e_{j}", (0, 0, 0)), (f"e_{k}", (0, 0, 0)))

        for i, j, k in ("xyz", "yzx", "zxy"):
            edge = f"e_{i}"
            new_edge = f"e_{i}_inner"
            new_face = f"f_{j}{k}"
            self.dual.simple_split(edge, new_edge, new_face, (f"f_{j}_inner", (0, 0, 0)), (f"f_{k}_inner", (0, 0, 0)))

class MaxSplitCubicCell(CubicCell):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.name = "max_split_cubic"
        self.color()
        self.verify_coloring()

        units = dict(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))

        for i, j, k in ("xyz", "yzx", "zxy"):
            self.n_split(f"f_{i}", 
                (f"f_{i}{j}", f"e_{i}_{i}{j}", [(f"e_{j}", (0, 0, 0))]), 
                (f"f_{i}{k}", f"e_{i}_{i}{k}", [(f"e_{k}", (0, 0, 0))]), 
                (f"f_{i}{k}_1", f"e_{i}_{i}{k}_1", [(f"e_{k}", units[j])])
            )

        for i, j, k in ("xyz", "yzx", "zxy"):
            self.dual.n_split(f"e_{i}", 
                (f"e_{i}{k}", f"f_{i}_{i}{k}", [(f"f_{k}{i}", (0, 0, 0))]), 
                (f"e_{i}{j}", f"f_{i}_{i}{j}", [(f"f_{j}{i}", (0, 0, 0))]), 
                (f"e_{i}{j}_1", f"f_{i}_{i}{j}_1", [(f"f_{j}{i}_1", units[k])])
            )

class ThreeDiamondCell(DiamondCell):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.name = "3_diamond"
        self.color()
        self.verify_coloring()

        self.n_split("f_1",
            ("f_11", "e_f11", [("e_y", (0, 0, 1)), ("e_z", (0, 1, 0))]), 
            ("f_12", "e_f12", [("e_x", (0, 0, 1)), ("e_z", (1, 0, 0))])
        )
        self.n_split("f_x", 
            ("f_x1", "e_fx1", [("e_y", (0, 0, 1)), ("e_z", (0, 1, 0))]),
            ("f_x2", "e_fx2", [("e_z", (0, 0, 0)), ("e_1", (0, 0, 1))])
        )
        self.n_split("f_y", 
            ("f_y1", "e_fy1", [("e_x", (0, 0, 1)), ("e_z", (1, 0, 0))]),
            ("f_y2", "e_fy2", [("e_z", (0, 0, 0)), ("e_1", (0, 0, 1))])
        )
        self.n_split("f_z", 
            ("f_z1", "e_fz1", [("e_1", (0, 1, 0)), ("e_y", (0, 0, 0))]),
            ("f_z2", "e_fz2", [("e_1", (1, 0, 0)), ("e_x", (0, 0, 0))])
        )

        self.dual.n_split("e_x",
            ("e_x1", "f_ex1", [("f_1", (0, 1, 0)), ("f_z", (0, 1, 0))]),
            ("e_x2", "f_ex2", [("f_12", (0, 0, 1)), ("f_y1", (0, 0, 1))])
        )

        self.dual.n_split("e_y",
            ("e_y1", "f_ey1", [("f_11", (0, 0, 1)), ("f_x1", (0, 0, 1))]),
            ("e_y2", "f_ey2", [("f_1", (1, 0, 0)), ("f_z", (1, 0, 0))])
        )

        self.dual.n_split("e_z",
            ("e_z1", "f_ez1", [("f_11", (0, 1, 0)), ("f_x1", (0, 1, 0))]),
            ("e_z2", "f_ez2", [("f_12", (1, 0, 0)), ("f_y1", (1, 0, 0))])
        )

        self.dual.n_split("e_1",
            ("e_11", "f_e11", [("f_z1", (0, 1, 0)), ("f_x", (0, 1, 0))]),
            ("e_12", "f_e12", [("f_z2", (1, 0, 0)), ("f_y", (1, 0, 0))])
        )

class TwoThreeDiamondCell(DiamondCell):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.name = "23_diamond"
        self.color()
        self.verify_coloring()

        self.n_split("f_1",
            ("f_11", "e_f11", [("e_x", (0, 1, 0)), ("e_z", (0, 1, 0))]),
            ("f_12", "e_f12", [("e_y", (1, 0, 0)), ("e_z", (1, 0, 0))])
        )

        self.simple_split("f_x", "f_x1", "e_fx1", ("e_y", (0, 0, 0)), ("e_1", (0, 1, 0)), ("e_z", (0, 1, 0)))
        self.simple_split("f_y", "f_y1", "e_fz1", ("e_x", (0, 0, 0)), ("e_1", (1, 0, 0)), ("e_z", (1, 0, 0)))
        self.simple_split("f_z", "f_z1", "e_fy1", ("e_x", (0, 1, 0)), ("e_y", (0, 0, 0)), ("e_1", (0, 1, 0)))

        self.dual.n_split("e_x",
            ("e_x1", "f_ex1", [("f_11", (0, 1, 0)), ("f_z1", (0, 1, 0))]),
            ("e_x2", "f_ex2", [("f_1", (0, 0, 1)), ("f_y", (0, 0, 1))])
        )

        self.dual.n_split("e_y",
            ("e_y1", "f_ey1", [("f_1", (0, 0, 1)), ("f_x", (0, 0, 1))]),
            ("e_y2", "f_ey2", [("f_12", (1, 0, 0)), ("f_z", (1, 0, 0))])
        )

        self.dual.n_split("e_z",
            ("e_z1", "f_ez1", [("f_11", (0, 1, 0)), ("f_x1", (0, 1, 0))]),
            ("e_z2", "f_ez2", [("f_12", (1, 0, 0)), ("f_y1", (1, 0, 0))])
        )

        self.dual.n_split("e_1",
            ("e_11", "f_e11", [("f_z1", (0, 1, 0)), ("f_x1", (0, 1, 0))]),
            ("e_12", "f_e12", [("f_z", (1, 0, 0)), ("f_y1", (1, 0, 0))])
        )

class DoubleEdgeBellCubicCell(DoubleEdgeCubicCell):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.name = "de_cubic_bell"
        self.color()
        self.verify_coloring()

        for i, j, k in ("xyz", "yzx", "zxy"):
            self.simple_split(f"f_{i}", f"f_{i}1", f"e_f{i}1", (f"e_{j}", (0, 0, 0)), (f"e_{k}", (0, 0, 0)), (f"e_{j}{j}", (0, 0, 0)), (f"e_{k}{k}", (0, 0, 0)))
            self.simple_split(f"f_{i}{i}", f"f_{i}{i}1", f"e_f{i}{i}1", (f"e_{j}", (0, 0, 0)), (f"e_{k}", (0, 0, 0)), (f"e_{j}{j}", (0, 0, 0)), (f"e_{k}{k}", (0, 0, 0)))

        for i, j, k in ("xyz", "yzx", "zxy"):
            self.dual.simple_split(f"e_{i}", f"e_{i}1", f"f_e{i}1", (f"f_{j}1", (0, 0, 0)), (f"f_{k}1", (0, 0, 0)), (f"f_{j}{j}1", (0, 0, 0)), (f"f_{k}{k}1", (0, 0, 0)))
            self.dual.simple_split(f"e_{i}{i}", f"e_{i}{i}1", f"f_e{i}{i}1", (f"f_{j}1", (0, 0, 0)), (f"f_{k}1", (0, 0, 0)), (f"f_{j}{j}1", (0, 0, 0)), (f"f_{k}{k}1", (0, 0, 0)))

class DoubleEdgeGHZCubicCell(DoubleEdgeCubicCell):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.name = "de_cubic_ghz"
        self.color()
        self.verify_coloring()

        units = dict(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))

        for i, j, k in ("xyz", "yzx", "zxy"):
            self.n_split(f"f_{i}", 
                (f"f_{i}1", f"e_f{i}1", [(f"e_{j}", (0, 0, 0)), (f"e_{j}{j}", (0, 0, 0))]), 
                (f"f_{i}2", f"e_f{i}2", [(f"e_{k}", (0, 0, 0)), (f"e_{k}{k}", (0, 0, 0))]), 
                (f"f_{i}3", f"e_f{i}3", [(f"e_{k}", units[j]), (f"e_{k}{k}", units[j])])
            )
            self.n_split(f"f_{i}{i}", 
                (f"f_{i}{i}1", f"e_f{i}{i}1", [(f"e_{j}", (0, 0, 0)), (f"e_{j}{j}", (0, 0, 0))]), 
                (f"f_{i}{i}2", f"e_f{i}{i}2", [(f"e_{k}", (0, 0, 0)), (f"e_{k}{k}", (0, 0, 0))]), 
                (f"f_{i}{i}3", f"e_f{i}{i}3", [(f"e_{k}", units[j]), (f"e_{k}{k}", units[j])])
            )

        for i, j, k in ("xyz", "yzx", "zxy"):
            self.dual.n_split(f"e_{i}", 
                (f"e_{i}1", f"f_e{i}1", [(f"f_{k}1", (0, 0, 0)), (f"f_{k}{k}1", (0, 0, 0))]), 
                (f"e_{i}2", f"f_e{i}2", [(f"f_{j}2", (0, 0, 0)), (f"f_{j}{j}2", (0, 0, 0))]), 
                (f"e_{i}3", f"f_e{i}3", [(f"f_{j}3", units[k]), (f"f_{j}{j}3", units[k])])
            )
            self.dual.n_split(f"e_{i}{i}", 
                (f"e_{i}{i}1", f"f_e{i}{i}1", [(f"f_{k}1", (0, 0, 0)), (f"f_{k}{k}1", (0, 0, 0))]), 
                (f"e_{i}{i}2", f"f_e{i}{i}2", [(f"f_{j}2", (0, 0, 0)), (f"f_{j}{j}2", (0, 0, 0))]), 
                (f"e_{i}{i}3", f"f_e{i}{i}3", [(f"f_{j}3", units[k]), (f"f_{j}{j}3", units[k])])
            )
