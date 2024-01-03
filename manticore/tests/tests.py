import pytest

import numpy as np

from collections import defaultdict

from geometry import Crystal
from prebuilt import SquareCell, CubicCell, HexCell

NUM_ELEMENTS = dict(square=[1, 2, 1], cubic=[1, 3, 3, 1], hex=[2, 3, 1])


SQUARE_CRYSTALS = [("square", (1, 1)), ("square", (1, 5)), ("square", (5, 5))]
CUBIC_CRYSTALS = [("cubic", (1, 1, 1)), ("cubic", (1, 5, 1)), ("cubic", (5, 5, 5))]
HEX_CRYSTALS = [("hex", (2, 2)), ("hex", (2, 5)), ("hex", (5, 5))]

ALL_CRYSTALS = SQUARE_CRYSTALS + CUBIC_CRYSTALS + HEX_CRYSTALS
# ALL_CRYSTALS = [("hex", (1, 1))]

@pytest.fixture(scope="module")
def cell(request):
    if request.param == "cubic":
        return CubicCell(name=request.param)
    elif request.param == "square":
        return SquareCell(name=request.param)
    elif request.param == "hex":
        return HexCell(name=request.param)
    else:
        raise ValueError("Invalid cell type")

@pytest.mark.parametrize("cell", NUM_ELEMENTS, indirect=True)
def test_num_elems(cell):
    num_elems = NUM_ELEMENTS[cell.name]

    for d in range(cell.dim + 1):
        assert sum(1 for _ in cell.filter_nodes(dim=d)) == num_elems[d]

@pytest.mark.parametrize("cell, reps", ALL_CRYSTALS, indirect=["cell"])
def test_crystal_cycles(cell, reps):
    crystal = Crystal(cell, reps)
    size = np.prod(reps)

    num_elems = NUM_ELEMENTS[cell.name]

    for i in range(1, crystal.dim + 1):
        di = crystal.boundary_matrix(i)

        assert di.shape == (num_elems[i] * size, num_elems[i - 1] * size)

    for i in range(2, crystal.dim + 1):
        di, dj = map(crystal.boundary_matrix, (i, i - 1))

        assert all((di @ dj).data % 2 == 0)

@pytest.mark.parametrize("cell", NUM_ELEMENTS, indirect=True)
def test_cell_cycles(cell):
    # TODO Pretty annoying; need something better
    def add_offsets(*args):
        offsets = map(lambda o: np.fromstring(o, dtype=int, sep=' '), args)
        return ' '.join(str(n) for n in sum(offsets))

    for dim in range(2, cell.dim):
        for u in cell.filter_nodes(dim=dim):
            ends = defaultdict(set)

            for _, v, p in cell.edges(u, data='offset'):
                for _, w, q in cell.edges(v, data='offset'):
                    total_offset = add_offsets(p, q)
                    ends[total_offset] ^= {w}
            
            assert all(len(e) == 0 for e in ends.values())

# @pytest.mark.parametrize("cell, reps", ALL_CRYSTALS, indirect=["cell"])
# def test_boundaries_degree(cell, reps):
#     bounds = Crystal(cell, reps).cardinal_boundary

#     expected_degree = 2 * cell.dim - reps.count(1) # TODO not a proper test
#     assert all(len(b) == expected_degree for b in bounds)

@pytest.mark.parametrize("cell", NUM_ELEMENTS, indirect=True)
def test_trivial_errors(cell):
    reps = cell.dim * (5,)
    size = np.prod(reps)
    crystal = Crystal(cell, reps)

    num_elems = NUM_ELEMENTS[cell.name]

    no_errors, no_syndromes = crystal.create_errors(0)
    full_errors, full_syndromes = crystal.create_errors(1)

    assert len(no_errors) == 0
    assert len(no_syndromes) == 0
    assert len(full_errors) == size * num_elems[cell.dim - 1]
    assert len(full_syndromes) == 0 # All errors cancel

if __name__ == '__main__':
    pytest.main(["-s", __file__])