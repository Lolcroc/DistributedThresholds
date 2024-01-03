# Python
import argparse

# Math
import numpy as np
import sympy as sp

# Manticore
from manticore.geometry import Lattice
from manticore.geometry.library import CubicCell, DiamondCell, TriamondCell, DoubleEdgeCubicCell
from manticore.errors.library import Monolithic
from manticore.channels import Channel, IsotropicChannel
from manticore.experiments.library import divide_shots, parameter_product, fan_parameters, ThresholdExperiment

DEFAULT_SIZES = [5, 7, 9, 11]
DEFAULT_NUM_SHOTS = 50000
DEFAULT_NUM_P = 7

class MonolithicThreshold(ThresholdExperiment):
    def __init__(self, unit_cell, p_min, p_max, num_p=7, num_shots=50000, num_processes=None, sizes=None, color_ordering=None):
        if sizes == None:
            sizes = DEFAULT_SIZES

        p, r = sp.symbols(tuple("pr"))

        unit_cell.color()
        error_model = Monolithic(unit_cell, p, color_ordering)
        if color_ordering is not None:
            error_model.name += "".join(str(i) for i in color_ordering)

        lat = Lattice((r, r, r))
        channel = IsotropicChannel(unit_cell, error_model, lat)

        p_values = np.linspace(p_min, p_max, num_p)
        parameter_sets = parameter_product(r=sizes, p=p_values)

        super().__init__(channel, num_shots, parameter_sets, num_processes=num_processes)

EXPERIMENTS = {
    "cubic": CubicCell(),  # 0.003 - 0.006
    "diamond": DiamondCell(),  # 0.005 - 0.008
    "triamond": TriamondCell(),  # 0.002 - 0.005
    "double-edge-cubic": DoubleEdgeCubicCell()  # 0.005 - 0.008
}

class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

    def __str__(self):
        return f"[{self.start},{self.end}]"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("unit_cell", type=str, choices=EXPERIMENTS, help="which unit cell to do")
    parser.add_argument("start_p", type=float, choices=Range(0.0, 1.0), help="starting value of p")
    parser.add_argument("stop_p", type=float, choices=Range(0.0, 1.0), help="stopping value of p")
    parser.add_argument("-r", "--resolution", type=int, default=DEFAULT_NUM_P, help=f"number of error rates to sample (default {DEFAULT_NUM_P})")
    parser.add_argument("-n", "--num_shots", type=int, default=DEFAULT_NUM_SHOTS, help=f"number of shots for each data point (default {DEFAULT_NUM_SHOTS})")
    parser.add_argument("-s", "--sizes", nargs="+", type=int, default=DEFAULT_SIZES, help=f"lattice sizes (default {DEFAULT_SIZES})")
    parser.add_argument("-o", "--ordering", nargs="+", type=int, default=None, help="ordering of CZ gates by its color index (default None, i.e. in order of color)")
    args = parser.parse_args()

    experiment = MonolithicThreshold(EXPERIMENTS[args.unit_cell], args.start_p, args.stop_p, 
        num_p=args.resolution, num_shots=args.num_shots, sizes=args.sizes, color_ordering=args.ordering
    )

    result = experiment.run_experiment()
    result.to_json()
