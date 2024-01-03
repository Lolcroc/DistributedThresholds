# Python
from __future__ import annotations
import argparse

# Math
import numpy as np
import sympy as sp

# Manticore
from manticore.geometry.library import CubicCell, DiamondCell, TriamondCell, DoubleEdgeCubicCell
from manticore.errors.library import SixRing, CubicGHZFusion, ThreeDiamondFusion, TwoThreeDiamondFusion, DoubleEdgeBell, DoubleEdgeGHZFusion, SixRingErasure, SixRingErasureDouble
from manticore.experiments.library import ParameterizedThreshold

DEFAULT_SIZES = [5, 7, 9, 11]
DEFAULT_NUM_SHOTS = 50000
DEFAULT_NUM_P = 7

# p: 0.003 0.006
# p_i: 0.008 0.012
class SixRingThreshold(ParameterizedThreshold):
    def __init__(self, *args, **kwargs):
        p, p_i = sp.symbols(("p", "p_i"))

        error_model = SixRing(p, p_i)
        unit_cell = CubicCell()

        super().__init__(unit_cell, error_model, *args, **kwargs)

# p: 0.0025 0.0035
# p_i: 0.008 0.012
class CubicGHZFusionThreshold(ParameterizedThreshold):
    def __init__(self, *args, **kwargs):
        p, p_i = sp.symbols(("p", "p_i"))

        error_model = CubicGHZFusion(p, p_i)
        unit_cell = CubicCell()
        
        super().__init__(unit_cell, error_model, *args, **kwargs)

# p: 0.004 0.006
# p_i: 0.015 0.025
class ThreeDiamondFusionThreshold(ParameterizedThreshold):
    def __init__(self, *args, **kwargs):
        p, p_i = sp.symbols(("p", "p_i"))

        error_model = ThreeDiamondFusion(p, p_i)
        unit_cell = DiamondCell()
        
        super().__init__(unit_cell, error_model, *args, **kwargs)

# p: 0.005 0.007
# p_i: 0.02 0.025
class TwoThreeDiamondFusionThreshold(ParameterizedThreshold):
    def __init__(self, *args, **kwargs):
        p, p_i = sp.symbols(("p", "p_i"))

        error_model = TwoThreeDiamondFusion(p, p_i)
        unit_cell = DiamondCell()
        
        super().__init__(unit_cell, error_model, *args, **kwargs)

# p: 0.002 0.003
# p_i: 0.004 0.007
class DoubleEdgeBellThreshold(ParameterizedThreshold):
    def __init__(self, *args, **kwargs):
        p, p_i = sp.symbols(("p", "p_i"))

        error_model = DoubleEdgeBell(p, p_i)
        unit_cell = DoubleEdgeCubicCell()
        
        super().__init__(unit_cell, error_model, *args, **kwargs)

# p: 0.004 0.005
# p_i: 0.012 0.017
class DoubleEdgeGHZFusionThreshold(ParameterizedThreshold):
    def __init__(self, *args, **kwargs):
        p, p_i = sp.symbols(("p", "p_i"))

        error_model = DoubleEdgeGHZFusion(p, p_i)
        unit_cell = DoubleEdgeCubicCell()
        
        super().__init__(unit_cell, error_model, *args, **kwargs)

# # p: 0.003 0.006
# # p_i: 0.008 0.012
# class SixRingErasureThreshold(ParameterizedThreshold):
#     def __init__(self, *args, **kwargs):
#         p, p_i = sp.symbols(("p", "p_i"))

#         error_model = SixRingErasure(p, p_i)
#         unit_cell = CubicCell()
        
#         super().__init__(unit_cell, error_model, *args, **kwargs)

# p: 0.003 0.006
# p_i: 0.008 0.012
class SixRingErasureDoubleThreshold(ParameterizedThreshold):
    def __init__(self, *args, **kwargs):
        p, p_i = sp.symbols(("p", "p_i"))

        error_model = SixRingErasureDouble(p, p_i)
        unit_cell = CubicCell()
        
        super().__init__(unit_cell, error_model, *args, **kwargs)

from manticore.errors.library import SixRingErasureNew

class SixRingErasureThreshold(ParameterizedThreshold):
    def __init__(self, *args, **kwargs):
        p, p_i = sp.symbols(("p", "p_i"))

        error_model = SixRingErasureNew(p, p_i)
        unit_cell = CubicCell()
        
        super().__init__(unit_cell, error_model, *args, **kwargs)

class OneOrTwo(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        if not 1 <= len(values) <= 2:
            raise argparse.ArgumentTypeError(f'argument "{self.dest}" requires either 1 or 2 arguments')
        setattr(args, self.dest, values)

EXPERIMENTS = {
    "six-ring": SixRingThreshold,
    "cubic-ghz-fusion": CubicGHZFusionThreshold,
    "3-diamond-fusion": ThreeDiamondFusionThreshold,
    "23-diamond-fusion": TwoThreeDiamondFusionThreshold,
    "double-edge-bell": DoubleEdgeBellThreshold,
    "double-edge-ghz-fusion": DoubleEdgeGHZFusionThreshold,
    "six-ring-erasure": SixRingErasureThreshold,
    "six-ring-double": SixRingErasureDoubleThreshold
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, choices=EXPERIMENTS, help="which experiment to do")
    parser.add_argument("-r", "--resolution", type=int, default=DEFAULT_NUM_P, help=f"number of error rates to sample (default {DEFAULT_NUM_P})")
    parser.add_argument("-n", "--num_shots", type=int, default=DEFAULT_NUM_SHOTS, help=f"number of shots for each data point (default {DEFAULT_NUM_SHOTS})")
    parser.add_argument("-s", "--sizes", nargs="+", type=int, default=DEFAULT_SIZES, help=f"lattice sizes (default {DEFAULT_SIZES})")
    parsed, unknown = parser.parse_known_args()

    # Hackish way to add kwargs
    for arg in unknown:
        if arg.startswith("-"):
            parser.add_argument(arg, type=float, nargs='+', action=OneOrTwo)

    kwargs = vars(parser.parse_args())
    name = kwargs.pop("name")

    experiment = EXPERIMENTS[name](**kwargs)

    result = experiment.run_experiment()
    result.to_json()
