# Python
from __future__ import annotations
from pathlib import Path
from typing import Iterable
from itertools import product
from collections import Counter, defaultdict
import multiprocessing as mp

# Math
import numpy as np
import sympy as sp

# Manticore
from manticore.experiments import Experiment
from manticore.channels import Channel, IsotropicChannel, BoundaryChannel

from manticore.geometry import Lattice
from manticore.errors.library import Phenomenological, Erasure, PhenomenologicalErasure, WeightedPhenomenological, WeightedErasure, WeightedPhenomenologicalErasure

DEFAULT_SIZES = [5, 7, 9, 11]

def divide_shots(min_shots: int, num_processes: int):
    shots_per_process, extra = divmod(min_shots, num_processes)

    if extra:
        shots_per_process += 1
    
    num_shots = num_processes * shots_per_process
    return num_shots, shots_per_process

def parameter_product(**parameter_ranges: Iterable) -> list[dict]:
    """Convert parameter ranges to distinct parameter sets by taking their product.

    Args:
        **parameter_ranges (Iterable): Lists of parameter ranges for each parameter.

    Returns:
        list[dict]: A list of parameters bound to values.
    """
    sets = []
    for value_set in product(*parameter_ranges.values()):
        parameter_set = dict(zip(parameter_ranges.keys(), value_set))
        sets.append(parameter_set)

    return sets

def fan_parameters(p_min, p_max, q_min, q_max, num_lines):
    i = np.arange(1, num_lines)  # Don't do i = 0 since slope is inf and this leads to divide by zero errors
    slopes = (q_max + q_min)/(p_max + p_min) * (num_lines - 1 - i) / i

    c_p = np.zeros(num_lines)
    c_q = np.ones(num_lines)
    c_p[1:] = 1 / (1 + slopes)
    c_q[1:] = slopes / (1 + slopes)

    t_min = 1 / (c_p/p_min + c_q/q_min)
    t_max = 1 / (c_p/p_max + c_q/q_max)

    return c_p, c_q, t_min, t_max

class DecodingExperiment(Experiment):
    # TODO outdated already (see ThresholdExperiment for newer implementation)
    def __init__(self, channel: Channel, min_shots: int, num_processes: int = None):
        """Create a new decoding experiment over a specific channel. This experiment uses multiprocessing.

        Args:
            channel (Channel): The channel to sample.
            min_shots (int): The minimum amount of shots to sample. The actual number might be slightly higher if the number
                of shots cannot be exactly divided over the number of processes.
        """
        super().__init__(f"decode_{channel.name}")
        self.channel = channel
        self.num_processes = num_processes or mp.cpu_count()
        self.num_shots, self.shots_per_process = divide_shots(min_shots, self.num_processes)

    def _sample_channel(self, i, queue):
        rng = np.random.default_rng(seed=i)  # Creates new rng for every subprocess
        queue.put(self.channel.sample(self.shots_per_process, rng))
    
    def run(self, data):
        processes = []
        output = mp.Queue()

        total_primal = Counter()
        total_dual = Counter()

        for i in range(self.num_processes):
            process = mp.Process(target=self._sample_channel, args=(i, output))
            processes.append(process)

        for process in processes:
            process.start()

        for process in processes:
            primal_counts, dual_counts = output.get()
            total_primal += primal_counts
            total_dual += dual_counts

        for process in processes:
            process.join()

        data["primal_counts"] = dict(total_primal)
        data["dual_counts"] = dict(total_dual)

    def to_json(self, path=None):
        pass

class ThresholdExperiment(Experiment):
    def __init__(self, channel, num_shots, parameter_sets, num_processes=None):
        super().__init__(f"threshold_{channel.name}")
        self.channel = channel
        self.num_processes = num_processes or mp.cpu_count()
        self.num_shots, self.shots_per_process = divide_shots(num_shots, self.num_processes)
        self.parameter_sets = parameter_sets

    def _sample_channel(self, i, j, **parameters):
        channel = self.channel.bind_parameters(**parameters)
        rng = np.random.default_rng(seed=(i, j))  # Creates new rng for every subprocess

        return i, channel.sample(self.shots_per_process, rng)

    def run(self, data):
        data["parameter_sets"] = self.parameter_sets
        data["num_shots"] = self.num_shots
        data["logicals"] = list(self.channel.unit_homology.operators.keys())
        data["logical_errors"] = [Counter() for _ in range(len(self.parameter_sets))]

        with mp.Pool(processes=self.num_processes) as pool:
            jobs = []
            
            # Start jobs
            for i, parameter_set in enumerate(self.parameter_sets):
                for j in range(self.num_processes):
                    job = pool.apply_async(self._sample_channel, args=(i, j), kwds=parameter_set)
                    jobs.append(job)

            # Collect results
            for job in jobs:
                i, logical_errors = job.get()
                data["logical_errors"][i] += logical_errors

            pool.close()
            pool.join()

    def to_json(self, path=None):
        pass

class ParameterizedThreshold(ThresholdExperiment):
    def __init__(self, unit_cell: UnitCell, error_model: ErrorModel,
        resolution=7, num_shots=50000, sizes=None, num_processes=None, num_lines=9, 
        **kwargs: int | float | tuple[float, float]
    ):
        if sizes == None:
            sizes = DEFAULT_SIZES

        s = sp.symbols("s")  # Lattice size
        lat = Lattice((s, s, s))
        channel = IsotropicChannel(unit_cell, error_model, lat)

        # Bind constant parameter values
        constant_parameters = {}
        for name, arg in list(kwargs.items()):
            try:
                if len(arg) == 1:
                    arg = arg[0]
            except TypeError:
                pass

            if isinstance(arg, (float, int)):
                constant_parameters[name] = arg
                del kwargs[name]

        # Create parameter ranges for the remaining parameters
        parameter_sets = []
        num_params = len(kwargs)
        if num_params == 0:
            raise ValueError(f"threshold experiment has no parameter sweeps. Check your parameters")
        if num_params == 1:
            p, (min_val, max_val) = kwargs.popitem()
            
            p_values = np.linspace(min_val, max_val, resolution)
            parameter_sets.extend(parameter_product(s=sizes, **{p: p_values}))
        elif num_params == 2:
            p1, (min1, max1) = kwargs.popitem()
            p2, (min2, max2) = kwargs.popitem()

            c_1, c_2, t = sp.symbols((f"c_{p1}", f"c_{p2}", "t"))
            channel.bind_parameters(inplace=True, **{p1: c_1*t, p2: c_2*t})

            c_1s, c_2s, t_mins, t_maxs = fan_parameters(min1, max1, min2, max2, num_lines)
            for c_1_values, c_2_values, t_min, t_max in zip(c_1s, c_2s, t_mins, t_maxs):
                t_values = np.linspace(t_min, t_max, resolution)
                param_set_one_line = parameter_product(s=sizes, t=t_values)

                for param_set in param_set_one_line:
                    param_set[f"c_{p1}"] = c_1_values
                    param_set[f"c_{p2}"] = c_2_values

                parameter_sets.extend(param_set_one_line)
        else:
            raise ValueError(f"threshold experiment for {num_params} > 2 parameter sweeps not supported")

        for parameter_set in parameter_sets:
            parameter_set.update(constant_parameters)
        
        super().__init__(channel, num_shots, parameter_sets, num_processes=num_processes)

from manticore.utils import time_method, count_method, count_generator, count_lens

class UFRuntimeExperiment(Experiment):
    def __init__(self, unit_cell, p_min, p_max, num_p, num_shots, sizes=None, num_processes=None):
        if sizes == None:
            sizes = DEFAULT_SIZES

        p, r = sp.symbols(tuple("pr"))

        error_model = Phenomenological(unit_cell, p)
        lat = Lattice((r, r, r))
        self.channel = IsotropicChannel(unit_cell, error_model, lat)

        p_values = np.linspace(p_min, p_max, num_p)
        self.parameter_sets = parameter_product(r=sizes, p=p_values)

        self.num_processes = num_processes or mp.cpu_count()
        self.num_shots, self.shots_per_process = divide_shots(num_shots, self.num_processes)

        super().__init__(f"threshold_{self.channel.name}")

    def _sample_channel(self, i, j, **parameters):
        channel = self.channel.bind_parameters(**parameters)
        rng = np.random.default_rng(seed=(i, j))  # Creates new rng for every subprocess

        # runtime_data = {k: 0 for k in ("primal", "dual")}
        # counts_data = defaultdict(lambda: 0)
        # for name, decoder in zip(runtime_data, (channel.primal_decoder, channel.dual_decoder)):
        #     decoder.decode = time_method(runtime_data, name)(decoder.decode)

        from manticore.decoders import uf
        # from manticore.decoders import uf_old
        profile_data = defaultdict(lambda: 0)
        # profile_data = dict(find=0, union=0, decode=0)
        # Cluster.union = count_method(profile_data)(Cluster.union)
        # old_find = uf.find
        # old_union = uf.union
        # old_decode = uf.UFDecoder.decode

        # uf.find = count_method(profile_data)(old_find)
        # uf.union = count_method(profile_data)(old_union)
        # uf.UFDecoder.decode = time_method(profile_data)(old_decode)

        old_peel = uf.Cluster.peeling_tree
        old_correct = uf.Leaf.root_path
        uf.Cluster.peeling_tree = count_generator(profile_data, "peel")(old_peel)
        uf.Leaf.root_path = count_generator(profile_data, "correct")(old_correct)

        # old_peel = uf_old.UFDecoder.dfs_tree
        # old_correct = uf_old.UFDecoder.peel
        # uf_old.UFDecoder.dfs_tree = count_lens(profile_data, "peel")(old_peel)
        # uf_old.UFDecoder.peel = count_lens(profile_data, "correct")(old_correct)  # Yes this is correct

        samples = channel.sample(self.shots_per_process, rng)

        uf.Cluster.peeling_tree = old_peel
        uf.Leaf.root_path = old_correct

        # uf_old.UFDecoder.dfs_tree = old_peel
        # uf_old.UFDecoder.peel = old_correct

        # uf.find = old_find
        # uf.union = old_union
        # uf.UFDecoder.decode = old_decode

        return i, samples, dict(profile_data) # runtime_data, counts_data

    def run(self, data):
        data["parameter_sets"] = self.parameter_sets
        data["num_shots"] = self.num_shots
        data["logicals"] = list(self.channel.unit_homology.operators.keys())
        data["logical_errors"] = [Counter() for _ in range(len(self.parameter_sets))]
        data["profile_data"] = [Counter() for _ in range(len(self.parameter_sets))]
        # data["primal_decode_runtime"] = len(self.parameter_sets) * [0]
        # data["dual_decode_runtime"] = len(self.parameter_sets) * [0]
        # data["union_counts"] = len(self.parameter_sets) * [0]
        # data["find_counts"] = len(self.parameter_sets) * [0]

        with mp.Pool(processes=self.num_processes) as pool:
            jobs = []
            
            # Start jobs
            for i, parameter_set in enumerate(self.parameter_sets):
                for j in range(self.num_processes):
                    job = pool.apply_async(self._sample_channel, args=(i, j), kwds=parameter_set)
                    jobs.append(job)

            # Collect results
            for job in jobs:
                i, logical_errors, profile_data = job.get()
                data["logical_errors"][i] += logical_errors
                data["profile_data"][i] += profile_data
                # data["primal_decode_runtime"][i] += runtime_data["primal"]
                # data["dual_decode_runtime"][i] += runtime_data["dual"]
                # data["union_counts"][i] += counts_data["union"]
                # data["find_counts"][i] += counts_data["find"]

            pool.close()
            pool.join()

    def to_json(self, path=None):
        pass

class PhenomThreshold(ThresholdExperiment):
    def __init__(self, unit_cell, p_min, p_max, num_p, num_shots, sizes=None, weighted=False, boundary=False):
        if sizes == None:
            sizes = DEFAULT_SIZES

        p, r = sp.symbols(tuple("pr"))
        lat = Lattice((r, r, r))

        if weighted:
            error_model = WeightedPhenomenological(unit_cell, p)
        else:
            error_model = Phenomenological(unit_cell, p)

        if boundary:
            channel = BoundaryChannel(unit_cell, error_model, lat)
        else:
            channel = IsotropicChannel(unit_cell, error_model, lat)

        p_values = np.linspace(p_min, p_max, num_p)
        parameter_sets = parameter_product(r=sizes, p=p_values)

        super().__init__(channel, num_shots, parameter_sets)

class ErasureThreshold(ThresholdExperiment):
    def __init__(self, unit_cell, p_min, p_max, num_p, num_shots, sizes=None, weighted=False, boundary=False):
        if sizes == None:
            sizes = DEFAULT_SIZES

        p, r = sp.symbols(tuple("pr"))
        lat = Lattice((r, r, r))

        if weighted:
            error_model = WeightedErasure(unit_cell, p)
        else:
            error_model = Erasure(unit_cell, p)
        
        if boundary:
            channel = BoundaryChannel(unit_cell, error_model, lat)
        else:
            channel = IsotropicChannel(unit_cell, error_model, lat)

        p_values = np.linspace(p_min, p_max, num_p)
        parameter_sets = parameter_product(r=sizes, p=p_values)

        super().__init__(channel, num_shots, parameter_sets)

class PhenomErasureThreshold(ThresholdExperiment):
    def __init__(self, unit_cell, min_p, max_p, min_q, max_q, num_p, num_lines, num_shots, sizes=None, weighted=False, boundary=True):
        if sizes == None:
            sizes = DEFAULT_SIZES

        # p: phenom probability
        # q: erasure probability
        # r: lattice size
        # t: shared parameter for p and q
        p, q, r, t = sp.symbols(tuple("pqrt"))
        lat = Lattice((r, r, r))

        # Define channel
        if weighted:
            error_model = WeightedPhenomenologicalErasure(unit_cell, p, q)
        else:
            error_model = PhenomenologicalErasure(unit_cell, p, q)

        if boundary:
            channel = BoundaryChannel(unit_cell, error_model, lat)
        else:
            channel = IsotropicChannel(unit_cell, error_model, lat)

        # Rebind p and q as functions of a single parameter t
        c_p, c_q = sp.symbols(("c_p", "c_q"))
        channel.bind_parameters(inplace=True, p=c_p*t, q=c_q*t)

        # Calculate parametrizations of the 'fan'
        c_ps, c_qs, t_mins, t_maxs = fan_parameters(min_p, max_p, min_q, max_q, num_lines)

        parameter_sets = []
        for c_p, c_q, t_min, t_max in zip(c_ps, c_qs, t_mins, t_maxs):
            t_values = np.linspace(t_min, t_max, num_p)
            param_set_one_line = parameter_product(r=sizes, t=t_values)

            for param_set in param_set_one_line:
                param_set["c_p"] = c_p
                param_set["c_q"] = c_q

            parameter_sets.extend(param_set_one_line)
        
        super().__init__(channel, num_shots, parameter_sets)

if __name__ == "__main__":
    from manticore.geometry.library import CubicCell, DiamondCell, DoubleEdgeCubicCell, TriamondCell
    # experiment = UFRuntimeExperiment(CubicCell(), 0.01, 0.04, num_p=4, num_shots=50000, sizes=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    experiment = PhenomErasureThreshold(CubicCell(), 0.02, 0.03, 0.2, 0.3, 3, 9, 5000, sizes=[3, 5, 7])
    experiment = PhenomThreshold(CubicCell(), 0.02, 0.03, num_p=3, num_shots=2000, sizes=[5, 7], boundary=True)
    # experiment = MonolithicThreshold(CubicCell(), 0.003, 0.0045)
    result = experiment.run_experiment()
    result.to_json()
