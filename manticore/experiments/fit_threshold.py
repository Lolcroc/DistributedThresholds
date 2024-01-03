# Python
import json
import random
from pathlib import Path
from collections import defaultdict

# Math
import numpy as np
from lmfit import Parameters, minimize, fit_report
import pandas as pd
from scipy.stats import chi2


def mle_binomial(errors, reps, z=1.96, include_z=True):
    p = errors/reps

    if include_z:
        return p, z * (np.sqrt(np.multiply(p, 1-p)/reps) + 0.5/reps)
    else:
        return p, np.sqrt(np.multiply(p, 1-p)/reps)


def residual(pars, x, data=None, eps=None, fit_type=2):
    parvals = pars.valuesdict()
    num_lines = parvals['num_lines']
    order = parvals['order']
    ep = x[0]
    ls = None if fit_type == 1 else x[1]

    X = np.vander(ep - parvals["x_th"], order + 1, increasing=True)

    param_matrix = np.zeros((order + 1, num_lines), dtype=float)
    param_matrix[0] = parvals["y_th"]

    for i in range(1, order + 1):
        for j in range(num_lines):
            param_matrix[i, j] = parvals[f"p{j}_{i}"] if fit_type == 1 \
                else parvals[f"p{i}"] * (ls[j] ** (i / parvals["nu"]))

    model = X @ param_matrix

    if data is None:
        return model
    if eps is None:
        return model - data

    return (model - data) / eps


def fit_once(mean, conf, order, x_init=None, y_init=0.1, coeff_init=0, fit_type=2):
    order = order       # if fit_type == 1 else 2
    error_probs = mean.index.values
    lattice_sizes = mean.columns.values
    min_prob, max_prob = error_probs[0], error_probs[-1]

    x_init = x_init or 0.5 * (min_prob + max_prob)

    # Define fitting parameters
    fit_params = Parameters()
    fit_params.add('x_th', value=x_init, min=min_prob, max=max_prob)
    fit_params.add('y_th', value=y_init)
    if fit_type == 2:
        fit_params.add('nu', value=0.974)

    num_lines = mean.shape[1]
    for i in range(1, order + 1):
        if fit_type == 2:
            fit_params.add(f"p{i}", value=coeff_init)
        else:
            for j in range(num_lines):
                fit_params.add(f"p{j}_{i}", value=coeff_init)

    fit_params.add('num_lines', value=num_lines, vary=False)
    fit_params.add('order', value=order, vary=False)
    fit_variables = (error_probs, lattice_sizes)

    # Perform fit
    result_fit = minimize(residual, fit_params, args=(fit_variables,),
                          kws={'data': mean, 'eps': conf, 'fit_type': fit_type},
                          scale_covar=True)
    return result_fit


def fit_threshold(mean, conf, order="auto", return_fit=False, max_tries=5, min_order=2, max_order=4, gof_sig=0.05,
                  seed=0, fit_type=2):
    if order == "auto":     # and fit_type == 1:
        gof = 0
        order = min_order - 1

        while gof < gof_sig and order < max_order:
            order += 1
            fit_result = fit_once(mean, conf, order, fit_type=fit_type)

            # Determine the goodness of fit
            dof = mean.shape[1] * (mean.shape[0] - order) - 2  # Degrees of freedom
            gof = chi2.sf(fit_result.chisqr, df=dof)

            # print(f"Order {order} gave g.o.f. {gof}")

    else:
        fit_result = fit_once(mean, conf, order, fit_type=fit_type)

    random.seed(seed)
    tries = 1
    while not (fit_result.success and fit_result.errorbars) and tries <= max_tries:  # Fit didn't work properly
        if tries == max_tries:
            raise RuntimeError(f"threshold fit failed, fit report:\n{fit_report(fit_result)}")

        x_init = random.uniform(fit_result.params["x_th"].min, fit_result.params["x_th"].max)
        # print(f"Threshold fit failed, trying again with x_th={x_init} and order={order}")

        fit_result = fit_once(mean, conf, order, x_init=x_init, fit_type=fit_type)
        tries += 1

    if return_fit:
        return fit_result

    x_th_param = fit_result.params["x_th"]
    return x_th_param.value, x_th_param.stderr


def load_json(path):
    path = Path(path)

    with open(path) as f:
        return json.load(f)


def logical_error_rates(path, threshold_param="p", size_param="r", error_when: tuple[str] = None, include_z=False):
    if error_when is None:
        error_when = ('primal_logical_x', 'dual_logical_y')
    
    result = load_json(path)
    data = result["data"]

    logical_indices = tuple(data["logicals"].index(e) for e in error_when)
    logical_counts_by_param = defaultdict(lambda: pd.DataFrame(dtype=int))

    # Collect counts and group by parameter sets
    for params, error_cts in zip(data["parameter_sets"], data["logical_errors"]):
        try:
            prob = params.pop(threshold_param)
            size = params.pop(size_param)
        except KeyError as e:
            raise ValueError(f"wrong parameter labels {threshold_param} and/or {size_param} for threshold fit") from e

        total_counts = 0

        for error_string, cts in error_cts.items():
            if any(error_string[i] == "1" for i in logical_indices):
                total_counts += cts

        remaining_params = tuple(sorted(params.items()))
        logical_counts_by_param[remaining_params].loc[prob, size] = total_counts

    # Calculate MLE estimates based on number of shots
    for params, logical_counts in logical_counts_by_param.items():
        logical_counts.index.name = threshold_param
        logical_counts.columns.name = size_param

        mean, conf = mle_binomial(logical_counts, data["num_shots"], include_z=include_z)

        yield dict(params), mean, conf


def main(mean, conf):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    plt.style.use(['science', 'nature'])

    fig, ax = plt.subplots(figsize=(13, 9), dpi=120)

    N = 1000
    error_probs = mean.index
    x = np.linspace(error_probs[0], error_probs[-1], N)

    fit_result = fit_threshold(mean, conf, order="auto", return_fit=True, seed=420)
    # fit_result = fit_threshold(mean, conf, order=2, return_fit=True, seed=420)
    Y = residual(fit_result.params, x)

    x_th = fit_result.params["x_th"].value
    x_err = fit_result.params["x_th"].stderr

    # print(x_th, x_err) #, f"({x_th-x_err}, {x_th+x_err})")

    for i, size in enumerate(mean):
        col = cmap(i)
        ax.errorbar(error_probs, mean[size], yerr=conf[size], capsize=2, c=col, ls='', marker='o', ms=3, mfc='w', ecolor=col, mec=col, label=f"{size}")

        ax.plot(x, Y[:, i], color=col, ls='-', lw=1)

    xlim_offset = 0.01 * (error_probs[-1] - error_probs[0])
    xlims = error_probs[0]-xlim_offset, error_probs[-1]+xlim_offset
    ax.set_xlim(xlims)
    ax.set_xticks(error_probs)
    ax.set_xticklabels((f"{100*p:.3f}" for p in error_probs))

    # yticks = [0.01, 0.02, 0.05, 0.1, 0.3]
    # ax.set_yscale("log")
    # ax.set_ylim(0, 0.45)
    # ax.set_yticks(yticks)
    # ax.set_yticklabels((f"{100*p:.0f}" for p in yticks))
    ax.set_xlabel('depolarizing error rate $p$ (\%)')
    ax.set_ylabel('logical error rate $p_L$ (\%)')
    ax.legend(loc='upper left', title='size')
    plt.show()


if __name__ == "__main__":
    # New
    # path = "20220404-132918_threshold_cubic_r_r_r_cubic_monolithic_p.json"
    # path = "20220404-132917_threshold_diamond_r_r_r_diamond_monolithic_p.json"
    # path = "20220404-132921_threshold_double_edge_cubic_r_r_r_double_edge_cubic_monolithic_p.json"
    # path = "20220404-135912_threshold_triamond_r_r_r_triamond_monolithic_p.json"

    # path = "20220404-194426_threshold_cubic_r_r_r_cubic_monolithic_p.json"  # Criss cross
    # path = "20220405-200235_threshold_cubic_r_r_r_cubic_monolithic_p.json"
    # path = "20220407-160836_threshold_double_edge_cubic_r_r_r_double_edge_cubic_monolithic_p.json"
    # path = "20220407-161240_threshold_double_edge_cubic_r_r_r_double_edge_cubic_monolithic_p.json"
    # path = "20220407-163008_threshold_double_edge_cubic_r_r_r_double_edge_cubic_monolithic_p.json"
    # path = "20220407-164512_threshold_double_edge_cubic_r_r_r_double_edge_cubic_monolithic_p.json"

    # Boundary
    # path = "20220407-212155_threshold_cubic_r_r_r_cubic_phenom_p.json"

    # Distributed
    # path = "20220410-000201_threshold_double_edge_cubic_s_s_s_double_edge_ghz_p_p_i.json"

    # Bug fix inconsistent results
    # Supercomputer
    # path = "20220409-221320_threshold_double_edge_cubic_s_s_s_double_edge_ghz_p_p_i.json"  # Fix hopefully yes

    # My PC
    # path = "20220409-221308_threshold_double_edge_cubic_s_s_s_double_edge_ghz_p_p_i.json"  # Fix hopefully yes

    # The last of us
    # path = "20220411-153142_threshold_cubic_s_s_s_cubic_six_ring_single_p_p_i.json"
    # path = "20220411-153549_threshold_cubic_s_s_s_cubic_six_ring_single_p_p_i.json"
    # path = "20220415-022240_threshold_cubic_r_r_r_cbcq_monolithic_p.json"
    # path = "20220418-230049_threshold_cubic_s_s_s_cbcq_ghz_p_p_i.json"  # p
    # path = "20220418-231200_threshold_cubic_s_s_s_cbcq_ghz_p_p_i.json"  # p_i
    # path = "20220428-163649_threshold_cubic_r_r_r_cubic_phenom_p_boundary.json"
    # path = "20220426-225941_threshold_cubic_r_r_r_cubic_monolithic_p0213.json"
    # path = "all_results_Yves/results/20220409-234726_threshold_cubic_s_s_s_cubic_ghz_fusion_p_p_i.json"
    path = "manticore/experiments/20231207-121346_threshold_cubic_r_r_r_cubic_monolithic_p.json"

    _, mean, conf = next(logical_error_rates(path, threshold_param="s", size_param="t"))

    print(mean.to_string())
    print(conf.to_string())
    # exit(0)

    main(mean, conf)
