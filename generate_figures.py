import numpy as np
from manticore.experiments.fit_threshold import fit_threshold, logical_error_rates, residual

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec


def plot_subthreshold_scaling_boundaries(paths_no_boundary=None, paths_boundary=None):
    if paths_no_boundary is None:
        paths_no_boundary = [   # No boundary
            "results/subthreshold_scaling/20231209-224950_threshold_cubic_r_r_r_cubic_phenom_p_erasure_q.json",
            "results/subthreshold_scaling/20231209-231134_threshold_diamond_r_r_r_diamond_phenom_p_erasure_q.json",
            "results/subthreshold_scaling/20231210-103552_threshold_double_edge_cubic_r_r_r_double_edge_cubic_phenom_p_erasure_q.json",
            "results/subthreshold_scaling/20231210-101302_threshold_triamond_r_r_r_triamond_phenom_p_erasure_q.json"
        ]
    if paths_boundary is None:
        paths_boundary = [  # Boundary
            "results/subthreshold_scaling/20231209-223425_threshold_cubic_r_r_r_cubic_phenom_p_erasure_q_boundary.json",
            "results/subthreshold_scaling/20231209-232936_threshold_diamond_r_r_r_diamond_phenom_p_erasure_q_boundary.json",
            "results/subthreshold_scaling/20231210-111054_threshold_double_edge_cubic_r_r_r_double_edge_cubic_phenom_p_erasure_q_boundary.json",
            "results/subthreshold_scaling/20231210-105237_threshold_triamond_r_r_r_triamond_phenom_p_erasure_q_boundary.json"
        ]

    cmap = plt.get_cmap("tab10")
    plt.style.use(['science', 'nature'])

    fig, axes = plt.subplots(2, 4, figsize=(7, 4), dpi=120, facecolor="w")
    axes = axes.flatten()

    markers = ("o", "D")
    names = ("p", "p_e")
    linestyles = (":", "-")

    error_probs_per_subplot = 8 * [None]

    for i, (path1, path2) in enumerate(zip(paths_no_boundary, paths_boundary)):
        for path, marker, ls in zip((path1, path2), markers, linestyles):
            for params, mean, conf in logical_error_rates(path, threshold_param="t"):
                c_p, c_q = params["c_p"], params["c_q"]
                if c_p != 1 and c_q != 1:
                    continue

                if c_p == 1:
                    j = i
                elif c_q == 1:
                    j = i + 4

                zorder = 2 if marker == "o" else 1

                sizes = mean.columns
                error_probs = mean.index[:4]
                error_probs_per_subplot[j] = error_probs

                ax = axes[j]
                for k, prob in enumerate(error_probs):
                    col = cmap(k)
                    ax.errorbar(sizes, mean.loc[prob], yerr=conf.loc[prob], capsize=2, c=col, ls=ls, marker=marker, ms=3,
                                mfc='w', ecolor=col, mec=col, zorder=zorder)

    lattice_names = ["cubic", "diamond", "d.e. cubic", "triamond"]

    for i, ax in enumerate(axes):
        ax.set_xlim(4.4, 11.5)
        ax.set_xticks(sizes)

        if i < 4:
            ylims = (0, 0.3)
            ax.set_ylim(*ylims)
            yticks = np.linspace(0, 0.25, 6)

            ax.set_xticklabels([])
            ax.set_title(lattice_names[i], fontweight="bold")
        else:
            ylims = (0, 0.5)
            ax.set_ylim(*ylims)
            yticks = np.linspace(0, 0.4, 5)
            ax.set_xlabel("Lattice size $L$")

        ax.set_yticks(yticks)
        ax.set_yticklabels((f"{100 * p:.0f}" for p in yticks))

        if i % 4 == 0:
            ax.set_ylabel("Logical error probability (\%)")
        else:
            ax.set_yticklabels([])

        error_probs = error_probs_per_subplot[i]

        color_handles = [Line2D([0], [0], color=cmap(i), marker='o', ms=3, mfc='w', ls='') for i in range(4)]
        name = names[0] if i < 4 else names[1]
        ax.legend(handles=color_handles, labels=[f"${name}={100 * prob:.1f}\%$" for prob in error_probs],
                  loc="upper center", bbox_to_anchor=(0.5, 1), handletextpad=0, columnspacing=0, ncol=2, fontsize=6)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig("figures/subthreshold_scaling_boundaries.pdf")


def plot_fault_tolerant_region_phenomenological_vs_erasure(paths_main=None, paths_compare=None, order="auto"):
    if paths_main is None:
        paths_main = [
            "results/ft_region_phenomenological_vs_erasure/20231223-161712_threshold_cubic_r_r_r_cubic_phenom_p_erasure_q.json",
            "results/ft_region_phenomenological_vs_erasure/20231223-163905_threshold_diamond_r_r_r_diamond_phenom_p_erasure_q.json",
            "results/ft_region_phenomenological_vs_erasure/20231223-171549_threshold_double_edge_cubic_r_r_r_double_edge_cubic_phenom_p_erasure_q.json",
            "results/ft_region_phenomenological_vs_erasure/20231223-182703_threshold_triamond_r_r_r_triamond_phenom_p_erasure_q.json",
        ]

    M = len(paths_main)
    N = 9

    X = np.zeros((M, N))
    X_errs = np.zeros((M, N))
    P_coeffs = np.zeros((M, N))
    Q_coeffs = np.zeros((M, N))

    for i, path in enumerate(paths_main):
        for j, (params, mean, conf) in enumerate(logical_error_rates(path, threshold_param="t")):
            x_th, x_err = fit_threshold(mean, conf, order=order)
            X[i, j] = x_th
            X_errs[i, j] = x_err
            P_coeffs[i, j] = params["c_p"]
            Q_coeffs[i, j] = params["c_q"]

    P = P_coeffs * X
    Q = Q_coeffs * X
    z = 1.96
    P_errs = z * P_coeffs * X_errs
    Q_errs = z * Q_coeffs * X_errs

    if paths_compare is None:
        paths_compare = [
            "results/ft_region_phenomenological_vs_erasure/20231223-162910_threshold_cubic_r_r_r_cubic_phenom_p_erasure_q_boundary.json",
            "results/ft_region_phenomenological_vs_erasure/20231223-165827_threshold_diamond_r_r_r_diamond_phenom_p_erasure_q_boundary.json",
            "results/ft_region_phenomenological_vs_erasure/20231223-175241_threshold_double_edge_cubic_r_r_r_double_edge_cubic_phenom_p_erasure_q_boundary.json",
            "results/ft_region_phenomenological_vs_erasure/20231223-190937_threshold_triamond_r_r_r_triamond_phenom_p_erasure_q_boundary.json",
        ]

    M = len(paths_compare)
    N = 9

    X2 = np.zeros((M, N))
    X2_errs = np.zeros((M, N))
    P2_coeffs = np.zeros((M, N))
    Q2_coeffs = np.zeros((M, N))

    for i, path in enumerate(paths_compare):
        for j, (params, mean, conf) in enumerate(logical_error_rates(path, threshold_param="t")):
            x_th, x_err = fit_threshold(mean, conf, order=order)

            X2[i, j] = x_th
            X2_errs[i, j] = x_err
            P2_coeffs[i, j] = params["c_p"]
            Q2_coeffs[i, j] = params["c_q"]

    P2 = P2_coeffs * X2
    Q2 = Q2_coeffs * X2
    z = 1.96
    P2_errs = z * P2_coeffs * X2_errs
    Q2_errs = z * Q2_coeffs * X2_errs

    names = ["cubic", "diamond", "d.e. cubic", "triamond"]
    valencies = np.array([4, 6, 8, 10], dtype=int)

    phenom_thresholds = P[:, -1]
    erasure_thresholds = Q[:, 0]
    phenom_errs = P_errs[:, -1]
    erasure_errs = Q_errs[:, 0]

    phenom_thresholds2 = P2[:, -1]
    erasure_thresholds2 = Q2[:, 0]
    phenom_errs2 = P2_errs[:, -1]
    erasure_errs2 = Q2_errs[:, 0]

    phenom_diff = phenom_thresholds2 - phenom_thresholds
    erasure_diff = erasure_thresholds2 - erasure_thresholds
    phenom_diff_err = np.sqrt(phenom_errs2**2 + phenom_errs**2)
    erasure_diff_err = np.sqrt(erasure_errs2**2 + erasure_errs**2)

    # for i, n in enumerate(names):
    #     print(f"{n} phenom. threshold at: p={100*phenom_thresholds[i]:.2f} \pm {100*phenom_errs[i]:.2f}\% (95% CI)")
    #     print(f"{n} phenom. threshold 2 at: p={100*phenom_thresholds2[i]:.2f} \pm {100*phenom_errs2[i]:.2f}\% (95% CI)")
    #     print(f"{n} erasure threshold at: p_e={100*erasure_thresholds[i]:.2f} \pm {100*erasure_errs[i]:.2f}\% (95% CI)")
    #     print(f"{n} erasure threshold 2 at: p_e={100*erasure_thresholds2[i]:.2f} \pm {100*erasure_errs2[i]:.2f}\% (95% CI)")
    #     print(f"{n} phenom. diff: p={100*phenom_diff[i]:.2f} \pm {100*phenom_diff_err[i]:.2f}\% (95% CI)")
    #     print(f"{n} erasure. diff: p={100*erasure_diff[i]:.2f} \pm {100*erasure_diff_err[i]:.2f}\% (95% CI)")

    alphabet = "abcdefghijkl"

    plt.style.use(['science', 'nature'])
    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(7, 3.2), dpi=120, facecolor="w")
    gs = GridSpec(2, 3, figure=fig, width_ratios=(2, 1, 1), wspace=0.3, hspace=0.25)

    ax = fig.add_subplot(gs[:, 0])

    for i, (p, q, p_err, q_err, name) in enumerate(zip(P, Q, P_errs, Q_errs, names)):
        col = cmap(i)
        # ax.plot(p, q, c=col, ls=(0, (2, 2)), marker='o', mfc='w', label=name)
        ax.errorbar(p, q, xerr=p_err, yerr=q_err, c=col, ls=(0, (2, 2)), marker='o', mfc='w', label=name)

        if i > 0:
            p_prev = P[i-1]
            q_prev = Q[i-1]
        else:
            p_prev = [0]
            q_prev = [0]

        xfill = np.sort(np.concatenate([p_prev, p]))
        y1fill = np.interp(xfill, p_prev, q_prev)
        y2fill = np.interp(xfill, p, q)
        ax.fill_between(xfill, y1fill, y2fill, color=col, alpha=0.3)

    xticks = np.linspace(0, 0.1, 11)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xticks(xticks)
    ax.set_xticklabels((f"{100*x:.0f}" for x in xticks))
    ax.set_xlabel('Phenom. error probability $p_\mathrm{m}$ (\%)')

    yticks = np.linspace(0, 0.6, 7)
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_yticks(yticks)
    ax.set_yticklabels((f"{100*y:.0f}" for y in yticks))
    ax.set_ylabel('Erasure error probability $p_\mathrm{e}$ (\%)')

    ax.legend(loc='upper right', title='Lattice')

    # Second plots
    linestyle = (0, (2, 2))
    xlims = (3.5, 10.5)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.errorbar(valencies, phenom_thresholds, yerr=phenom_errs, c='k', ls='-.', marker='o', mfc='w', label="periodic")
    ax1.errorbar(valencies, phenom_thresholds2, yerr=phenom_errs2, c='dimgray', ls=linestyle, marker='D', mfc='w', label="boundary")
    yticks = np.linspace(0.02, 0.1, 5)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels((f"{100*y:.0f}" for y in yticks))
    ax1.set_ylim(0.01, 0.11)
    ax1.set_xticks(valencies)
    ax1.set_xlim(*xlims)
    ax1.set_ylabel('Phenom. threshold (\%)')
    leg = ax1.legend(loc='lower right', markerfirst=False)
    # leg._legend_box.align = "right"

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.errorbar(valencies, phenom_thresholds/valencies, yerr=phenom_errs/valencies, c='k', ls='-.', marker='o', mfc='w', label="periodic")
    ax2.errorbar(valencies, phenom_thresholds2/valencies, yerr=phenom_errs2/valencies, c='dimgray', ls=linestyle, marker='D', mfc='w', label="boundary")
    yticks = np.linspace(0.006, 0.01, 5)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels((f"{100*y:.1f}" for y in yticks))
    ax2.set_ylim(0.006, 0.0105)
    ax2.set_xticks(valencies)
    ax2.set_xlim(*xlims)
    leg = ax2.legend(loc='lower right', markerfirst=False)
    # leg._legend_box.align = "right"

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.errorbar(valencies, erasure_thresholds, yerr=erasure_errs, c='k', ls='-.', marker='o', mfc='w', label="periodic")
    ax3.errorbar(valencies, erasure_thresholds2, yerr=erasure_errs2, c='dimgray', ls=linestyle, marker='D', mfc='w', label="boundary")
    yticks = np.linspace(0.2, 0.6, 5)
    ax3.set_yticks(yticks)
    ax3.set_yticklabels((f"{100*y:.0f}" for y in yticks))
    ax3.set_ylim(0.2, 0.6)
    ax3.set_xticks(valencies)
    ax3.set_xlim(*xlims)
    ax3.set_xlabel('Cluster state valency $z$')
    ax3.set_ylabel('Erasure threshold (\%)')
    leg = ax3.legend(loc='lower right', markerfirst=False)
    # leg._legend_box.align = "right"

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.errorbar(valencies, erasure_thresholds/valencies, yerr=erasure_errs/valencies, c='k', ls='-.', marker='o', mfc='w', label="periodic")
    ax4.errorbar(valencies, erasure_thresholds2/valencies, yerr=erasure_errs2/valencies, c='dimgray', ls=linestyle, marker='D', mfc='w', label="boundary")
    yticks = np.linspace(0.054, 0.066, 5)
    ax4.set_yticks(yticks)
    ax4.set_yticklabels((f"{100*y:.1f}" for y in yticks))
    ax4.set_ylim(0.054, 0.066)
    ax4.set_xticks(valencies)
    ax4.set_xlim(*xlims)
    ax4.set_xlabel('Cluster state valency $z$')
    leg = ax4.legend(loc='lower left', markerfirst=False)
    # leg._legend_box.align = "left"

    for i, ax in enumerate((ax, ax1, ax2, ax3, ax4)):
        label_offsetx = 0.1 if i == 0 else 0.2
        ax.text(0.0 - label_offsetx, 1.0, alphabet[i], va="bottom", ha="right", transform=ax.transAxes, fontweight="bold", fontsize=12)

    fig.savefig("figures/ft_region_phenomenological_vs_erasure.pdf")
    # plot_simple_thresholds(valencies, erasure_thresholds, phenom_thresholds, names)
    return paths_main, paths_compare, names, \
           phenom_thresholds*100, phenom_errs*100, \
           erasure_thresholds*100, erasure_errs*100, \
           phenom_diff*100, phenom_diff_err*100, \
           erasure_diff*100, erasure_diff_err*100


def plot_simple_thresholds(valencies, erasure_thresholds, phenom_thresholds, names):
    fig, ax = plt.subplots(figsize=(2.5, 2), dpi=240, facecolor="w")

    cmap = plt.get_cmap("tab10")
    ax.plot(valencies, erasure_thresholds, c=cmap(1), ls='-.', marker='o', mfc='w', label="erasure")
    ax.plot(valencies, phenom_thresholds, c=cmap(0), ls='-.', marker='o', mfc='w', label="bit-flip")

    ax.legend(loc="center right")
    ax.set_xlim(3, 11)

    ax.set_xticks(valencies)
    ax.set_xticklabels(names)

    yticks = np.linspace(0, 0.6, 7)
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_yticks(yticks)
    ax.set_yticklabels((f"{100*y:.0f}" for y in yticks))
    ax.set_ylabel(f"Threshold value (\%)")

    ax.minorticks_off()
    fig.savefig("figures/simple_thresholds_phenomenological_vs_erasure.pdf")


def plot_thresholds_phenomenological_vs_erasure(path=None, order="auto", threshold_param="t",
                                                size_param="r", phys_params=("c_p", "c_q")):
    if path is None:
        path = "results/ft_region_phenomenological_vs_erasure/20231222-173203_threshold_diamond_r_r_r_diamond_phenom_p_erasure_q.json"
    fit_type = 2

    x_thresholds = []
    x_errs = []

    p_coeffs = []
    q_coeffs = []

    cmap = plt.get_cmap("tab10")
    plt.style.use(['science', 'nature'])

    fig, axes = plt.subplots(3, 3, figsize=(7, 5), dpi=120, facecolor="w")
    axes = axes.flatten()

    for i, (params, mean, conf) in enumerate(logical_error_rates(path, threshold_param=threshold_param, size_param=size_param)):
        error_probs = mean.index
        lattice_sizes = mean.columns
        xlim_offset = 0.03 * (error_probs[-1] - error_probs[0])
        xlims = (error_probs[0] - xlim_offset, error_probs[-1] + xlim_offset)

        N = 1000
        x = np.linspace(*xlims, N)

        fit_result = fit_threshold(mean, conf, order=order, return_fit=True, fit_type=fit_type)
        Y = residual(fit_result.params, (x, lattice_sizes), fit_type=fit_type)

        x_th = fit_result.params["x_th"].value
        x_err = fit_result.params["x_th"].stderr
        # print(f"Threshold at {params}: {x_th} +- {1.96 * x_err} (95% CI)")

        x_thresholds.append(x_th)
        x_errs.append(x_err)
        c_p, c_q = params[phys_params[0]], params[phys_params[1]]
        p_coeffs.append(c_p)
        q_coeffs.append(c_q)

        ax = axes[i]
        for j, size in enumerate(mean):
            col = cmap(j)
            ax.errorbar(error_probs, mean[size], yerr=conf[size], capsize=2, c=col, ls='', lw=1, marker='o', ms=3,
                        mfc='w',
                        ecolor=col, mec=col, label=f"$L={size}$")

            ax.plot(x, Y[:, j], color=col)

        if order == "auto":
            ax.text(0.95, 0.05, f"$k={fit_result.params['order'].value}$", va="bottom", ha="right", transform=ax.transAxes)
        ax.plot((x_th, x_th), (0, np.max(residual(fit_result.params, ([x_th], lattice_sizes), fit_type=fit_type))),
                c='k', ls='-')

        x_th_conf = np.array([x_th - 1.96 * x_err, x_th + 1.96 * x_err])
        ax.fill_between(x_th_conf, (0, 0), np.max(residual(fit_result.params, (x_th_conf, lattice_sizes),
                                                           fit_type=fit_type), axis=1), color='k', alpha=0.2)

        ax.set_xlim(*xlims)

        ymaxval = int(np.ceil(np.max(mean.values) * 10))
        ylims = (0, ymaxval / 10)
        yticks = np.linspace(*ylims, ymaxval + 1)
        ax.set_ylim(*ylims)
        ax.set_yticks(yticks)
        ax.set_yticklabels((f"{100 * p:.0f}" for p in yticks))

        ax.legend(loc='upper left')
        if i != 0:
            ax.set_xticks(error_probs)
            ax.set_xticklabels((f"{100 * p:.2f}" for p in c_p * error_probs))
        else:
            ax.set_xticks([])

        ax.grid(c='lightgray', ls=':')

        ax2 = ax.twiny()
        ax2.set_xlim(*ax.get_xlim())
        if i != 8:
            ax2.set_xticks(error_probs)
            ax2.set_xticklabels((f"{100 * q:.2f}" for q in c_q * error_probs))
        else:
            ax2.set_xticks([])

        ax2.grid(c='lightgray', ls=':')

        if i < 3:
            if phys_params[1] == "c_q":
                ax2.set_xlabel('Erasure error probability (\%)')
            elif phys_params[1] == "c_p":
                ax2.set_xlabel('Gate error probability (\%)')
        elif i > 5:
            if phys_params[0] == "c_p" and phys_params[1] == "c_q":
                ax.set_xlabel('Phenomenological error probability (\%)')
            elif phys_params[0] == "c_p_i" and phys_params[1] == "c_p":
                ax.set_xlabel('Network error probability (\%)')
        if i % 3 == 0:
            ax.set_ylabel('Logical error probability (\%)')

        ax.tick_params(labelsize=6)
        ax2.tick_params(labelsize=6)

    fig.subplots_adjust(hspace=0.25)
    fig.savefig("figures/thresholds_phenomenological_vs_erasure.pdf")


def plot_monolithic_thresholds(paths=None, names=None, orderings=None, order="auto"):
    if paths is None:
        paths = [
            "results/monolithic/20231224-153252_threshold_cubic_r_r_r_cubic_monolithic_p.json",
            "results/monolithic/20231224-153345_threshold_cubic_r_r_r_cubic_monolithic_p0213.json",
            "results/monolithic/20231224-154318_threshold_double_edge_cubic_r_r_r_double_edge_cubic_monolithic_p.json",
            "results/monolithic/20231224-153836_threshold_diamond_r_r_r_diamond_monolithic_p.json",
            "results/monolithic/20231224-153956_threshold_diamond_r_r_r_diamond_monolithic_p024135.json",
            "results/monolithic/20231224-153446_threshold_triamond_r_r_r_triamond_monolithic_p.json",
            "results/monolithic/20231224-154136_threshold_diamond_r_r_r_diamond_monolithic_p031425.json",
            "results/monolithic/20231224-154529_threshold_double_edge_cubic_r_r_r_double_edge_cubic_monolithic_p02461357.json",
            "results/monolithic/20231224-154815_threshold_double_edge_cubic_r_r_r_double_edge_cubic_monolithic_p04152637.json"
        ]

    if names is None:
        names = [
            "cubic",
            "cubic",
            "d.e. cubic",
            "diamond",
            "diamond",
            "triamond",
            "diamond",
            "d.e. cubic",
            "d.e. cubic"
        ]

    if orderings is None:
        orderings = [
            "",
            "0231",
            "",
            "",
            "024135",
            "",
            "031425",
            "02461357",
            "04152637"
        ]

    cmap = plt.get_cmap("tab10")
    plt.style.use(['science', 'nature'])

    fig, axes = plt.subplots(3, 3, figsize=(7, 5), dpi=120, facecolor="w")
    axes = axes.flatten()

    monolithic_thresholds = []
    monolithic_errors = []

    for i, (path, name, ordering) in enumerate(zip(paths, names, orderings)):
        # _, mean, conf = next(logical_error_rates(path, threshold_param="p", size_param="r", error_when=("dual_logical_x",)))
        # _, mean, conf, num_shots = next(logical_error_rates(path, threshold_param="p", size_param="r",
        #                                                     output_num_shots=True))  # , error_when=('primal_logical_x', 'primal_logical_y', 'dual_logical_x', 'dual_logical_y')))
        _, mean, conf = next(logical_error_rates(path, threshold_param="p", size_param="r"))

        error_probs = mean.index
        lattice_sizes = mean.columns
        xlim_offset = 0.03 * (error_probs[-1] - error_probs[0])
        xlims = (error_probs[0] - xlim_offset, error_probs[-1] + xlim_offset)

        N = 1000
        x = np.linspace(*xlims, N)

        ax = axes[i]
        try:
            fit_result = fit_threshold(mean, conf, order=order, return_fit=True, seed=19)
            Y = residual(fit_result.params, (x, lattice_sizes))
        except RuntimeError:
            for j, size in enumerate(mean):
                col = cmap(j)
                ax.errorbar(error_probs, mean[size], yerr=conf[size], capsize=2, c=col, ls=':', lw=1, marker='o',
                            ms=3,
                            mfc='w', ecolor=col, mec=col, label=f"$L={size}$")

            continue

        x_th = fit_result.params["x_th"].value
        x_err = fit_result.params["x_th"].stderr
        # print(name, ordering, f"Threshold at: {x_th} +- {1.96 * x_err} (95% CI)")
        monolithic_thresholds.append(x_th)
        monolithic_errors.append(x_err)

        for j, size in enumerate(mean):
            col = cmap(j)
            ax.errorbar(error_probs, mean[size], yerr=conf[size], capsize=2, c=col, ls='', lw=1, marker='o', ms=3,
                        mfc='w',
                        ecolor=col, mec=col, label=f"$L={size}$")
            ax.plot(x, Y[:, j], color=col)

        ax.text(0.8, 0.95, f"{name}\n{ordering}", va="top", ha="right", fontweight="bold", fontsize=8, transform=ax.transAxes)
        if order == "auto":
            ax.text(0.95, 0.05, f"$k={fit_result.params['order'].value}$", va="bottom", ha="right", transform=ax.transAxes)
        ax.plot((x_th, x_th), (0, np.max(residual(fit_result.params, ([x_th], lattice_sizes)))), c='k', ls='-')
        x_th_conf = np.array([x_th - 1.96 * x_err, x_th + 1.96 * x_err])
        ax.fill_between(x_th_conf, (0, 0),
                        np.max(residual(fit_result.params, (x_th_conf, lattice_sizes)), axis=1),
                        color='k', alpha=0.2)

        ax.set_xlim(*xlims)
        ymaxval = int(np.ceil(np.max(mean.values) * 10))
        ylims = (0, ymaxval / 10)
        yticks = np.linspace(*ylims, ymaxval + 1)
        ax.set_ylim(*ylims)
        ax.set_yticks(yticks)
        ax.set_yticklabels((f"{100 * p:.0f}" for p in yticks))

        ax.legend(loc='upper left')
        ax.set_xticks(error_probs)
        ax.set_xticklabels((f"{100 * p:.2f}" for p in error_probs))
        ax.grid(c='lightgray', ls=':')

        if i > 5:
            ax.set_xlabel('Error probability $p_\mathrm{o}$ (\%)')
        if i % 3 == 0:
            ax.set_ylabel('Logical error probability (\%)')

        ax.tick_params(labelsize=6)

    fig.subplots_adjust(hspace=0.25)
    fig.savefig("figures/monolithic_thresholds.pdf")

    return np.array([monolithic_thresholds[1], monolithic_thresholds[6], monolithic_thresholds[8], monolithic_thresholds[5]])*100, \
           np.array([monolithic_errors[1], monolithic_errors[6], monolithic_errors[8], monolithic_errors[5]])*196


def plot_fault_tolerant_region_distributed(paths=None, names=None, order="auto"):
    if paths is None:
        paths = [
            "results/ft_region_distributed/20231219-203526_threshold_cubic_s_s_s_cubic_ghz_fusion_p_p_i.json",
            "results/ft_region_distributed/20231219-202833_threshold_cubic_s_s_s_cubic_six_ring_p_p_i.json",
            "results/ft_region_distributed/20231219-204223_threshold_diamond_s_s_s_three_diamond_fusion_p_p_i.json",
            "results/ft_region_distributed/20231219-205251_threshold_diamond_s_s_s_23_diamond_fusion_p_p_i.json",
            "results/ft_region_distributed/20231219-210354_threshold_double_edge_cubic_s_s_s_double_edge_bell_p_p_i.json",
            "results/ft_region_distributed/20231219-212042_threshold_double_edge_cubic_s_s_s_double_edge_ghz_p_p_i.json",
        ]
    if names is None:
        names = [
            "2-node",
            "6-ring",
            "4-ring",
            "7-node",
            "12-node",
            "4-ring"
        ]

    M = len(paths)
    N = 9

    X = np.zeros((M, N))
    X_errs = np.zeros((M, N))
    P_coeffs = np.zeros((M, N))
    P_i_coeffs = np.zeros((M, N))

    for i, path in enumerate(paths):
        for j, (params, mean, conf) in enumerate(logical_error_rates(path, threshold_param="t", size_param="s")):
            x_th, x_err = fit_threshold(mean, conf, order=order, seed=420)

            X[i, j] = x_th
            X_errs[i, j] = x_err
            P_coeffs[i, j] = params["c_p"]
            P_i_coeffs[i, j] = params["c_p_i"]

    P = P_coeffs * X
    P_i = P_i_coeffs * X
    z = 1.96
    P_errs = z * P_coeffs * X_errs
    P_i_errs = z * P_i_coeffs * X_errs

    P = np.flip(P, axis=1)
    P_i = np.flip(P_i, axis=1)
    P_errs = np.flip(P_errs, axis=1)
    P_i_errs = np.flip(P_i_errs, axis=1)

    phenom_thresholds = P[:, -1]
    pi_thresholds = P_i[:, 0]
    phenom_errs = P_errs[:, -1]
    pi_errs = P_i_errs[:, 0]

    # for i, n in enumerate(names):
    #     print(
    #         f"{n} phenom. threshold at: p_{{th}}={100 * phenom_thresholds[i]:.3f} \pm {100 * phenom_errs[i]:.3f}\% (95% CI)")
    #     print(f"{n} pi threshold at: p_{{i,th}}={100 * pi_thresholds[i]:.3f} \pm {100 * pi_errs[i]:.3f}\% (95% CI)")

    alphabet = "abcdefghijkl"

    plt.style.use(['science', 'nature'])
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), dpi=240, facecolor="w")

    xticks_ax = [
        np.linspace(0, 0.005, 6),
        np.linspace(0, 0.006, 7),
        np.linspace(0, 0.005, 6)
    ]

    yticks_ax = [
        np.linspace(0, 0.012, 7),
        np.linspace(0, 0.025, 6),
        np.linspace(0, 0.015, 6)
    ]

    ax_titles = [
        "cubic",
        "diamond",
        "d.e. cubic"
    ]

    for k, (ax, xticks, yticks, title) in enumerate(zip(axes, xticks_ax, yticks_ax, ax_titles)):
        for i in range(2 * k, 2 * k + 2):
            p = P[i]
            p_i = P_i[i]
            p_err = P_errs[i]
            p_i_err = P_i_errs[i]
            name = names[i]

            col = cmap(i)
            ax.errorbar(p, p_i, xerr=p_err, yerr=p_i_err, c=col, ls=(0, (2, 2)), marker='o', mfc='w', label=name)

            if i % 2 == 1:
                p_prev = P[i - 1]
                p_i_prev = P_i[i - 1]
            else:
                p_prev = [0]
                p_i_prev = [0]

            xfill = np.sort(np.concatenate([p_prev, p]))
            y1fill = np.interp(xfill, p_prev, p_i_prev)
            y2fill = np.interp(xfill, p, p_i)
            ax.fill_between(xfill, y1fill, y2fill, color=col, alpha=0.3, where=y2fill > y1fill, interpolate=True)

        ax.set_xlim(xticks[0], xticks[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels((f"{100 * x:.1f}" for x in xticks))
        ax.set_xlabel('Error probability $p_\mathrm{o}$ (\%)')

        ax.set_ylim(yticks[0], yticks[-1])
        ax.set_yticks(yticks)
        ax.set_yticklabels((f"{100 * y:.1f}" for y in yticks))
        ax.set_ylabel('Network error probability $p_\mathrm{n}$ (\%)')

        ax.legend(loc='upper right', title='Architecture')
        ax.set_title(title, fontweight="bold")

    fig.subplots_adjust(wspace=0.3)
    fig.savefig("figures/ft_region_distributed.pdf")
    return paths, pi_thresholds*100, pi_errs*100, \
           phenom_thresholds*100, phenom_errs*100


def interpolate_protocols(failprob, infideli, intermediate_points=100):
    infideli_tot = []
    failprob_tot = []
    for i_i in range(len(infideli) - 1):
        p1 = 1 - failprob[i_i + 1]
        p2 = 1 - failprob[i_i]
        F1 = 1 - infideli[i_i + 1]
        F2 = 1 - infideli[i_i]
        for rr in range(intermediate_points + 1):
            r = rr / intermediate_points
            failprob_tot.append(1 - (r * p1 + (1 - r) * p2))
            infideli_tot.append(1 - ((1 / (1 - failprob_tot[-1])) * (r * p1 * F1 + (1 - r) * p2 * F2)))
    return failprob_tot, infideli_tot


def plot_trade_off_network_error_erasure(path=None, order="auto"):
    # Erasure
    if path is None:
        path = "results/trade_off_network_erasure/20231219-003033_threshold_cubic_s_s_s_cubic_six_ring_single_p_p_i.json"
    N = 9

    X = np.zeros(N)
    X_errs = np.zeros(N)
    P_coeffs = np.zeros(N)
    P_i_coeffs = np.zeros(N)

    for i, (params, mean, conf) in enumerate(logical_error_rates(path, threshold_param="t", size_param="s")):
        # if i == 0:
        #     print(mean)
        #     print(params)
        #     exit(0)
        x_th, x_err = fit_threshold(mean, conf, order=order, seed=420)

        X[i] = x_th
        X_errs[i] = x_err
        P_coeffs[i] = params["c_p"]
        P_i_coeffs[i] = params["c_p_i"]

    P = P_coeffs * X
    P_i = P_i_coeffs * X
    z = 1.96
    P_errs = z * P_coeffs * X_errs
    P_i_errs = z * P_i_coeffs * X_errs

    P = np.flip(P)
    P_i = np.flip(P_i)
    P_errs = np.flip(P_errs)
    P_i_errs = np.flip(P_i_errs)

    plt.style.use(['science', 'nature'])
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(3, 3), dpi=240, facecolor="w")

    # DEJMPS and 5-to-1 distillation data
    data_cDEJMPS = {0.947: {'failprob': [0, 0.06816977777777788, 0.13277066903703727, 0.19289153682765459, 0.23616780034152018, 0.27735675322321296, 0.31560486573599367], 'infideli': [0.05300000000000005, 0.03724832086722529, 0.020345957159016326, 0.0028271113757869504, 0.0021620398449551104, 0.0008527269770218782, 0.0002334867832874954]}, 0.969: {'failprob': [0, 0.04047911111111113, 0.07971219674074093, 0.1173409372424693, 0.1448913436638437, 0.17167204875361408, 0.19733870159846523], 'infideli': [0.031000000000000028, 0.021315962318231074, 0.011223575728452784, 0.000918504012782706, 0.0006964555413510043, 0.00025665122278140107, 4.364337146511854e-05]}}
    data_5to1 = {0.947: {0: {'failprob': [0.0, 0.01588065999999999, 0.03176133000000003, 0.04764199000000002, 0.06352265999999995, 0.07940332000000005, 0.09528398999999999, 0.11116464999999998, 0.12704532000000002, 0.14292598, 0.15880665000000005, 0.17468731000000004, 0.19056797999999997, 0.20644863999999996, 0.22232931, 0.23820997], 'infideli': [0.024919910000000045, 0.02364396000000002, 0.022326150000000045, 0.020964390000000055, 0.019556449999999947, 0.018099929999999986, 0.016592280000000015, 0.015030749999999982, 0.013412410000000041, 0.01173409999999997, 0.009992420000000002, 0.008183709999999955, 0.006304030000000016, 0.004349119999999984, 0.00231437000000001, 0.00019477999999995]}, 1: {'failprob': [0.23820997], 'infideli': [0.00019477999999995]}}, 0.969: {0: {'failprob': [0.0, 0.009710140000000034, 0.019420279999999956, 0.02913041000000005, 0.038840549999999974, 0.04855069000000001, 0.05826083000000004, 0.06797096999999996, 0.07768111, 0.08739123999999998, 0.09710138000000001, 0.10681152000000005, 0.11652165999999997, 0.1262318, 0.13594193, 0.14565207000000002], 'infideli': [0.008964259999999946, 0.008450770000000052, 0.007927119999999954, 0.007392990000000044, 0.006848060000000045, 0.006292019999999954, 0.005724500000000021, 0.005145170000000032, 0.004553629999999975, 0.003949509999999989, 0.003332389999999963, 0.0027018500000000056, 0.0020574599999999554, 0.001398739999999954, 0.0007252199999999709, 3.637999999994701e-05]}, 1: {'failprob': [0.14565207000000002], 'infideli': [3.637999999994701e-05]}}}

    for i_f, fid in enumerate([0.947, 0.969]):
        col = cmap(i_f)

        infideli = data_cDEJMPS[fid]['infideli']
        failprob = data_cDEJMPS[fid]['failprob']
        failprob_tot, infideli_tot = interpolate_protocols(failprob, infideli, 10)
        ax.plot(failprob_tot, infideli_tot, c=col, marker=None, ls=(0, (1, 1)))
        ax.plot(failprob, infideli, c=col, marker='v', mfc='w', ls='none')
        ax.plot([], [], c=col, marker='v', mfc='w', ls=(0, (1, 1)), label=r'cDEJMPS, $F_\mathrm{n}' + f'={fid}$')
        k_range = [1, 2, 3, 4, 5, 6, 7] if fid == 0.969 else [3, 4, 5, 6]
        for k in k_range:
            offset_x = -0.011 if k == 3 or (k in [2, 4] and fid == 0.969) else (0.005 if k == 1 else 0)
            offset_y = 0.0005 if (k in [5, 6, 7]) or (k == 4 and fid == 0.947) else -0.0006
            ax.text(failprob[k - 1] + offset_x, infideli[k - 1] + offset_y, str(k), c=col)
        asp = 2
        x_middle = (failprob[asp] + failprob[asp + 1]) / 2
        y_middle = (infideli[asp] + infideli[asp + 1]) / 2 + 0.003
        slope = (infideli[asp + 1] - infideli[asp]) / (failprob[asp + 1] - failprob[asp])
        x_base = x_middle - 0.01
        x_tip = x_middle + 0.01
        y_base = y_middle - slope * 0.01
        y_tip = y_middle + slope * 0.01
        ax.annotate("",
                    xy=(x_tip, y_tip), xycoords='data',
                    xytext=(x_base, y_base), textcoords='data',
                    arrowprops=dict(arrowstyle="-|>",
                                    connectionstyle="arc3", color=col),
                    )

        labels = [r'5-to-1, $F_\mathrm{n}' + f'={fid}$', f'']
        for i_d in range(2):
            failprob = data_5to1[fid][i_d]['failprob']
            infideli = data_5to1[fid][i_d]['infideli']
            if i_d == 0:
                failprob_tot, infideli_tot = interpolate_protocols(failprob, infideli)
                ax.plot(failprob_tot, infideli_tot, c=col, marker=None, ls=(0, (1, 1)))
                ax.plot([], [], c=col, marker='o', mfc='w', ls=(0, (1, 1)), label=labels[i_d])
            ax.plot(failprob, infideli, c=col, marker="o", mfc='w' if i_d == 0 else col, ls='none')

    ax.errorbar(P, P_i, xerr=P_errs, yerr=P_i_errs, c='k', ls=(0, (2, 2)), marker='o', mfc='w')

    xfill = np.sort(np.concatenate([[0], P]))
    y1fill = np.interp(xfill, [0], [0])
    y2fill = np.interp(xfill, P, P_i)
    ax.fill_between(xfill, y1fill, y2fill, color='k', alpha=0.3)

    ax.legend(loc='upper right')

    xticks = np.linspace(0, 0.3, 6)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xticks(xticks)
    ax.set_xticklabels((f"{100 * x:.1f}" for x in xticks))
    ax.set_xlabel(r'Erasure error probability $p_\mathrm{e}$ (\%)')

    yticks = np.linspace(0, 0.032, 6)
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_yticks(yticks)
    ax.set_yticklabels((f"{100 * y:.1f}" for y in yticks))
    ax.set_ylabel(r'Network error probability $p_\mathrm{n}$ (\%)')
    fig.savefig("figures/trade_off_network_error_erasure.pdf")
    return path


def overview_figure_thresholds(
        names=None,
        phenom_thresholds=None,
        phenom_errors=None,
        erasure_thresholds=None,
        erasure_errors=None,
        phenom_boundary_diffs=None,
        phenom_boundary_errors=None,
        erasure_boundary_diffs=None,
        erasure_boundary_errors=None,
        # phenom_boundary=None,
        # phenom_boundary_errors=None,
        # erasure_boundary=None,
        # erasure_boundary_errors=None,
        monolithic_thresholds=None,
        monolithic_errors=None,
        distributed_names=None,
        network_thresholds=None,
        network_errors=None,
        circuit_thresholds=None,
        circuit_errors=None
):
    names = np.array(["cubic", "diamond", "d.e. cubic", "triamond"]) if names is None else names

    # Phenom
    phenom_thresholds = np.array([2.65, 5.33, 8.05, 9.75]) if phenom_thresholds is None else phenom_thresholds
    phenom_errors = np.array([0.01, 0.02, 0.04, 0.02]) if phenom_errors is None else phenom_errors
    erasure_thresholds = np.array([24.94, 39.09, 49.91, 55.17]) if erasure_thresholds is None else erasure_thresholds
    erasure_errors = np.array([0.06, 0.07, 0.07, 0.02]) if erasure_errors is None else erasure_errors

    # # Boundary
    # phenom_boundary_diffs = np.array([-0.01, -0.06, +0.05, +0.01]) if (phenom_boundary is None and phenom_thresholds is None) else phenom_boundary - phenom_thresholds
    # phenom_boundary_errors = np.array([0.02, 0.05, 0.04, 0.04]) if (phenom_boundary_errors is None and phenom_errors is None) else phenom_boundary_errors + phenom_errors
    # erasure_boundary_diffs = np.array([-0.25, -0.10, -0.24, +0.07]) if (erasure_boundary is None and erasure_thresholds is None) else erasure_boundary - erasure_thresholds
    # erasure_boundary_errors = np.array([0.12, 0.11, 0.13, 0.10]) if (erasure_boundary_errors is None and erasure_errors is None) else erasure_boundary_errors + erasure_errors
    phenom_boundary_diffs = np.array([-0.01, -0.06, +0.05, +0.01]) if (phenom_boundary_diffs is None) else phenom_boundary_diffs
    phenom_boundary_errors = np.array([0.02, 0.05, 0.04, 0.04]) if (phenom_boundary_errors is None) else phenom_boundary_errors
    erasure_boundary_diffs = np.array([-0.25, -0.10, -0.24, +0.07]) if (erasure_boundary_diffs is None) else erasure_boundary_diffs
    erasure_boundary_errors = np.array([0.12, 0.11, 0.13, 0.10]) if (erasure_boundary_errors is None) else erasure_boundary_errors


    # Monolithic
    monolithic_thresholds = np.array([0.52, 0.631, 0.38, 0.352]) if monolithic_thresholds is None else monolithic_thresholds
    monolithic_errors = np.array([0.01, 0.003, 0.01, 0.003]) if monolithic_errors is None else monolithic_errors

    distributed_names = np.array(["2-node", "6-ring", "4-ring", "7-node", "12-node", "4-ring"]) if distributed_names is None else distributed_names
    network_thresholds = np.array([1.01, 0.96, 1.99, 2.23, 0.618, 1.43]) if network_thresholds is None else network_thresholds
    network_errors = np.array([0.01, 0.01, 0.02, 0.01, 0.004, 0.01]) if network_errors is None else network_errors
    circuit_thresholds = np.array([0.309, 0.454, 0.518, 0.551, 0.276, 0.455]) if circuit_thresholds is None else circuit_thresholds
    circuit_errors = np.array([0.002, 0.005, 0.003, 0.005, 0.002, 0.002]) if circuit_errors is None else circuit_errors

    # Don't show this to my boss
    distributed_bad_names = distributed_names[::2]
    distributed_good_names = distributed_names[1::2]

    network_bad_thresholds = network_thresholds[::2]
    network_good_thresholds = network_thresholds[1::2]
    network_bad_errors = network_errors[::2]
    network_good_errors = network_errors[1::2]

    circuit_bad_thresholds = circuit_thresholds[::2]
    circuit_good_thresholds = circuit_thresholds[1::2]
    circuit_bad_errors = circuit_errors[::2]
    circuit_good_errors = circuit_errors[1::2]

    # Distributed
    # 2-node 1.01(1) 0.309(2)
    # 6-ring 0.96(1) 0.454(5)
    # 4-ring 1.99(2) 0.518(3)
    # 7-node 2.23(1) 0.551(5)
    # 12-node 0.618(4) 0.276(2)
    # 4-ring 1.43(1) 0.455(2)

    x = np.arange(4)
    x3 = np.arange(3)
    cmap = plt.get_cmap("tab10")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(7, 3.5), dpi=120, facecolor='w',
                                        gridspec_kw=dict(width_ratios=[5, 5, 9]))

    # --- a ---
    width = 0.4  # the width of the bars

    c = cmap(0)
    bars = ax0.bar(x, phenom_thresholds, width, yerr=phenom_errors, color=c, alpha=0.4, ecolor=c, capsize=3,
                   label="bit-flip")
    labels = [f"{t:.2f}({100 * e:.0f})" for t, e in zip(phenom_thresholds, phenom_errors)]
    ax0.bar_label(bars, labels, rotation=90, padding=5, color=c, weight="bold")

    c = cmap(1)
    bars = ax0.bar(x + width, erasure_thresholds, width, yerr=erasure_errors, color=c, alpha=0.5, ecolor=c, capsize=3,
                   label="erasure")
    labels = [f"{t:.2f}({100 * e:.0f})" for t, e in zip(erasure_thresholds, erasure_errors)]
    ax0.bar_label(bars, labels, rotation=90, padding=5, color=c, weight="bold")

    ax0.set_xticks(x + 0.5 * width)
    ax0.set_xticklabels(names, rotation=45)

    ax0.set_ylim(0, 70)
    ax0.set_ylabel("Error threshold (\%)")

    ax0.legend(loc="upper left")
    ax0.text(-0.17, 1.0, "a", va="center", ha="right", transform=ax0.transAxes, fontweight="bold", fontsize=12)
    ax0.tick_params(axis='x', which='minor', bottom=False)

    # --- b ---

    c = cmap(0)
    bars = ax1.bar(x, phenom_boundary_diffs, width, yerr=phenom_boundary_errors, color=c, alpha=0.4, ecolor=c,
                   capsize=3, label="bit-flip")
    labels = [f"{t:.3f}({1000 * e:.0f})" for t, e in zip(phenom_boundary_diffs, phenom_boundary_errors)]
    ax1.bar_label(bars, labels, rotation=90, padding=5, color=c, weight="bold")

    c = cmap(1)
    bars = ax1.bar(x + width, erasure_boundary_diffs, width, yerr=erasure_boundary_errors, color=c, alpha=0.5, ecolor=c,
                   capsize=3, label="erasure")
    labels = [f"{t:.3f}({1000 * e:.0f})" for t, e in zip(erasure_boundary_diffs, erasure_boundary_errors)]
    ax1.bar_label(bars, labels, rotation=90, padding=5, color=c, weight="bold")

    ax1.set_xticks(x + 0.5 * width)
    ax1.set_xticklabels(names, rotation=45)

    ax1.set_ylim(-0.6, 0.5)
    ax1.set_ylabel('Boundary threshold difference (p.p.)')

    ax1.legend(loc="upper left")
    ax1.text(-0.17, 1.0, "b", va="center", ha="right", transform=ax1.transAxes, fontweight="bold", fontsize=12)
    ax1.tick_params(axis='x', which='minor', bottom=False)

    # --- c ---
    width = 0.17

    c = cmap(2)
    bars = ax2.bar(x - 2 * width, monolithic_thresholds, width, yerr=monolithic_errors, color=c, alpha=0.4, ecolor=c,
                   capsize=3, label="monolithic")
    labels = [f"{t:.3f}({1000 * e:.0f})" for t, e in zip(monolithic_thresholds, monolithic_errors)]
    ax2.bar_label(bars, labels, rotation=90, padding=5, color=c, weight="bold")

    c = cmap(3)
    bars = ax2.bar(x3 - width, circuit_bad_thresholds, width, yerr=circuit_bad_errors, color=c, alpha=0.5, ecolor=c,
                   capsize=3, label="distributed (circuit)")
    labels = [f"{t:.3f}({1000 * e:.0f})" for t, e in zip(circuit_bad_thresholds, circuit_bad_errors)]
    ax2.bar_label(bars, labels, rotation=90, padding=5, color=c, weight="bold")

    c = cmap(4)
    bars = ax2.bar(x3, network_bad_thresholds, width, yerr=network_bad_errors, color=c, alpha=0.5, ecolor=c, capsize=3,
                   label="distributed (network)")
    labels = [f"{t:.3f}({1000 * e:.0f})" for t, e in zip(network_bad_thresholds, network_bad_errors)]
    ax2.bar_label(bars, labels, rotation=90, padding=5, color=c, weight="bold")

    c = cmap(3)
    bars = ax2.bar(x3 + width, circuit_good_thresholds, width, yerr=circuit_good_errors, color=c, alpha=0.5, ecolor=c,
                   capsize=3)
    labels = [f"{t:.3f}({1000 * e:.0f})" for t, e in zip(circuit_good_thresholds, circuit_good_errors)]
    ax2.bar_label(bars, labels, rotation=90, padding=5, color=c, weight="bold")

    c = cmap(4)
    bars = ax2.bar(x3 + 2 * width, network_good_thresholds, width, yerr=network_good_errors, color=c, alpha=0.5,
                   ecolor=c, capsize=3)
    labels = [f"{t:.3f}({1000 * e:.0f})" for t, e in zip(network_good_thresholds, network_good_errors)]
    ax2.bar_label(bars, labels, rotation=90, padding=5, color=c, weight="bold")

    combined_ticks = np.concatenate((x - 2 * width, x3 - 0.5 * width, x3 + 1.5 * width))
    combined_names = np.concatenate((names, distributed_bad_names, distributed_good_names))
    ax2.set_xticks(combined_ticks)
    ax2.set_xticklabels(combined_names, rotation=90)

    ax2.set_ylim(0, 3.5)
    ax2.set_ylabel("Error thresholds (\%)")

    ax2.legend(loc="upper left")
    ax2.text(-0.12, 1.0, "c", va="center", ha="right", transform=ax2.transAxes, fontweight="bold", fontsize=12)
    ax2.tick_params(axis='x', which='minor', bottom=False)

    fig.subplots_adjust(wspace=0.35)
    fig.savefig("figures/summary_thresholds.pdf")


if __name__ == "__main__":
    # plot_fault_tolerant_region_phenomenological_vs_erasure(order=2)
    plot_monolithic_thresholds(order=2)
