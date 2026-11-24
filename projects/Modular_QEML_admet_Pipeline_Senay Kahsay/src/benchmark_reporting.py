"""Reproducible scientific figures from verified benchmark records."""
from __future__ import annotations

from pathlib import Path


def _save_figure(figure, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, metadata={"Date": None, "Creator": "ADMET reproducible benchmark"})
    if destination.suffix.lower() == ".svg":
        # Matplotlib emits path-line trailing spaces. Normalize only lexical
        # whitespace before content hashing; vector coordinates stay unchanged.
        destination.write_text("\n".join(line.rstrip() for line in destination.read_text().splitlines())+"\n")


def plot_transport(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    plt.rcParams.update({"svg.hashsalt": "admet-research-v1", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    labels = {"bare": "Bare annealing", "first_order_cd": "First-order CD",
              "exact_ground_cd": "Exact ground-projector CD"}
    colors = {"bare": "#526777", "first_order_cd": "#007F86", "exact_ground_cd": "#B85229"}
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for method, label in labels.items():
        rows = [row for row in report["rows"] if row["method"] == method]
        axes[0, 0].semilogx([r["duration"] for r in rows], [r["ground_fidelity"] for r in rows],
                            "o-", label=label, color=colors[method], markersize=4)
        if method != "bare":
            axes[0, 1].loglog([r["duration"] for r in rows],
                [r["integrated_control_hs_squared_quadrature"] for r in rows], "o-",
                color=colors[method], label=label, markersize=4)
        convergence = [row for row in report["convergence"] if row["method"] == method]
        axes[1, 0].loglog([r["midpoint_steps"] for r in convergence],
                          [max(r["phase_aligned_state_distance"], 1e-16) for r in convergence], "o-",
                          label=label, color=colors[method], markersize=4)
    snapshots = report["qsvt_snapshots"]
    if snapshots:
        x = [r["interpolation"] for r in snapshots]
        axes[1, 1].semilogy(x, [r["reference"]["recovered_evolution_error"] for r in snapshots],
                            "o-", label="Measured operator error", color="#007F86")
        axes[1, 1].semilogy(x, [r["reference"]["certified_recovered_evolution_error_bound"] for r in snapshots],
                            "s--", label="Numerical bound", color="#B85229")
        axes[1, 1].axhline(snapshots[0]["epsilon"], color="#526777", linestyle=":", label="Requested tolerance")
    else:
        axes[1, 1].text(.5, .5, "QSVT snapshots disabled", ha="center", va="center", transform=axes[1, 1].transAxes)
    axes[0, 0].set(title="A  Final ground-state fidelity", xlabel="Duration T (dimensionless)",
                   ylabel="Ground-state probability", ylim=(-.025, 1.04))
    axes[0, 1].set(title="B  Additional control cost", xlabel="Duration T (dimensionless)",
                   ylabel=r"$\int \mathrm{Tr}[(H_{CD}-H)^2]/2^n\,dt$")
    convergence_time = report["convergence"][0]["duration"]
    axes[1, 0].set(title="C  Independent midpoint convergence", xlabel=f"Midpoint steps (T = {convergence_time:g})",
                   ylabel="Phase-aligned state distance")
    axes[1, 1].set(title="D  Frozen QSVT snapshots", xlabel="Interpolation s", ylabel="Recovered operator 2-norm error")
    durations = sorted({row["duration"] for row in report["rows"]})
    for axis in axes[0]:
        axis.xaxis.set_major_locator(FixedLocator(durations))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
    axes[1, 0].xaxis.set_major_locator(FixedLocator(sorted({row["midpoint_steps"] for row in report["convergence"]})))
    axes[1, 0].xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
    axes[1, 0].xaxis.set_minor_formatter(NullFormatter())
    for axis in axes.flat:
        axis.grid(True, alpha=.16)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8, loc="best")
    figure.suptitle(f"{report['num_system_qubits']}-qubit information-derived Ising experiment • smooth schedule • ℏ = 1", fontsize=13)
    _save_figure(figure, destination)
    plt.close(figure)


def plot_information(report, destination):
    """Show estimator ranges without treating jitter seeds as sample replicates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"svg.hashsalt": "admet-research-v1", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    profiles = report["profiles"]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    colors = ["#007F86", "#B85229", "#526777"]
    for index, profile in enumerate(profiles):
        color = colors[index % len(colors)]
        for axis, key, exact_key, estimates_key in (
            (axes[0, 0], "pair_comparisons", "exact_discrete_cmi_bits", "ksg_cmi_bits_by_seed"),
            (axes[0, 1], "triad_comparisons", "exact_discrete_co_information_bits", "ksg_co_information_bits_by_seed")):
            records = profile[key]
            if records:
                x, y, lower, upper = [], [], [], []
                for row in records:
                    values = [value for value in row[estimates_key] if value is not None]
                    mean = float(np.mean(values))
                    x.append(row[exact_key]); y.append(mean)
                    lower.append(max(0., mean-min(values))); upper.append(max(0., max(values)-mean))
                axis.errorbar(x, y, yerr=[lower, upper], fmt="o" if index == 0 else "s",
                              markersize=6 if index == 0 else 4, capsize=2, color=color, alpha=.8,
                              markerfacecolor="none" if index == 0 else color, zorder=4-index,
                              label=f"Top {profile['top_k']} (jitter range)")
        for key, offset, marker, label, marker_color in (
            ("pair_jaccard_between_seeds", -.08, "o", "Pair support", "#007F86"),
            ("triad_jaccard_between_seeds", .08, "s", "Triad support", "#B85229")):
            values = profile[key]
            if not values:
                if key == "pair_jaccard_between_seeds":
                    axes[1, 0].text(index, .5, "Undefined\n(one jitter seed)", ha="center", color="#526777", fontsize=9)
                continue
            mean = float(np.mean(values))
            axes[1, 0].errorbar(index+offset, mean, yerr=[[max(0., mean-min(values))], [max(0., max(values)-mean)]],
                                fmt=marker, capsize=4, color=marker_color,
                                label=label if index == 0 else None)
        diagnostic = profile["label_permutation_diagnostic"]
        observed = diagnostic["observed_fixed_feature_mean_mi_bits"]
        shuffled = diagnostic["permuted_fixed_feature_mean_mi_bits"]
        mean = float(np.mean(shuffled))
        axes[1, 1].plot(index-.08, observed, "o", color="#007F86", label="Observed labels" if index == 0 else None)
        axes[1, 1].errorbar(index+.08, mean, yerr=[[max(0., mean-min(shuffled))], [max(0., max(shuffled)-mean)]],
                            fmt="s", capsize=4, color="#526777", label="Shuffled mean/range" if index == 0 else None)
    for axis, threshold in ((axes[0, 0], .10), (axes[0, 1], .15)):
        low, high = min(*axis.get_xlim(), *axis.get_ylim()), max(*axis.get_xlim(), *axis.get_ylim())
        low, high = min(low, 0.), max(high, threshold*1.15)
        axis.plot([low, high], [low, high], ":", color="#526777", alpha=.65, label="Equal estimates")
        axis.axvline(threshold, color="#526777", linestyle="--", alpha=.3)
        axis.axhline(threshold, color="#526777", linestyle="--", alpha=.3)
        axis.set(xlim=(low, high), ylim=(low, high))
    axes[0, 0].set(title="A  Pair conditional information", xlabel="Exact discrete CMI (bits)",
                   ylabel="Mean KSG CMI (bits)")
    axes[0, 1].set(title="B  Signed three-feature co-information", xlabel="Exact discrete co-information (bits)",
                   ylabel="Mean KSG co-information (bits)")
    axes[1, 0].set(title="C  Support stability across jitter seeds", ylabel="Pairwise seed Jaccard (mean/range)",
                   ylim=(-.03, 1.06))
    axes[1, 1].set(title="D  Fixed-feature shuffles (descriptive)", ylabel="Mean feature–label MI (bits)")
    for axis in axes[1]:
        axis.set_xticks(range(len(profiles)), [f"Top {p['top_k']}" for p in profiles])
    for axis in axes.flat:
        axis.grid(True, alpha=.16)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8, loc="best")
    figure.suptitle(f"BBB_Martins • {report['raw_rows']:,} observations • {report['unique_canonical_molecules']:,} canonical identities", fontsize=13)
    _save_figure(figure, destination)
    plt.close(figure)
