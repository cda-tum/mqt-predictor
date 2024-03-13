from __future__ import annotations

import pathlib

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16


def read_mqt_predictor_file(num_qubits: int, device: str) -> list[float]:
    best_res = [1.0]
    for file in pathlib.Path("results/").glob(f"mqt_predictor_{num_qubits}_qubits*"):
        found_device = False
        with pathlib.Path(file).open() as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Device:") and line.split(":")[1].strip() == device:
                    found_device = True
                if line.startswith("evaluation_data:") and found_device:
                    eval_data = eval(line.split(":")[1])
                    if min(eval_data) < min(best_res):
                        best_res = eval_data
    return best_res


def read_baseline_date(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_baseline = np.loadtxt(path)
    sorted_data = sorted(data_baseline, key=lambda x: min(x))
    best_run = sorted_data[0]
    worst_run = sorted_data[-1]
    median_run = sorted_data[len(sorted_data) // 2]
    worst_overall_values = np.maximum.reduce(list(sorted_data))
    best_overall_values = np.minimum.reduce(list(sorted_data))
    return (
        best_run,
        worst_run,
        median_run,
        best_overall_values,
        worst_overall_values,
    )


def generate_eval_plot(
    best_overall_O3: np.ndarray,
    best_overall_O1: np.ndarray,
    worst_overall_O3: np.ndarray,
    worst_overall_O1: np.ndarray,
    mqt_predictor: list[float],
    device: str,
    num_qubits: int,
):
    for run in [best_overall_O3, best_overall_O1, worst_overall_O3, worst_overall_O1, mqt_predictor]:
        local_min = 1
        for i in range(len(run)):
            run[i] = min(run[i], local_min)
            local_min = min(run[i], local_min)

    colors = ["#D81B60", "#FFC107", "#004D40", "#C6E5AA", "#0065bd"]

    linewidth = 2.5
    plt.plot(mqt_predictor, label="Proposed Approach", color=colors[4], linewidth=linewidth)

    plt.fill_between(
        range(len(best_overall_O1)),
        best_overall_O1,
        worst_overall_O1,
        alpha=0.2,
        color=colors[0],
        label="Qiskit Default",
    )
    plt.fill_between(
        range(len(best_overall_O3)),
        best_overall_O3,
        worst_overall_O3,
        alpha=0.5,
        color=colors[2],
        label="Qiskit Most-optimized",
    )

    plt.title(f"Evaluation for {num_qubits} qubits on {device}")
    plt.ylabel("KL Divergence")
    plt.xlabel("Episodes")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"results/{num_qubits}_qubits_{device}.pdf", bbox_inches="tight")
