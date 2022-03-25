import os

import matplotlib.pyplot as plt

from qiskit_plugin import get_qiskit_scores
from pytket_plugin import get_tket_scores


def evaluation_qasm_files(directory_path: str = "../MQTbench/qasm_output"):
    i = 0
    path_to_all_files = directory_path
    res = []
    for file in os.listdir(path_to_all_files):
        if "indep" in file:
            print(file)
            filepath = os.path.join(path_to_all_files, file)

            try:
                res_qiskit = get_qiskit_scores(filepath, opt_level=0)
                res_pytket = get_tket_scores(filepath, opt_level=0)
                for i in range(len(res_pytket)):
                    if res_pytket[i] != 0:
                        score = (res_qiskit[i]) / (res_pytket[i])
                        res.append(score)
            except Exception as e:
                print("fail: ", e)
            i += 1
            if i > 5:
                break

    return res


def save_plot_res(res):
    plt.hist(res, bins=100)
    plt.title("Comparison of Pytket vs. Qiskit Performance")
    plt.savefig("hist.png")


if __name__ == "__main__":
    res = evaluation_qasm_files()
    save_plot_res(res)
