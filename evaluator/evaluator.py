from evaluator.qiskit_plugin import *
from evaluator.pytket_plugin import *

from evaluator.utils import get_openqasm_gates
from mqt.bench import benchmark_generator

from pytket.extensions.qiskit import qiskit_to_tk

import numpy as np
import signal

import json

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

def timeout_watcher(func, args, timeout):
    class TimeoutException(Exception):  # Custom exception class
        pass

    def timeout_handler(signum, frame):  # Custom signal handler
        raise TimeoutException

    # Change the behavior of SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)

    signal.alarm(timeout)
    try:
        res = func(*args)
    except TimeoutException:
        print("Calculation/Generation exceeded timeout limit for ", func, args[1:])
        return False
    except Exception as e:
        print("Something else went wrong: ", e)
        return False
    else:
        # Reset the alarm
        signal.alarm(0)

    return res

def dict_to_featurevector(gate_dict, num_qubits):
    openqasm_gates_list = get_openqasm_gates()
    res_dct = {openqasm_gates_list[i] for i in range(0, len(openqasm_gates_list))}
    res_dct = dict.fromkeys(res_dct, 0)
    for key, val in dict(gate_dict).items():
        if not key in res_dct:
            print(key, "gate not found in openQASM 2.0 gateset")
        else:
            res_dct[key] = val

    res_dct["num_qubits"] = num_qubits
    return res_dct

def create_gate_lists(min_qubit: int, max_qubit: int, stepsize: int = 1, timeout: int = 10):
    benchmarks = [
        "dj",
        "grover-noancilla",
        "grover-v-chain",
        "ghz",
        "graphstate",
        "qft",
        "qftentangled",
        "qpeexact",
        "qpeinexact",
        "qwalk-noancilla",
        "qwalk-v-chain",
        "realamprandom",
        "su2random",
        "twolocalrandom",
        "vqe",
        "wstate",
        "vqe",
        "qaoa",
        "portfoliovqe",
        "portfolioqaoa",
        "qgan"
    ]
    res = []
    for benchmark in benchmarks:
        for num_qubits in range(min_qubit, max_qubit, stepsize):
            print(benchmark, num_qubits)
            qc = timeout_watcher(benchmark_generator.get_one_benchmark, [benchmark, 1, num_qubits], timeout)
            if not qc:
                break
            actual_num_qubits = qc.num_qubits
            qasm_qc = qc.qasm()
            qc = QuantumCircuit.from_qasm_str(qasm_qc)
            qiskit_gates = timeout_watcher(get_qiskit_gates, [qc], timeout)
            if not qiskit_gates:
                break
            try:
                qc_tket = qiskit_to_tk(qc)
                ops_list = qc.count_ops()
                feature_vector = dict_to_featurevector(ops_list, actual_num_qubits)
                tket_gates = timeout_watcher(get_tket_gates, [qc_tket], timeout)
                if not tket_gates:
                    break
                res.append((benchmark, feature_vector, qiskit_gates+tket_gates))
            except Exception as e:
                print("fail: ", e)


    jsonString = json.dumps(res, indent=4, sort_keys=True)
    with open('json_data.json', 'w') as outfile:
        outfile.write(jsonString)
    return

def extract_training_data_from_json(json_path:str="json_data.json"):
    with open(json_path, 'r') as f:
        data = json.load(f)
    training_data = []

    for benchmark in data:
        scores = []
        num_qubits = benchmark[1]["num_qubits"]
        # Qiskit Scores
        for elem in benchmark[2][1]:
            if (elem[0] is None):
                score = 10000000000
            else:
                score = calc_score_from_gates_list(elem[0], get_backend_information(elem[1]))
            scores.append(score)
        # Tket Scores
        for elem in benchmark[2][3]:
            if (elem[0] is None):
                score = 10000000000
            else:
                score = calc_score_from_gates_list(elem[0], get_backend_information(elem[1]))
            scores.append(score)

        training_data.append((list(benchmark[1].values()), np.argmin(scores)))
        machines = [
            "qiskit_ibm_washington",
            "qiskit_ibm_montreal",
            "qiskit_ionq",
            "qiskit_rigetti",
            "qiskit_oqc",
            "tket_ibm_washington", "tket_ibm_montreal", "tket_ionq", "tket_rigetti", "tket_oqc"

        ]
        print(num_qubits, machines[np.argmin(scores)])
    return training_data


def train_simple_ml_model(X, y, show_test_pred=False):
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.30, random_state=40)
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=43))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100)

    pred_train = model.predict(X_train)
    scores = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

    pred_test = model.predict(X_test)
    scores2 = model.evaluate(X_test, y_test, verbose=True)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

    if(show_test_pred):
        check_test_predictions(X_test, pred_test, y_test)

    return model


def check_test_predictions(X_test, pred_test, y_test):
    machines = [
        "qiskit_ibm_washington",
        "qiskit_ibm_montreal",
        "qiskit_ionq",
        "qiskit_rigetti",
        "qiskit_oqc",
        "tket_ibm_washington", "tket_ibm_montreal", "tket_ionq", "tket_rigetti", "tket_oqc"

    ]
    for i, pred in enumerate(pred_test):
        print(machines[np.argmax(pred)], machines[y_test[i]], X_test[i, 19])

    return


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Create Training Data")
    # parser.add_argument(
    #     "--min", type=int, default=3,
    # )
    # parser.add_argument(
    #     "--max", type=int, default=20,
    # )
    # parser.add_argument("--step", type=int, default=3)
    # parser.add_argument("--timeout", type=int, default=10)
    #
    # args = parser.parse_args()
    # #characteristics = create_training_data(args.min, args.max, args.step, args.timeout)
    # #data = create_gate_lists(args.min, args.max, args.step, args.timeout)

    training_data = extract_training_data_from_json("json_data_big.json")
    X, y = zip(*training_data)
    train_simple_ml_model(X, y, True)
    print("Done")
