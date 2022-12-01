import argparse
import glob
import sys
from multiprocessing import Pipe, Process
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, load
from mqt.bench.utils import qiskit_helper, tket_helper
from multiprocess.connection import wait
from pytket import OpType, architecture
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import PlacementPass, RoutingPass, auto_rebase_pass
from pytket.placement import GraphPlacement
from pytket.qasm import circuit_to_qasm_str
from qiskit import QuantumCircuit, transpile
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from mqt.predictor import utils
from mqt.predictor.PhaseOrdererEnv import PhaseOrdererEnv

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources

plt.rcParams["font.family"] = "Times New Roman"


class Predictor:
    def __init__(self):
        self.clf = None

    def set_classifier(self, clf):
        self.clf = clf

    def compile_all_circuits_for_qc(
        self,
        filename: str,
        source_path: str = None,
        target_path: str = None,
        timeout: int = 10,
    ):
        """Handles the creation of one training sample.

        Keyword arguments:
        filename -- qasm circuit sample filename
        source_path -- path to file
        target_path -- path to directory for compiled circuit
        timeout -- timeout in seconds

        Return values:
        True -- at least one compilation option succeeded
        False -- if not
        """
        if source_path is None:
            source_path = str(
                resources.files("mqt.predictor").joinpath("training_samples")
            )

        if target_path is None:
            target_path = str(
                resources.files("mqt.predictor").joinpath("training_samples_compiled")
            )

        print("compile_all_circuits_for_qc:", filename)

        qc = QuantumCircuit.from_qasm_file(Path(source_path) / filename)

        if not qc:
            return False

        compilation_pipeline = utils.get_compilation_pipeline()

        results = []
        comp_path_id = 0
        try:
            for gate_set_name, devices in compilation_pipeline.get("devices").items():
                for device_name, max_qubits in devices:
                    for compiler, settings in compilation_pipeline["compiler"].items():
                        if "qiskit" in compiler:
                            for opt_level in settings["optimization_level"]:
                                target_filename = (
                                    filename.split(".qasm")[0] + "_" + str(comp_path_id)
                                )
                                comp_path_id += 1
                                if max_qubits >= qc.num_qubits:
                                    tmp = utils.timeout_watcher(
                                        qiskit_helper.get_mapped_level,
                                        [
                                            qc,
                                            gate_set_name,
                                            qc.num_qubits,
                                            device_name,
                                            opt_level,
                                            False,
                                            False,
                                            target_path,
                                            target_filename,
                                        ],
                                        timeout,
                                    )
                                    results.append(tmp)
                                    if not tmp:
                                        continue
                        elif "tket" in compiler:
                            for lineplacement in settings["lineplacement"]:
                                target_filename = (
                                    filename.split(".qasm")[0] + "_" + str(comp_path_id)
                                )
                                comp_path_id += 1
                                if max_qubits >= qc.num_qubits:
                                    tmp = utils.timeout_watcher(
                                        tket_helper.get_mapped_level,
                                        [
                                            qc,
                                            gate_set_name,
                                            qc.num_qubits,
                                            device_name,
                                            lineplacement,
                                            False,
                                            False,
                                            target_path,
                                            target_filename,
                                        ],
                                        timeout,
                                    )
                                    results.append(tmp)
                                    if not tmp:
                                        continue

            if all(x is False for x in results):
                print("No compilation succeeded for this quantum circuit.")
                return False
            return True

        except Exception as e:
            print("fail: ", e)
            return False

    def generate_compiled_circuits(
        self,
        source_path: str = None,
        target_path: str = None,
        timeout: int = 10,
    ):
        """Handles the creation of all training samples.

        Keyword arguments:
        source_path -- path to file
        target_directory -- path to directory for compiled circuit
        timeout -- timeout in seconds

        """
        if source_path is None:
            source_path = str(
                resources.files("mqt.predictor").joinpath("training_samples")
            )

        if target_path is None:
            target_path = str(
                resources.files("mqt.predictor").joinpath("training_samples_compiled")
            )

        global TIMEOUT
        TIMEOUT = timeout

        source_circuits_list = []

        for file in Path(source_path).iterdir():
            if "qasm" in str(file):
                source_circuits_list.append(str(file))

        if (
            len(source_circuits_list) == 0
            and (Path(source_path) / "mqtbench_training_samples.zip").exists()
        ):
            path_zip = str(Path(source_path) / "mqtbench_training_samples.zip")
            import zipfile

            with zipfile.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(source_path)

        if not Path(source_path).is_dir():
            Path(source_path).mkdir()

        Parallel(n_jobs=-1, verbose=100)(
            delayed(self.compile_all_circuits_for_qc)(
                filename, source_path, target_path, timeout
            )
            for filename in source_circuits_list
        )

    def generate_trainingdata_from_qasm_files(
        self,
        source_path: str = None,
        target_path: str = None,
    ):
        """Handles to create training data from all generated training samples

        Keyword arguments:
        source_path -- path to file
        target_directory -- path to directory for compiled circuit

        Return values:
        training_data -- training data
        name_list -- names of all training samples
        scores -- evaluation scores for all compilation options
        """
        if source_path is None:
            source_path = str(
                resources.files("mqt.predictor").joinpath("training_samples")
            )

        if target_path is None:
            target_path = str(
                resources.files("mqt.predictor").joinpath("training_samples_compiled")
            )

        if utils.init_all_config_files():
            print("Calibration files successfully initiated")
        else:
            print("Calibration files Initiation failed")
            return None

        # init resulting list (feature vector, name, scores)
        training_data = []
        name_list = []
        scores_list = []

        results = Parallel(n_jobs=-1, verbose=100)(
            delayed(self.generate_training_sample)(
                str(filename), source_path, target_path
            )
            for filename in Path(source_path).iterdir()
        )
        for sample in results:
            if not sample:
                continue

            training_sample, circuit_name, scores = sample
            training_data.append(training_sample)
            name_list.append(circuit_name)
            scores_list.append(scores)

        return (training_data, name_list, scores_list)

    def generate_training_sample(
        self,
        file: str,
        source_path: str = None,
        target_path: str = None,
    ):
        """Handles to create training data from a single generated training sample

        Keyword arguments:
        file -- filename for the training sample
        source_path -- path to file
        target_path -- path to directory for compiled circuit

        Return values:
        training_sample -- training data sample
        circuit_name -- names of the training sample circuit
        scores -- evaluation scores for all compilation options
        """
        if source_path is None:
            source_path = str(
                resources.files("mqt.predictor").joinpath("training_samples")
            )

        if target_path is None:
            target_path = str(
                resources.files("mqt.predictor").joinpath("training_samples_compiled")
            )
        if ".qasm" not in file:
            return False

        LUT = utils.get_index_to_comppath_LUT()
        utils.init_all_config_files()
        print("Checking ", file)
        scores = []
        for _ in range(len(LUT)):
            scores.append([])
        all_relevant_paths = Path(target_path) / (file.split(".")[0] + "*")
        all_relevant_files = glob.glob(str(all_relevant_paths))

        for filename in all_relevant_files:
            if (file.split(".")[0] + "_") in filename and filename.endswith(".qasm"):
                comp_path_index = int(filename.split("_")[-1].split(".")[0])
                device = LUT.get(comp_path_index)[1]

                score = utils.reward_expected_fidelity(filename, device)
                scores[comp_path_index] = score

        num_not_empty_entries = 0
        for i in range(len(LUT)):
            if not scores[i]:
                scores[i] = utils.get_width_penalty()
            else:
                num_not_empty_entries += 1

        if num_not_empty_entries == 0:
            return False

        feature_vec = utils.create_feature_dict(str(Path(source_path) / file))
        training_sample = (list(feature_vec.values()), np.argmax(scores))
        circuit_name = file.split(".")[0]

        return (training_sample, circuit_name, scores)

    def train_random_forest_classifier(self, visualize_results=False):

        (
            X_train,
            X_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
            names_list,
            scores_list,
        ) = self.get_prepared_training_data(save_non_zero_indices=True)

        scores_filtered = [scores_list[i] for i in indices_test]
        names_filtered = [names_list[i] for i in indices_test]

        tree_param = [
            {
                "n_estimators": [100, 200, 500],
                "max_features": ["auto", "sqrt"],
                "max_depth": list(range(8, 30, 6)),
                "min_samples_split": list(range(2, 20, 6)),
                "min_samples_leaf": list(range(2, 20, 6)),
                "bootstrap": [True, False],
            },
        ]

        clf = RandomForestClassifier(random_state=0)
        clf = GridSearchCV(clf, tree_param, cv=5, n_jobs=8).fit(X_train, y_train)

        if visualize_results:
            y_pred = np.array(list(clf.predict(X_test)))
            res = self.plot_eval_histogram(
                scores_filtered, y_pred, y_test, filename="RandomForestClassifier"
            )

            print("Best Accuracy: ", clf.best_score_)
            top3 = (res.count(1) + res.count(2) + res.count(3)) / len(res)
            print("Top 3: ", top3)
            print("Feature Importance: ", clf.best_estimator_.feature_importances_)

            self.plot_eval_all_detailed_compact_normed(
                names_filtered, scores_filtered, y_pred, y_test
            )

        self.set_classifier(clf.best_estimator_)
        utils.save_classifier(clf.best_estimator_)
        print("Random Forest classifier is trained and saved.")

        return self.clf is not None

    def get_prepared_training_data(self, save_non_zero_indices=False):
        training_data, names_list, scores_list = utils.load_training_data()
        X, y = zip(*training_data)
        X = list(X)
        y = list(y)
        for i in range(len(X)):
            X[i] = list(X[i])
            scores_list[i] = list(scores_list[i])

        X, y, indices = np.array(X), np.array(y), np.array(range(len(y)))

        # Store all non zero feature indices
        non_zero_indices = []
        for i in range(len(X[0])):
            if sum(X[:, i]) > 0:
                non_zero_indices.append(i)
        X = X[:, non_zero_indices]

        if save_non_zero_indices:
            data = np.asarray(non_zero_indices)
            np.save("non_zero_indices.npy", data)

        (
            X_train,
            X_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
        ) = train_test_split(X, y, indices, test_size=0.3, random_state=5)

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
            names_list,
            scores_list,
        )

    def plot_eval_histogram(
        self, scores_filtered, y_pred, y_test, filename="histogram"
    ):
        """Method to generate the histogram of the resulting ranks

        Keyword arguments:
        scores_filtered -- all scores filtered for the respectively predicted indices of all training data
        y_pred -- predicted labels
        y_test -- actual labels
        filename -- name of the saved figure

        Return values:
        training_sample -- training data sample
        circuit_name -- names of the training sample circuit
        scores -- all achieved ranks
        """
        res = []
        for i in range(len(y_pred)):
            assert np.argmax(scores_filtered[i]) == y_test[i]
            predicted_score = scores_filtered[i][y_pred[i]]
            score = list(np.sort(scores_filtered[i])[::-1]).index(predicted_score)
            res.append(score + 1)

        assert len(res) == len(y_pred)

        plt.figure(figsize=(10, 5))

        num_of_comp_paths = len(utils.get_index_to_comppath_LUT())
        plt.bar(
            list(range(0, num_of_comp_paths, 1)),
            height=[
                res.count(i) / len(res) for i in range(1, num_of_comp_paths + 1, 1)
            ],
            width=1,
        )
        plt.xticks(
            list(range(0, num_of_comp_paths, 1)),
            list(range(1, num_of_comp_paths + 1, 1)),
            fontsize=16,
        )
        plt.yticks(fontsize=16)

        plt.xlabel(
            "Best prediction                                                        Worst prediction",
            fontsize=18,
        )
        plt.ylabel("Relative frequency", fontsize=18)
        result_path = Path("results")
        if not result_path.is_dir():
            result_path.mkdir()
        plt.savefig(result_path / (filename + ".pdf"))
        plt.show()

        return res

    def plot_eval_all_detailed_compact_normed(
        self, names_list, scores_filtered, y_pred, y_test
    ):
        """Method to generate the detailed graph to examine the differences in evaluation scores

        Keyword arguments:
        names_list -- all names filtered for the respectively predicted indices of all training data
        scores_filtered -- all scores filtered for the respectively predicted indices of all training data
        y_pred -- predicted labels
        y_test -- actual labels
        """

        # Create list of all qubit numbers and sort them
        names_list_num_qubits = []
        for i in range(len(names_list)):
            assert np.argmax(scores_filtered[i]) == y_test[i]
            names_list_num_qubits.append(
                int(names_list[i].split("_")[-1].split(".")[0])
            )

        # Sort all other list (num_qubits, scores and y_pred) accordingly
        (
            qubit_list_sorted,
            scores_filtered_sorted_accordingly,
            y_pred_sorted_accordingly,
        ) = zip(*sorted(zip(names_list_num_qubits, scores_filtered, y_pred)))
        plt.figure(figsize=(17, 8))
        for i in range(len(names_list_num_qubits)):
            tmp_res = scores_filtered_sorted_accordingly[i]
            max_score = max(tmp_res)
            for j in range(len(tmp_res)):
                plt.plot(i, tmp_res[j] / max_score, "b.", alpha=1.0, markersize=1.7)

            plt.plot(
                i,
                tmp_res[y_pred_sorted_accordingly[i]] / max_score,
                "#ff8600",
                marker=".",
                linestyle="None",
            )

        plt.xticks(
            list(range(0, len(scores_filtered), 20)),
            [qubit_list_sorted[i] for i in range(0, len(scores_filtered), 20)],
            fontsize=18,
        )
        plt.yticks(fontsize=18)
        plt.xlabel(
            "Unseen test circuits (sorted along the number of qubits)", fontsize=18
        )
        plt.ylabel(
            "Evaluation scores of combinations of options \n (normalized per test circuit)",
            fontsize=18,
        )
        plt.tight_layout()

        plt.ylim(0, 1.05)
        plt.xlim(0, len(scores_filtered))
        result_path = Path("results")
        if not result_path.is_dir():
            result_path.mkdir()
        plt.savefig(result_path / "y_pred_eval_normed.pdf")

        return

    def predict(self, qasm_str_or_path: str):
        """Returns a compilation option prediction index for a given qasm file path or qasm string."""

        if self.clf is None:
            path = resources.files("mqt.predictor") / "trained_clf.joblib"
            if path.is_file():
                self.clf = load(str(path))
            else:
                print("Fail: Classifier is neither trained nor saved!")
                return None

        feature_dict = utils.create_feature_dict(qasm_str_or_path)
        if not feature_dict:
            return None
        feature_vector = list(feature_dict.values())

        path = resources.files("mqt.predictor") / "non_zero_indices.npy"
        non_zero_indices = np.load(str(path), allow_pickle=True)
        feature_vector = [feature_vector[i] for i in non_zero_indices]

        return self.clf.predict([feature_vector])[0]

    def compile_predicted_compilation_path(
        self, qasm_str_or_path: str, prediction: int
    ):
        """Returns the compiled quantum circuit as a qasm string when the original qasm circuit is provided as either
        a string or a file path and the prediction index is given."""

        LUT = utils.get_index_to_comppath_LUT()
        if prediction < 0 or prediction >= len(LUT):
            print("Provided prediction is faulty.")
            return None

        if Path(qasm_str_or_path).exists():
            print("Reading from .qasm path: ", qasm_str_or_path)
            qc = QuantumCircuit.from_qasm_file(qasm_str_or_path)
        elif QuantumCircuit.from_qasm_str(qasm_str_or_path):
            print("Reading from .qasm str")
            qc = QuantumCircuit.from_qasm_str(qasm_str_or_path)
        else:
            print("Neither a qasm file path nor a qasm str has been provided.")
            return False

        prediction_information = LUT.get(prediction)
        gate_set_name = prediction_information[0]
        device = prediction_information[1]
        compiler = prediction_information[2]
        compiler_settings = prediction_information[3]

        print("")
        if compiler == "qiskit":
            compiled_qc = qiskit_helper.get_mapped_level(
                qc, gate_set_name, qc.num_qubits, device, compiler_settings, False, True
            )
            return compiled_qc.qasm()
        elif compiler == "tket":
            compiled_qc = tket_helper.get_mapped_level(
                qc,
                gate_set_name,
                qc.num_qubits,
                device,
                compiler_settings,
                False,
                True,
            )
            return circuit_to_qasm_str(compiled_qc)
        else:
            print("Error: Compiler not found.")
            return False

    def compile_using_supervised_ML(self, qc):
        qc_compiled = self.compile_predicted_compilation_path(qc, self.predict(qc))
        return qc_compiled

    def compile_using_RL(self, qc):
        pass

    def train_all_RL_models(
        self,
        timestep=1000,
        tensorboard_log="./training",
        verbose=2,
        fid=False,
        dep=False,
        par=False,
    ):
        reward_functions = []  # ["fidelity", "depth", "two_qubit_count"]
        if fid:
            reward_functions.append("fidelity")
            print("fidelity added")
        if dep:
            reward_functions.append("critical_depth")
            print("critical_depth added")
        if par:
            reward_functions.append("parallelism")
            print("parallelism added")

        for rew in reward_functions:
            print("Start training for: ", rew)
            env = PhaseOrdererEnv(reward_function=rew)

            model = MaskablePPO(
                MaskableActorCriticPolicy,
                env,
                verbose=verbose,
                tensorboard_log=tensorboard_log + "_" + rew,
                gamma=0.98,
            )
            model.learn(total_timesteps=timestep, progress_bar=True)
            model.save("model_" + rew)

    def evaluate_sample_circuit_using_RL(self, file):
        print(file)

        reward_functions = ["parallelism", "fidelity", "critical_depth"]
        for rew in reward_functions:
            model = MaskablePPO.load("model_" + rew)

            env = PhaseOrdererEnv(rew)
            obs, _ = env.reset(file)
            qc = env.state
            start_time = time.time()
            while True:
                action_masks = get_action_masks(env)
                action, _states = model.predict(obs, action_masks=action_masks)
                action = int(action)
                obs, reward, done, trunc, info = env.step(action)
                if done:
                    duration = time.time() - start_time
                    if rew == "fidelity":
                        RL_fid = np.round(
                            utils.reward_expected_fidelity(env.state, env.device), 2
                        )
                        RL_fid_time = np.round(duration, 2)
                        RL_fid_crit_depth = np.round(
                            utils.reward_crit_depth(env.state), 2
                        )
                        RL_fid_parallelism = np.round(
                            utils.reward_parallelism(env.state), 2
                        )
                    elif rew == "parallelism":
                        RL_parallelism = np.round(
                            utils.reward_parallelism(env.state), 2
                        )
                        RL_parallelism_time = np.round(duration, 2)
                        RL_parallelism_fid = np.round(
                            utils.reward_expected_fidelity(env.state, env.device), 2
                        )
                        RL_parallelism_crit_depth = np.round(
                            utils.reward_crit_depth(env.state), 2
                        )
                    elif rew == "critical_depth":
                        RL_crit_depth = np.round(utils.reward_crit_depth(env.state), 2)
                        RL_crit_depth_time = np.round(duration, 2)
                        RL_crit_depth_fid = np.round(
                            utils.reward_expected_fidelity(env.state, env.device), 2
                        )
                        RL_crit_depth_parallelism = np.round(
                            utils.reward_parallelism(env.state), 2
                        )
                    break

        start_time = time.time()
        transpiled_qc_qiskit = transpile(
            qc,
            basis_gates=utils.get_ibm_native_gates(),
            coupling_map=utils.get_cmap_from_devicename("ibm_washington"),
            optimization_level=3,
            seed_transpiler=1,
        )
        duration = time.time() - start_time
        qiskit_o3_fid = np.round(
            utils.reward_expected_fidelity(transpiled_qc_qiskit, "ibm_washington"), 2
        )
        qiskit_o3_crit_depth = utils.reward_crit_depth(transpiled_qc_qiskit)
        qiskit_o3_parallel = utils.reward_parallelism(transpiled_qc_qiskit)
        qiskit_o3_time = np.round(duration, 2)

        tket_qc = qiskit_to_tk(qc)
        arch = architecture.Architecture(
            utils.get_cmap_from_devicename("ibm_washington")
        )
        ibm_rebase = auto_rebase_pass(
            {OpType.Rz, OpType.SX, OpType.X, OpType.CX, OpType.Measure}
        )

        start_time = time.time()
        ibm_rebase.apply(tket_qc)

        conn_read, conn_write = Pipe(duplex=False)
        p = Process(
            target=utils.execute_TKET_FullPeephole,
            args=(
                tket_qc,
                conn_write,
            ),
        )
        p.start()

        success = False
        while p.is_alive():
            wait([p.sentinel, conn_read])  # block-wait until something gets ready
            if conn_read.poll():  # check if something can be received
                msg = conn_read.recv()
                success = msg == "success"
        p.join()
        if not success:
            print("Fail in execute_TKET_FullPeephole")
            return False

        PlacementPass(GraphPlacement(arch)).apply(tket_qc)
        RoutingPass(arch).apply(tket_qc)
        ibm_rebase.apply(tket_qc)
        duration = time.time() - start_time

        transpiled_qc_tket = tk_to_qiskit(tket_qc)

        tket_fid = np.round(
            utils.reward_expected_fidelity(transpiled_qc_tket, "ibm_washington"), 2
        )
        tket_crit_depth = utils.reward_crit_depth(transpiled_qc_tket)
        tket_parallelism = utils.reward_parallelism(transpiled_qc_tket)
        tket_time = np.round(duration, 2)

        return (
            str(file).split("/")[-1].split(".")[0].replace("_", " ").split(" ")[0],
            str(file).split("/")[-1].split(".")[0].replace("_", " ").split(" ")[-1],
            qiskit_o3_fid,
            tket_fid,
            RL_fid,
            qiskit_o3_parallel,
            tket_parallelism,
            RL_parallelism,
            qiskit_o3_crit_depth,
            tket_crit_depth,
            RL_crit_depth,
            qiskit_o3_time,
            tket_time,
            RL_fid_time,
            RL_parallelism_time,
            RL_crit_depth_time,
            RL_fid_crit_depth,
            RL_fid_parallelism,
            RL_crit_depth_fid,
            RL_crit_depth_parallelism,
            RL_parallelism_fid,
            RL_parallelism_crit_depth,
        )

    def eval_all_sample_circuits_using_RL(self):
        res_csv = []
        res_csv.append(
            (
                "Benchmark",
                "Qubits",
                "Qiskit O3 Fid",
                "TKET Fid",
                "RL Fid",
                "Qiskit O3 Parallelism",
                "TKET Parallelism",
                "RL Parallelism",
                "Qiskit O3 Crit Depth",
                "TKET Crit Depth",
                "RL Crit Depth",
                "Qiskit O3 Time",
                "TKET Time",
                "RL Fid Time",
                "RL Parallelism Time",
                "RL Crit Depth Time",
                "RL Fid Crit Depth",
                "RL Fid Parallelism",
                "RL Crit Depth Fid",
                "RL Crit Depth Parallelism",
                "RL Parallelism Fid",
                "RL Parallelism Crit Depth",
            )
        )

        results = Parallel(n_jobs=-1, verbose=3, backend="threading")(
            delayed(evaluate_sample_circuit)(file)
            for file in list(Path("./sample_circuits").glob("*.qasm"))
        )
        for res in results:
            res_csv.append(res)
        # print(results)
        np.savetxt("res.csv", res_csv, delimiter=",", fmt="%s")


def evaluate_sample_circuit(file):
    print(file)

    reward_functions = ["parallelism", "fidelity", "critical_depth"]
    for rew in reward_functions:
        model = MaskablePPO.load("model_" + rew)

        env = PhaseOrdererEnv(rew)
        obs, _ = env.reset(file)
        qc = env.state
        start_time = time.time()
        while True:
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks)
            action = int(action)
            obs, reward, done, trunc, info = env.step(action)
            if done:
                duration = time.time() - start_time
                if rew == "fidelity":
                    RL_fid = np.round(
                        utils.reward_expected_fidelity(env.state, env.device), 2
                    )
                    RL_fid_time = np.round(duration, 2)
                    RL_fid_crit_depth = np.round(utils.reward_crit_depth(env.state), 2)
                    RL_fid_parallelism = np.round(
                        utils.reward_parallelism(env.state), 2
                    )
                elif rew == "parallelism":
                    RL_parallelism = np.round(utils.reward_parallelism(env.state), 2)
                    RL_parallelism_time = np.round(duration, 2)
                    RL_parallelism_fid = np.round(
                        utils.reward_expected_fidelity(env.state, env.device), 2
                    )
                    RL_parallelism_crit_depth = np.round(
                        utils.reward_crit_depth(env.state), 2
                    )
                elif rew == "critical_depth":
                    RL_crit_depth = np.round(utils.reward_crit_depth(env.state), 2)
                    RL_crit_depth_time = np.round(duration, 2)
                    RL_crit_depth_fid = np.round(
                        utils.reward_expected_fidelity(env.state, env.device), 2
                    )
                    RL_crit_depth_parallelism = np.round(
                        utils.reward_parallelism(env.state), 2
                    )
                break

    start_time = time.time()
    transpiled_qc_qiskit = transpile(
        qc,
        basis_gates=utils.get_ibm_native_gates(),
        coupling_map=utils.get_cmap_from_devicename("ibm_washington"),
        optimization_level=3,
        seed_transpiler=1,
    )
    duration = time.time() - start_time
    qiskit_o3_fid = np.round(
        utils.reward_expected_fidelity(transpiled_qc_qiskit, "ibm_washington"), 2
    )
    qiskit_o3_crit_depth = utils.reward_crit_depth(transpiled_qc_qiskit)
    qiskit_o3_parallel = utils.reward_parallelism(transpiled_qc_qiskit)
    qiskit_o3_time = np.round(duration, 2)

    tket_qc = qiskit_to_tk(qc)
    arch = architecture.Architecture(utils.get_cmap_from_devicename("ibm_washington"))
    ibm_rebase = auto_rebase_pass(
        {OpType.Rz, OpType.SX, OpType.X, OpType.CX, OpType.Measure}
    )

    start_time = time.time()
    ibm_rebase.apply(tket_qc)

    conn_read, conn_write = Pipe(duplex=False)
    p = Process(
        target=utils.execute_TKET_FullPeephole,
        args=(
            tket_qc,
            conn_write,
        ),
    )
    p.start()

    success = False
    while p.is_alive():
        wait([p.sentinel, conn_read])  # block-wait until something gets ready
        if conn_read.poll():  # check if something can be received
            msg = conn_read.recv()
            success = msg == "success"
    p.join()
    if not success:
        print("Fail in execute_TKET_FullPeephole")
        return False

    PlacementPass(GraphPlacement(arch)).apply(tket_qc)
    RoutingPass(arch).apply(tket_qc)
    ibm_rebase.apply(tket_qc)
    duration = time.time() - start_time

    transpiled_qc_tket = tk_to_qiskit(tket_qc)

    tket_fid = np.round(
        utils.reward_expected_fidelity(transpiled_qc_tket, "ibm_washington"), 2
    )
    tket_crit_depth = utils.reward_crit_depth(transpiled_qc_tket)
    tket_parallelism = utils.reward_parallelism(transpiled_qc_tket)
    tket_time = np.round(duration, 2)

    return (
        str(file).split("/")[-1].split(".")[0].replace("_", " ").split(" ")[0],
        str(file).split("/")[-1].split(".")[0].replace("_", " ").split(" ")[-1],
        qiskit_o3_fid,
        tket_fid,
        RL_fid,
        qiskit_o3_parallel,
        tket_parallelism,
        RL_parallelism,
        qiskit_o3_crit_depth,
        tket_crit_depth,
        RL_crit_depth,
        qiskit_o3_time,
        tket_time,
        RL_fid_time,
        RL_parallelism_time,
        RL_crit_depth_time,
        RL_fid_crit_depth,
        RL_fid_parallelism,
        RL_crit_depth_fid,
        RL_crit_depth_parallelism,
        RL_parallelism_fid,
        RL_parallelism_crit_depth,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create Training Data")

    parser.add_argument("--timeout", type=int, default=120)

    args = parser.parse_args()

    predictor = Predictor()

    # Generate compiled circuits and save them as qasm files
    predictor.generate_compiled_circuits(
        timeout=args.timeout,
    )
    # Postprocess some of those qasm files
    utils.postprocess_ocr_qasm_files()
    # Generate training data from qasm files
    res = predictor.generate_trainingdata_from_qasm_files()
    # Save those training data for faster re-processing
    utils.save_training_data(res)
    # Train the Random Forest Classifier on created training data
    predictor.train_random_forest_classifier()

    # Old RL driver
    # parser = argparse.ArgumentParser(description="Create RL Models")
    #
    # parser.add_argument("--fid", type=bool, default=False)
    # parser.add_argument("--dep", type=bool, default=False)
    # parser.add_argument("--par", type=bool, default=False)
    # args = parser.parse_args()
    #
    # utils.init_all_config_files()
    # # train_all_models(timestep=100000, tensorboard_log="./training", verbose=2, fid=args.fid, dep=args.dep, par=args.par)
    # eval_all_sample_circuits()
