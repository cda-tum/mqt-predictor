[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Lint](https://github.com/cda-tum/MQTPredictor/actions/workflows/linter.yml/badge.svg)](https://github.com/cda-tum/MQTPredictor/actions/workflows/linter.yml)
[![CodeCov](https://github.com/cda-tum/MQTPredictor/actions/workflows/coverage.yml/badge.svg)](https://github.com/cda-tum/MQTPredictor/actions/workflows/coverage.yml)
[![Deploy to PyPI](https://github.com/cda-tum/MQTPredictor/actions/workflows/deploy.yml/badge.svg)](https://github.com/cda-tum/MQTPredictor/actions/workflows/deploy.yml)
[![codecov](https://codecov.io/gh/cda-tum/MQTPredictor/branch/main/graph/badge.svg?token=ZL5js1wjrB)](https://codecov.io/gh/cda-tum/MQTPredictor)

# MQT Predictor: Automatic Prediction of Good Compilation Paths

MQT Predictor is a framework suggesting a compilation options to use for an arbitrary quantum circuit according to the user's needs.
To this end, we provide two models prediction good compilation options and returning the accordingly compiled quantum circuit.

## Supervised Machine Learning Model (referred to as "ML")

Here, the problem is treated as a statistical classification task.
Furthermore, the resulting methodology does not only provide end-users with a prediction on the best compilation options,
but additionally provides insights on why certain decisions have been made—allowing them to learn from the predicted results.

For evaluation of our methodology, seven supervised machine learning classifiers have been used:

- Random Forest
- Gradient Boosting
- Decision Tree
- Nearest Neighbor
- Multilayer Perceptron
- Support Vector Machine
- Naive Bayes

In our exemplary scenario, the Random Forest classifier achieved the best performance.

This ML model comprises three main functionalities:

- The pre-trained Random Forest classifier to easily predict compilation options for an unseen quantum circuit
  in real-time and compile for the respective prediction,
- all other trained algorithms, and
- the possibility to adjust and customize the whole training data generation process, e.g., to add training data, compilation options, or adapt the evaluation function.

## Reinforcement Learning Model (referred to as "RL")

In this work, we take advantage of decades of classical compiler optimization and propose a
reinforcement learning framework for developing optimized quantum circuit compilation flows.
Through distinct constraints and a unifying interface, the framework supports the combination of techniques
from different compilers and optimization tools in a single compilation flow.
The compilation process is modelled as a Markov Decision Process:

![](https://raw.githubusercontent.com/cda-tum/mqtpredictor/add_RL_extension/img/mdp.png)

In this implementation, compilation passes from both IBM's Qiskit and Quantinuum's TKET are utilized for the RL training
of the optimized compiler.
We trained one RL model for each of the three optimization criteria of expected fidelity, minimal critical depth, and
maximal parallelism.

# Usage of MQT Predictor

First, the package must be installed:

```console
(venv) $ pip install mqt.predictor
```

Now a prediction can be made for any Qiskit::QuantumCircuit object or qasm file:

```python
from mqt.predictor.driver import compile

compiled_qc_ML, predicted_best_device_ML = compile("qasm_file_path", model="ML")
compiled_qc_RL, predicted_best_device_RL = compile(
    "qasm_file_path", model="RL", opt_objective="fidelity"
)
```

In the RL model, the `opt_objective` options are `fidelity`, `critical_depth`, and `parallelism`.

# Examination of all seven trained classifiers of the ML model

To play around with all the examined models, please use the `notebooks/mqt_predictor.ipynb` Jupyter notebook.

## Adjustment of training data generation process

The adjustment of the following parts is possible:

### Compilation Path and Compilation Pipelines

Definition of the to be considered compilation options for

- chosen qubit technologies,
- their respective devices,
- the suitable compilers, and
- their compilation settings.

### Evaluation Metric

To make predictions which compilation options are the best ones for a given quantum circuits, a goodness definition is needed.
In principle, this evaluation metric can be designed to be arbitrarily complex, e.g., factoring in actual costs of executing quantum circuits on the respective platform or availability limitations for certain devices.
However, any suitable evaluation metric should, at least, consider characteristics of the compiled quantum circuit and the respective device.
An exemplary metric could be the overall fidelity of a compiled quantum circuit for its targeted device.

### Generation of Training Data

To train the model, sufficient training data must be provided as qasm files in the `./training_samples_folder`.
We provide the training data used for the pre-trained model.

After the adjustment is finished, the following methods need to be called to generate the training data:

```python
from mqt.predictor import ml

predictor = ml.Predictor()
predictor.generate_compiled_circuits()
res = predictor.generate_trainingdata_from_qasm_files()
ml.helper.save_training_data(res)
```

Now, the Random Forest classifier can be trained:

```python
predictor.train_random_forest_classifier()
```

Additionally, the raw training data may be extracted and can be used for any machine learning model:

```python
(
    X_train,
    X_test,
    y_train,
    y_test,
    indices_train,
    indices_test,
    names_list,
    scores_list,
) = predictor.get_prepared_training_data(save_non_zero_indices=True)
```

# Repository Structure

```
.
├── notebooks/
│ ├── ml/
│ │ ├── ...
│ └── rl/
│     └── ...
├── src/
│ ├── mqt/
│ └── predictor/
│     ├── calibration_files/
│     ├── ml/
│     │ └── training_data/
│     │     ├── trained_model
│     │     ├── training_circuits
│     │     ├── training_circuits_compiled
│     │     └── training_data_aggregated
│     └── rl/
│          └── training_data/
│              ├── trained_model
│              └── training_circuits
└── tests
    ├── ml/
    └── rl/
```

# References

In case you are using MQT Predictor with the ML model in your work, we would be thankful if you referred to it by citing the following publication:

```bibtex
@misc{quetschlich2022mqtpredictor,
  title={Predicting Good Quantum Circuit Compilation Options},
  shorttitle = {{{MQT Predictor}}},
  author={Quetschlich, Nils and Burgholzer, Lukas and Wille, Robert},
  year={2022},
  eprint = {2210.08027},
  eprinttype = {arxiv},
  publisher = {arXiv},
}
```
