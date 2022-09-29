[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Lint](https://github.com/nquetschlich/MQTPredictor/actions/workflows/linter.yml/badge.svg)](https://github.com/nquetschlich/MQTPredictor/actions/workflows/linter.yml)
[![CodeCov](https://github.com/nquetschlich/MQTPredictor/actions/workflows/coverage.yml/badge.svg)](https://github.com/nquetschlich/MQTPredictor/actions/workflows/coverage.yml)

# MQT Predictor: Automatic Prediction of Good Compilation Paths

MQT Predictor is a framework suggesting a compilation options to use for an arbitrary quantum circuit according to the user's needs.
To this end, we treat the problem as a statistical classification task and apply supervised machine learning to solve it.
Furthermore, the resulting methodology does not only provide end-users with a prediction on the best compilation options,
but additionally provides insights on why certain decisions have been made—allowing them to learn from the predicted results.

For evaluation of our methodology, seven supervised machine learning classifiers have been used:

- Random Forest
- Gradient Boosting
- Decision Tree
- Nearest Neighbor
- Support Vector Machine (SVM)
- Naive Bayes
- Stochastic Gradient Descent (SGD)

In our exemplary scenario, the Random Forest classifier achieved the best performance.

This software comprises three main functionalities:

- The trained Random Forest classifier to easily predict compilation options for an unseen quantum circuit
  in real-time and compile for the respective prediction,
- all other trained algorithms, and
- the possibility to adjust and customize the whole training data generation process, e.g., to add training data, compilation options, or adapt the evaluation function.

# Usage of Random Forest pre-trained Classifier

First, the package must be installed:

```console
(venv) $ pip install mqt.predictor
```

Now a prediction can be made for any qasm file:

```python
from mqt.predictor import Predictor

predictor = Predictor()
prediction_index = predictor.predict("qasm_file_path")
```

This prediction index can be translated into a tuple of (gate set, device, compiler, compiler_settings):

```python
from mqt.predictor import utils

look_up_table = utils.get_index_to_comppath_LUT()
prediction_tuple = look_up_table[prediction_index]
print(prediction_tuple)
```

Afterwards, the circuit can be compiled respectively and the compiled circuit is returned as a qasm string:

```python
from mqt.predictor import Predictor

compiled_qasm_str = predictor.compile_predicted_compilation_path(
    "qasm_file_path", prediction_index
)
```

# Examination of all seven trained classifiers

To play a round with the models examined by us, please use the provided Jupyter notebook `mqt_predictor.ipynb`.

# Adjustment of training data generation process

The adjustment of all three parts is possible and described in the following:

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
from mqt.predictor import Predictor

predictor = Predictor()
predictor.generate_compiled_circuits(
    source_path="./training_samples",
    target_path="./training_samples_compiled",
    timeout=120,
)
utils.postprocess_ocr_qasm_files(directory="./training_samples_compiled")
res = predictor.generate_trainingdata_from_qasm_files(
    source_path="./training_samples", target_path="./training_samples_compiled/"
)
utils.save_training_data(res)
```

After the training data is saved, it can be loaded and used for any machine learning model:

```python
import numpy as np

training_data, names_list, scores_list = utils.load_training_data()
X, y = zip(*training_data)
X = list(X)
y = list(y)
for i in range(len(X)):
    X[i] = list(X[i])
    scores_list[i] = list(scores_list[i])

X, y = np.array(X), np.array(y)
```

# Repository Structure

```
MQTPredictor/
│ - README.md
│ - mqt_predictor.ipynb
│
└───mqt/predictor/
    │───benchmark_generator.py
    │───driver.py
    └───tests/
    │   └───...
    └───calibration files/
    │    └───...
    └───src/
        │ - utils.py
```

# Reference

In case you are using MQT Predictor in your work, we would be thankful if you referred to it by citing the following publication:

```bibtex
@misc{quetschlich2022mqtpredictor,
  title={{{MQT Predictor}}: Automatic Prediction of Good Compilation Paths},
  author={Quetschlich, Nils and Burgholzer, Lukas and Wille, Robert},
  year={2022},
}
```
