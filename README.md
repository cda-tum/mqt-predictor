[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Lint](https://github.com/nquetschlich/MQTPredictor/actions/workflows/linter.yml/badge.svg)](https://github.com/nquetschlich/MQTPredictor/actions/workflows/linter.yml)
[![CodeCov](https://github.com/nquetschlich/MQTPredictor/actions/workflows/coverage.yml/badge.svg)](https://github.com/nquetschlich/MQTPredictor/actions/workflows/coverage.yml)
[![Deploy to PyPI](https://github.com/cda-tum/MQTPredictor/actions/workflows/deploy.yml/badge.svg)](https://github.com/cda-tum/MQTPredictor/actions/workflows/deploy.yml)

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
- Multilayer Perceptron
- Support Vector Machine
- Naive Bayes

In our exemplary scenario, the Random Forest classifier achieved the best performance.

This software comprises three main functionalities:

- The pre-trained Random Forest classifier to easily predict compilation options for an unseen quantum circuit
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
from mqt.predictor.driver import Predictor

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
from mqt.predictor.driver import Predictor

predictor = Predictor()
compiled_qasm_str = predictor.compile_predicted_compilation_path(
    "qasm_file_path", prediction_index
)
```

# Examination of all seven trained classifiers

To play around with all the examined models, please use the `notebooks/mqt_predictor.ipynb` Jupyter notebook.

# Adjustment of training data generation process

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
from mqt.predictor.driver import Predictor
from mqt.predictor import utils

predictor = Predictor()
predictor.generate_compiled_circuits()
utils.postprocess_ocr_qasm_files()
res = predictor.generate_trainingdata_from_qasm_files()
utils.save_training_data(res)
```

Now, the Random Forest classifier can be trained:

```python
predictor.train_random_forest_classifier()
```

Additionally, the raw training data may be extracted and can be used for any machine learning model:

```python
from mqt.predictor import utils
import numpy as np

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
|-- notebooks
|   |-- runtime_comparison.ipynb
|   |-- mqt_predictor.ipynb
|   |-- results/
|-- src
|   |-- mqt
|       `-- predictor
|           |-- driver.py
|           |-- utils.py
|           |-- calibration_files/
|           |-- training_data/
|           |-- training_samples/
|           `-- training_samples_compiled/
`-- tests/
```
