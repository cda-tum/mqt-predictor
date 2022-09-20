[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Lint](https://github.com/nquetschlich/MQTPredictor/actions/workflows/linter.yml/badge.svg)](https://github.com/nquetschlich/MQTPredictor/actions/workflows/linter.yml)
[![CodeCov](https://github.com/nquetschlich/MQTPredictor/actions/workflows/coverage.yml/badge.svg)](https://github.com/nquetschlich/MQTPredictor/actions/workflows/coverage.yml)

# MQT Predictor: Automatic Prediction of Good Compilation Paths

MQT Predictor is a framework to create a software tool suggesting a compilation options to use for an arbitrary quantum circuit according to the user's needs.
To this end, we treat the problem as a statistical classification task and apply supervised machine learning to solve it.
Furthermore, the resulting methodology does not only provide end-users with a prediction on the best compilation options,
but additionally provides insights on why certain decisions have been made—allowing them to learn from the predicted results.
For that, a Decision Tree Classifier is used as the machine learning model.

# Exemplary Model for Demonstration Purposes

To adapt this framework to a specific use-case, it must be instantiated as described above.
However, the pre-trained model mentioned described in the experimental evaluations of the corresponding [reference](#reference) is provided.
This can be used to input arbitrary quantum circuits and get a prediction of compilation options based on the used search space
of possible combinations of compilation options.

# Instantiation

For the instantiation, three steps must be followed.

## Compilation Path and Compilation Pipelines

Definition of the to be considered compilation options for

- chosen qubit technologies,
- their respective devices,
- the suitable compilers, and
- their compilation settings.

## Evaluation Metric

To make predictions which compilation options are the best ones for a given quantum circuits, a goodness definition is needed.
In principle, this evaluation metric can be designed to be arbi- trarily complex, e.g., factoring in actual costs of executing quantum circuits on the respective platform or availability limitations for certain devices.
However, any suitable evaluation metric should, at least, consider characteristics of the compiled quantum circuit and the respective device.
An exemplary metric could be the overall fidelity of a compiled quantum circuit for its targeted device.

## Generation of Training Data

To train the model, sufficient training data must be provided.

# Usage Guide

After the model is accordingly instatiated, it can be trained.
Afterwards, predictions for arbitrary quantum circuits can be conducted in real-time.
To extract explicit knowledge, the underlying decision tree can be extracted.

# Repository Structure

# Reference

In case you are using MQT Predictor in your work, we would be thankful if you referred to it by citing the following publication:

```bibtex
@misc{quetschlich2022mqtpredictor,
  title={{{MQT Predictor}}: Automatic Prediction of Good Compilation Paths},
  author={Quetschlich, Nils and Burgholzer, Lukas and Wille, Robert},
  year={2022},
}
```
