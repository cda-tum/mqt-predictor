Device Selection: Automatic Device Selection Using Supervised Machine Learning
===============================================================================

A naive approach to select the best quantum device for a given quantum circuit would be to compile it for all devices, e.g., using the trained RL models which act as specialized compilers for supported quantum devices.
Afterwards, the resulting compiled circuits must be evaluated according to some figure of merit to identify the most promising device.
However, doing this for each and every to-be-compiled quantum circuit is practically infeasible since compilation is a time-consuming task.

The MQT Predictor learns from previous compilations of other quantum circuits and models the problem of determining the most promising device for a circuit and figure of merit as a statistical classification task—a task well suited for supervised machine learning.
For that, the framework is trained with based on three inputs:

1. Training circuits
2. The compilation options for all supported devices
3. The figure of merit to optimize for

.. image:: /_static/ml.png
   :width: 100%
   :alt: Illustration of the ML model
   :align: center

The trained model then acts as a predictor and can be used to predict the most suitable device for a given quantum circuit and figure of merit.

For evaluation of our methodology, seven supervised machine learning classifiers have been used:

- Random Forest
- Gradient Boosting
- Decision Tree
- Nearest Neighbor
- Multilayer Perceptron
- Support Vector Machine
- Naive Bayes

In our exemplary scenario, the Random Forest classifier achieved the best performance.

Examination of all seven trained classifiers of the ML model
------------------------------------------------------------

To play around with all the examined models, please use the `evaluations/ml/evaluation.ipynb` Jupyter notebook.

Generation of Training Data
---------------------------

To train the model, sufficient training data must be provided as qasm files in the `./training_samples_folder`.
We provide the training data used for the pre-trained model.

After the adjustment is finished, the following methods need to be called to generate the training data:

.. code-block:: python

    from mqt.predictor import ml

    predictor = ml.Predictor()
    predictor.generate_compiled_circuits(figure_of_merit="expected_fidelity")
    training_data, name_list, scores_list = predictor.generate_trainingdata_from_qasm_files(
        figure_of_merit="expected_fidelity"
    )
    ml.helper.save_training_data(
        training_data, name_list, scores_list, figure_of_merit="expected_fidelity"
    )

Now, the Random Forest classifier can be trained:

.. code-block:: python

    predictor.train_random_forest_classifier(figure_of_merit="expected_fidelity")


Additionally, the raw training data may be extracted and can be used for any machine learning model:

.. code-block:: python

    training_data = predictor.get_prepared_training_data(
        save_non_zero_indices=True, figure_of_merit="expected_fidelity"
    )
