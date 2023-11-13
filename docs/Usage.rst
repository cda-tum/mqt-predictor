Repository Usage
================
There are two ways how to use MQT Predictor:

#. Via the pip package ``mqt.predictor``
#. Directly via this repository

Usage via pip package
---------------------

MQT predictor is available via `PyPI <https://pypi.org/project/mqt.predictor/>`_

.. code-block:: console

   (venv) $ pip install mqt.predictor

To compile a quantum circuit, use the ``qcompile`` method:

.. automodule:: mqt.predictor
    :members: qcompile

Currently available figures of merit are `expected_fidelity` and `critical_depth`.

An example how `qcompile` is used can be found in the :doc:`quickstart <Quickstart>` jupyter notebook.

Usage directly via this repository
----------------------------------

For that, the repository must be cloned and installed:

.. code-block::

   git clone https://github.com/cda-tum/MQTpredictor.git
   cd MQTpredictor
   pip install .

Afterwards, the package can be used as described `above <#Usage via pip package>`_.
