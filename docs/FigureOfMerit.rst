Figure of Merit
================

The figure of merit determines the optimization criteria for both the ML and RL model and, thus, is a crucial part of the MQT Predictor framework.
In general, it can be completely customizable and consider anything from (compiled)
circuit characteristics such as gate counts to actual execution costs by a specific vendor. Nevertheless,
a comprehensive figure of merit should consider at least the characteristics of the compiled quantum
circuit and the characteristics of the selected quantum device. Therefore, the respective hardware
information needs to be collected, such as, e.g., gate/readout fidelities, gate execution times, or
decoherence times. While for some devices, this information may be publicly available, for other
devices it may be estimated from comparable devices, previous records, or insider knowledge.

So far, two figures of merit are implemented for all devices: ``expected_fidelity`` and ``critical_depth``.
On top of that, the ``estimated_success_probability`` is available for devices where the necessary calibration data is available.

Expected Fidelity
-----------------

Consider the figure of merit that takes into account the following two aspects:

- If the selected device is not large enough to execute a given quantum circuit with respective to its number of qubits, the worst-possible score is assigned.
- If the device is large enough, the evaluation score is calculated using the formula:

.. math::
    \mathit{\mathcal{F}}=\prod_{i=1}^{|G|} \mathit{\mathcal{F}}(g_i) \prod_{j=1}^{m} \mathit{\mathcal{F}_{RO}}(q_j)

with :math:`\mathit{\mathcal{F}}(g_i)` being the expected execution fidelity of gate :math:`g_i` on its corresponding qubit(s),
:math:`\mathit{\mathcal{F}_{RO}}(q_j)` being the expected execution fidelity of a measurement operation :math:`q_j` on its corresponding qubit and :math:`|G|` respectively :math:`m` being the number of gates and measurements in the compiled circuit.


This figure of merit determines an estimate of the probability that a quantum circuit will return the expected result, the so-called ``expected fidelity``, which ranges between :math:`0.0` and :math:`1.0` with higher values being better.


Critical Depth
--------------
A potential alternative could be the ``critical depth`` (taken from :cite:labelpar:`tomesh2022supermarq`)---a measure to describe the percentage of multi-qubit gates on the longest path through a compiled quantum circuit (determining the depth).
A respective value close to 1 would indicate a very sequential circuit while a value of 0 would indicate a highly parallel one.


Estimated Success Probability
-----------------------------
The ``estimated_success_probability`` (based on :cite:labelpar:`esp-lifetime-min` and :cite:labelpar:`esp-idle`) is a figure of merit that is based on the ``expected_fidelity`` but also multiplies it with a factor that considers the decoherence times :math:`T_1, T_2` of a device:

.. math::
   \prod_{q} \exp{(t_{q}^{\mathrm{idle}}/\min{(T_1, T_2)})}

with :math:`t_{q}^{\mathrm{idle}}` being the sum of each qubit's idle times.
Therefore, exactly the execution times of all gates and the decoherence times must be available.
Note that some variants of this figure of merit do not take the minimum of both decoherence times but create one exponential factor for each decoherence time, others consider the entire qubit lifetime instead of idle times only.
