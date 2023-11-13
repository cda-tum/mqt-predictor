Compilation: Device-specific Quantum Circuit Compiler Using Reinforcement Learning Model
========================================================================================



Compilation, fortunately, is not new per-se, since classical compilers have seen a similar trend of an increasing complexity and variety in the past.
To not reinvent the wheel and make use of the decades of classical compiler optimization, quantum compilation is modeled in a similar fashion and classical reinforcement learning is used to predict compilation pass sequences optimizing for the chosen figure of merit.

Through distinct constraints and a unifying interface, the framework supports the combination of techniques
from different compilers and optimization tools in a single compilation flow.
The compilation process is modelled as a Markov Decision Process and takes three inputs:

1. Training circuits
2. The targeted quantum device
3. The figure of merit to optimize for


.. image:: /_static/rl.png
   :width: 100%
   :alt: Illustration of the RL model
   :align: center

The trained reinforcement learning model then acts as a compiler and can be used to compile any quantum circuit for the targeted device.

In this implementation, compilation passes from both IBM's Qiskit and Quantinuum's TKET are utilized for the RL training
of the optimized compiler.
We trained one RL model for each currently supported quantum device:

- OQC Lucy with 8 qubits
- IonQ Harmony with 11 qubits
- IonQ Aria1 with 25 qubits
- IBM Montreal with 27 qubits
- Quantinuum H2 with 32 qubits
- Rigetti Aspen-M2 with 80 qubits
- IBM Washington with 127 qubits