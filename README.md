[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CodeCov](https://github.com/cda-tum/mqt-predictor/actions/workflows/coverage.yml/badge.svg)](https://github.com/cda-tum/mqt-predictor/actions/workflows/coverage.yml)
[![Deploy to PyPI](https://github.com/cda-tum/mqt-predictor/actions/workflows/deploy.yml/badge.svg)](https://github.com/cda-tum/mqt-predictor/actions/workflows/deploy.yml)
[![codecov](https://codecov.io/gh/cda-tum/mqt-predictor/branch/main/graph/badge.svg?token=ZL5js1wjrB)](https://codecov.io/gh/cda-tum/mqt-predictor)
[![Documentation](https://img.shields.io/readthedocs/mqt-predictor?logo=readthedocs&style=flat-square)](https://mqt.readthedocs.io/projects/predictor)

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mqtpredictor/main/docs/_static/mqt_light.png" width="60%">
  <img src="https://raw.githubusercontent.com/cda-tum/mqtpredictor/main/docs/_static/mqt_dark.png" width="60%">
</picture>
</p>

# MQT Predictor: Automatic Device Selection with Device-Specific Circuit Compilation for Quantum Computing

MQT Predictor is a framework that allows one to automatically select a suitable quantum device for a particular application and provides an optimized compiler for the selected device.
It not only supports end-users in navigating the vast landscape of choices, it also allows to mix-and-match compiler passes from various tools to create optimized compilers that transcend the individual tools.
Evaluations on more than 500 quantum circuits and seven devices have shown that—compared to Qiskit's and TKET's most optimized compilation flows—the MQT Predictor yields circuits with an expected fidelity that is on par with the best possible result that could be achieved by trying out all combinations of devices and compilers and even achieves a similar performance when considering the critical depth as an alternative figure of merit.

Therefore, MQT Predictor tackles this problem from two angles:

1. It provides a method (based on Reinforcement Learning) that produces device-specific quantum circuit compilers by combining compilation passes from various compiler tools and learning optimized sequences of those passes with respect to a customizable figure of merit). This mix-and-match of compiler passes from various tools allows one to eliminate vendor locks and to create optimized compilers that transcend the individual tools.

2. It provides a prediction method (based on Supervised Machine Learning) that, without performing any compilation, automatically predicts the most suitable device for a given application. This completely eliminates the manual and laborious task of determining a suitable target device and guides end-users through the vast landscape of choices without the need for quantum computing expertise.

<p align="center">
<picture>
  <img src="docs/_static/problem.png" width="100%">
</picture>
</p>

For more details, please refer to:

<p align="center">
  <a href="https://mqt.readthedocs.io/projects/predictor">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

MQT Predictor is part of the Munich Quantum Toolkit (MQT) developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and is hosted at [https://www.cda.cit.tum.de/mqtbench/](https://www.cda.cit.tum.de/mqtbench/).

If you have any questions, feel free to create a [discussion](https://github.com/cda-tum/mqt-predictor/discussions) or an [issue](https://github.com/cda-tum/mqt-predictor/issues) on [GitHub](https://github.com/cda-tum/mqt-predictor).

MQT Predictor is part of the Munich Quantum Toolkit (MQT) developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/).

## Getting Started

`mqt-predictor` is available via [PyPI](https://pypi.org/project/mqt.predictor/).

```console
(venv) $ pip install mqt.predictor
```

The following code gives an example on the usage:

```python3
from mqt.predictor import qcompile
from mqt.bench import get_benchmark

# get a benchmark circuit on algorithmic level representing the GHZ state with 5 qubits from [MQT Bench](https://github.com/cda-tum/mqt-bench)
qc_uncompiled = get_benchmark(benchmark_name="dj", level="alg", circuit_size=5)

# compile it using the MQT Predictor
qc_compiled, compilation_information, quantum_device = qcompile(qc_uncompiled)

# print the selected device and the compilation information
print(quantum_device, compilation_information)

# draw the compiled circuit
print(qc_compiled.draw())
```

**Detailed documentation and examples are available at [ReadTheDocs](https://mqt.readthedocs.io/projects/predictor).**

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European
Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement
No. 101001318), the Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, as well as the
Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mqt-predictor/main/docs/_static/tum_dark.svg" width="28%">
<img src="https://raw.githubusercontent.com/cda-tum/mqt-predictor/main/docs/_static/tum_light.svg" width="28%">
</picture>
<picture>
<img src="https://raw.githubusercontent.com/cda-tum/mqt-predictor/main/docs/_static/logo-bavaria.svg" width="16%">
</picture>
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mqt-predictor/main/docs/_static/erc_dark.svg" width="24%">
<img src="https://raw.githubusercontent.com/cda-tum/mqt-predictor/main/docs/_static/erc_light.svg" width="24%">
</picture>
<picture>
<img src="https://raw.githubusercontent.com/cda-tum/mqt-predictor/main/docs/_static/logo-mqv.svg" width="28%">
</picture>
</p>
