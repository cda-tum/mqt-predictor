import os

from setuptools import find_namespace_packages, setup
from setuptools_scm import get_version

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()


version = get_version(root=".", relative_to=__file__)


setup(
    name="mqt.predictor",
    packages=find_namespace_packages(include=["*mqt.*"]),
    package_dir={"mqt.predictor": "src"},
    setup_requires=["setuptools_scm"],
    include_package_data=True,
    python_requires=">=3.8",
    license="MIT",
    description="MQT Predictor",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Nils Quetschlich",
    author_email="nils.quetschlich@tum.de",
    url="https://github.com/cda-tum/mqtpredictor",
    install_requires=[
        "qiskit~=0.35",
        "pytket~=1.1",
        "numpy>=1.21.5,<1.24.0",
        "mqt.bench~=0.1.3",
        "pytket-qiskit>=0.24,<0.28",
        "matplotlib~=3.5.1",
        "scikit-learn>=1.0.2,<1.2.0",
        "natsort~=8.1.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
)
