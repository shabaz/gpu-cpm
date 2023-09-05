# GPU CPM

Implementation of a GPU-based Cellular Potts Model (CPM) as described in the manuscript

"A parallelized cellular Potts model that enables simulations at tissue scale"

by Shabaz Sultan, Sapna Devi, Scott Mueller, and Johannes Textor.

The framework is implemented in Cuda/C++, but we mainly use it from Python through a wrapper. Therefore we provide this implementation as a Python module.  

## System requirements

The framework requires an Nvidia GPU that supports the Nvidia Cuda framework, version 10.0 or higher.  The size of feasible simulations will depend on the amount of RAM that is available.

To compile the framework, the following software is required:

 * Nvidia Cuda framework (includig nvcc compiler), version 11 or higher
 * Python, version 3.6 or higher
 * numpy, version 1.19 or higher
 * setuptools, version 45.1 or higher

If you use pip to build the module, then numpy and setuptools will be installed automatically while building. The main difficulty is to get the Cuda framework working. If you have never compiled Cuda code before and just want to test the framework, we recommend using Google Colab, which provides all necessary ingredients to compile and test the framework. 

## Installation of the framework on Google Colab

If you don't own a system with an NVIDIA GPU, or you just wish to quickly test the package, Google Colab is a good option. To use the framework from within Google Colab, follow these steps:

 * Open Google Colab https://colab.research.google.com and create a new notebook. Use the "Python 3" runtime and change hardware acceleration to "T4 GPU" (this provides basic GPU support)
 * Install the framework within the Colab session by executing this command:
   `!wget https://computational-immunology.org/gpu-cpm.zip && unzip -q gpu-cpm.zip && pip3 install ./gpu-cpm`
 * Test the installed framework by running the provided examples using
  `!python gpu-cpm/examples/example2d.py` or `!python gpu-cpm/examples/example3d.py`

Installation of the framework on Google Colab should take about 1 minute.

## Installation on your local system

We have tested the steps below using version 12.0 of the Cuda compilation tools on Ubuntu Linux 20.04.6 LTS.

To compile and install the framework on your own system, you will have to have an NVIDIA GPU with the Cuda framework and developer tools set up and working. One way to achieve this is by using Anaconda. For instance, we were able to install the NVIDIA toolchain, compile the framework, and run the examples using the commands below on our Ubuntu 20.04.06 LTS system with an NVIDIA GeForce RTX 2080 GPU. 

The initial solving and installation of the conda environment can take several minutes; compilation and installation of the framework takes about 1 minute on our system, and running the examples should not take longer than a few seconds.

```
# This will create a new environment "gpucpm_build", containing
# a minimal install of the Cuda tools, Python setuptools, numpy (including headers),
# and libraries required to run the examples (pillow and matplotlib with mplot3d extension)
conda env create -f cuda_build_environment.yml

# Activate the new environment
conda activate gpucpm_build

# Change into the build directory; adapt this as necessary.
cd cpu-cpm

# Compile and install the gpucpm package within the new environment
python setup.py install

# Or, if you would like to use pip instead, you could run
# python -m pip install .

# You should now be able to run the provided examples
python example2d.py

python example3d.py
```

## Sorting demo

The file `examples/sorting.py` contains a more detailed example implementation of the classic cell sorting simulation of Graner and Glazier (https://doi.org/10.1103/PhysRevLett.69.2013). Running this simulation should take only a few seconds. The script will produce a png file showing the final state of the simulation.

## Sorting demo on Google Colab

The file `examples/gpucpm_sorting_example.ipynb` contains a Jupyter notebook that can be uploaded to Google Colab (using the menu "File" -> "Upload notebook"). It defines a cell sorting simulation similar to the one in `examples/sorting.py` and ties the simulation to an interactive widget, which allows the simulation progress to be watched within the notebook. 


