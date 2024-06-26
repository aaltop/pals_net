# PALS Tool

A tool to train a PyTorch Convolutional Neural Network (CNN) on simulated 
Positron Annihilation Lifetime Spectroscopy (PALS) spectra to be able 
to predict the components of real PALS experiments' spectra.

## Description

This tool was created to assist scientists in the analysis of spectra
that are the result of Positron Annihilation Lifetime Spectroscopy.
The task of analysing PALS spectra was thought to be time-consuming 
to do manually (by the use of programs such as [PALSfit](http://palsfit.dk/)) 
and a process that could be made more efficient by more recent machine
learning approaches.

The general idea of the approach the tool takes is to simulate many
PALS spectra based on potential features of a spectrum, and train
the CNN on the simulated data. The training data
could just as well be data from real experiments, but the amount and
variety of data needed to train a good model is not necessarily
readily available.

In its current state, the tool has been tested on simulated data and
found to be potentially viable for analysing PALS spectra.

## Getting Started

Usage can be a little involved, but should not be too much so. Usage
will require a Python installation along with the below dependencies
and likely some Python knowledge.

### Dependencies

The tool was written in Python 3.11.5. Dependencies are

- [PyTorch](https://pytorch.org/get-started/locally/), recommended
installation through [conda environment and package manager](https://docs.conda.io/en/latest/miniconda.html).

- [NumPy](https://numpy.org/)

- [Matplotlib](https://matplotlib.org/)

- [SciPy](https://scipy.org/)

### Usage

Once all the dependencies are installed, there are essentially three
steps to the process: 

1. Simulation data creation, handled in the file **simulate_spectra.py**.
(see simulate_spectra_example.py for examples.)

2. Training the model, handled in the file **train.py**. (See train_example.py
for examples.)

3. Evaluating the model, handled in the file **evaluate.py**.

Each file has a module docstring that contains instructions for running 
the file, and where to look if changes to the process are wanted. A quick
look through the files is recommended before running anything, but 
nothing entirely unexpected should happen. The file **example.ipynb**
Gathers all the steps into one file.
