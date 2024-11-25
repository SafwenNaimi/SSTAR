# SSTAR

![alt text](https://github.com/SafwenNaimi/SSTAR/tree/blob/main/architecture.png)

This repository contains the official implementation for the paper "SSTAR: Skeleton-based Spatio-temporal action recognition for intelligent video surveillance and suicide prevention in metro stations"

# Installation

* tensorflow 2.6.0
* Python 3.8.5
* numpy 1.23.5

Clone this repository.

    git clone git@github.com:GIT-USERNAME/SSTAR.git
    cd SSTAR

Clone the repository and install the required pip packages (We recommend a virtual environment):

    pip install -r requirements.txt

# Data preparation for our ARMM dataset:

We provide the skeleton ground truth annotations of our ARMM dataset, you can directly download them and use them for training & testing.

# Training:
To run a baseline SSTAR experiment:

    python main.py -b 
