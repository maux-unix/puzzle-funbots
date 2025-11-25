# Puzzle-Funbots Robotic Arm Simulation

This repository contains the robotic arm simulation that solves 4x4 puzzle.
It is our project for *Data-driven Control System* college class.

## Overview

[TODO: Insert simulation image]

## Prerequisites

Make sure you have the following softwares installed on your system:

- Anaconda
- Python 3.13+
- ipykernel (install with pip)

## Getting Started

Follow the following commands to getting started:

```shell
conda init $SHELL
conda config --add channels conda-forge
conda env create -f environment.yml
conda activate puzzle-funbots
python3 puzzle_funbots_data-driven.py
python3 puzzle_funbots_classic.py
```

## For ILC + NN

```shell
conda init $SHELL
conda config --add channels conda-forge
conda create -n funbots_baru python=3.10
conda activate funbots_baru
conda install pybullet
pip install numpy matplotlib
conda install intel-openmp
pip install torch torchvision torchaudio
```
