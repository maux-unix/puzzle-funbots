"""
puzzle_funbots_classic.py
Classic control and simulation of FunBots robots using PyBullet.

Usage:
    python3 puzzle_funbots_classic.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
import pybullet_data as pybd

GUI = True

if GUI:
    physics_client = pyb.connect(pyb.GUI)
else:
    physics_client = pyb.connect(pyb.DIRECT)


