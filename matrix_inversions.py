#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:20:39 2024

@author: filip
"""

import numpy as np
from sympy import symbols, Matrix

# Define symbols
g11, A, B, g12, g21, g22, g31, g32, g41, g42 = symbols('g11 A B g12 g21 g22 g31 g32 g41 g42')
np.random.seed(2)
# Random values for g11 to g42
g_values = np.random.randint(1, 10, size=3)  # Values from 1 to 10
#g12, g21, g22, g31, g32, g41, g42, AB, vx, i_n, i_nn = g_values
vx, i_n, i_nn = g_values
i_n, i_nn = 0, 0
# Define the matrix of coefficients using the random values and symbolic A and B
M = Matrix([
    [-g11 - A*B*g31 - A*B*g32 - g21, 0, B*g31, B*g41],
    [0, -g22 - A*B*g41 - A*B*g42 - g12, B*g41, B*g42],
    [A*g31, A*g41, -(g31 + g41), 0],
    [A*g41, A*g42, 0, -(g32 + g42)]
])

# Compute the inverse of the matrix M
M_inv = M.inv()

# Display the inverted matrix
print("Matrix M (Inverted):")
print(M_inv)