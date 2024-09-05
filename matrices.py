import numpy as np
import torch
from scipy.optimize import minimize

# Given the matrix G (example values)
A = 10
B = 1
g_0_11, g_0_12, g_0_21, g_0_22 = 4.0, 1/2, 1/3, 1/4
g_1_11, g_1_12, g_1_21, g_1_22 = 1/5, 10, 1/7, 1/8

# Define the fixed value for e_5
v_x = 3.0

# Define the matrix G with the updated values

G = np.array([
    [g_0_11 + A * B * g_1_11 + A * B * g_1_12 + g_0_21, 0, -(B * g_1_11), -(B * g_1_12), -g_0_11],
    [0, g_0_22 + A * B * g_1_21 + A * B * g_1_22 + g_0_12, -B * g_1_21, -B * g_1_22, -g_0_12],
    [-A * g_1_11, -A * g_1_21, g_1_11 + g_1_21, 0, 0],
    [-A * g_1_12, -A * g_1_22, 0, g_1_12 + g_1_22, 0],
    [-g_0_11, -g_0_12, 0, 0, g_0_11 + g_0_12]
])

G_a = np.array([
    [g_0_11 + A * A * g_1_11 + A * A * g_1_12 + g_0_21, 0, -(A * g_1_11), -(A * g_1_12), -g_0_11],
    [0, g_0_22 + A * A * g_1_21 + A * A * g_1_22 + g_0_12, -A * g_1_21, -A * g_1_22, -g_0_12],
    [-A * g_1_11, -A * g_1_21, g_1_11 + g_1_21, 0, 0],
    [-A * g_1_12, -A * g_1_22, 0, g_1_12 + g_1_22, 0],
    [-g_0_11, -g_0_12, 0, 0, g_0_11 + g_0_12]
])

G_b = np.array([
    [g_0_11 + A * B * g_1_11 + A * B * g_1_12 + g_0_21, 0, -(B * g_1_11), -(B * g_1_12), -g_0_11],
    [0, g_0_22 + A * B * g_1_21 + A * B * g_1_22 + g_0_12, -B * g_1_21, -B * g_1_22, -g_0_12],
    [-A * g_1_11 , -A * g_1_21, (g_1_11 + g_1_21), 0, 0],
    [-A * g_1_12, -A * g_1_22, 0, (g_1_12 + g_1_22), 0],
    [-g_0_11, -g_0_12, 0, 0, g_0_11 + g_0_12]
])


G_b = np.array([
    [g_0_11 + A  * g_1_11 + A * g_1_12 + g_0_21, 0, -( g_1_11), -(g_1_12), -g_0_11],
    [0, g_0_22 + A * g_1_21 + A * g_1_22 + g_0_12, -g_1_21, -g_1_22, -g_0_12],
    [-g_1_11 , -g_1_21, (g_1_11 + g_1_21)/A, 0, 0],
    [-g_1_12, -g_1_22, 0, (g_1_12 + g_1_22)/A, 0],
    [-g_0_11, -g_0_12, 0, 0, g_0_11 + g_0_12]
])


G_vv = G[:4, :4]
G_a_vv = G_a[:4, :4]
G_b_vv = G_b[:4, :4]

# Calculate the inverses
G_vv_inv = np.linalg.inv(G_vv)
G_a_vv_inv = np.linalg.inv(G_a_vv)
G_b_vv_inv = np.linalg.inv(G_b_vv)


G_c = G.copy()
G_c[0] *= 1
G_c[1] *= 1 
G_c[2] *= B/A
G_c[3] *= B/A
G_c[4] *= 1

# Extract the submatrix (first 4x4 part of G_c)
G_c_vv = G_c[:4, :4]

# Calculate the determinant of the 4x4 submatrix of G_c
det_G_c_vv = np.linalg.det(G_c_vv)

G_inv = np.linalg.inv(G)
G_c_inv = np.linalg.inv(G_c)
G_b_inv = np.linalg.inv(G_b)
G_sub = G_c[0:4,0:4]
e_free2 = - np.dot(np.linalg.inv(G_sub), v_x * G_c[4,0:4])
# Define the objective function
def objective(e):
    return e.T @ G_c @ e

# Define the constraint: e[4] = 3
def constraint(e):
    return e[4] - 3

# Initial guess for the optimization
initial_guess = np.zeros(5)

# Set up the constraints
constraints = [{'type': 'eq', 'fun': constraint}]

# Perform the minimization
result = minimize(objective, initial_guess, constraints=constraints)

# Output the optimized error vector
optimized_e = result.x

print("Optimized e:", optimized_e)
print("Minimum value of e^T G e:", objective(optimized_e))


G_inv = np.linalg.inv(G)
#Define the source vector
i_nudge = 0
i_eq = (v_x-G_inv[4,3]*i_nudge)/G_inv[4,4]
i_s = np.array([0.0, 0.0, 0.0, i_nudge, 0])
i_eq_test = np.array([0.0, 0.0, 0.0, i_nudge, i_eq])
def func(e_free):
    e = np.hstack((e_free, [v_x]))  # Stack the free variables with the fixed e_5
    e = e.reshape(5, 1)  # Reshape to 5x1 for matrix operations
    #return (e.T @ G @ e - 2 * e.T @ i_s).item()
    return (e.T @ G @ e - 2 * e.T @ i_s).item()

# Initial guess for the free components of e (e_1, e_2, e_3, e_4)
e_free_initial = np.zeros(4)

# Perform the minimization
result = minimize(func, e_free_initial)

# Combine the free components with the fixed e_5
e_optimal = np.hstack((result.x, v_x))

# Output the result
print("Optimal values of free variables (e_1 to e_4):", result.x)
print("Optimal complete vector e:", e_optimal)
print("Minimum value of the functional:", result.fun)

# # Partition G into submatrices
# G_ff = G[:-1, :-1]  # Interaction of free variables
# G_fv = G[:-1, -1]  # Interaction between free and fixed variables
# G_vf = G[-1, :-1]  # Same as G_fv due to symmetry

# # Solve the system to minimize the quadratic form
# e_f_opt = np.linalg.solve(G_ff, G_fv * v_x)

# # The full solution e
# e_opt = np.append(e_f_opt, v_x)

# The minimum value of the quadratic form
#min_value = e_opt.T @ G @ e_opt

#print("Optimal values of free variables (e_f):", result)
#print("Full solution (e):", e_opt)
#print("Minimum value of the quadratic form:", min_value)