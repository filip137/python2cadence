import numpy as np

# Define the parameters
A = 4
B = 1/4

# Specific values for the g_ij elements
g_0_11 = 1.0
g_0_12 = 1/2
g_0_21 = 1/3
g_0_22 = 1/4
g_1_11 = 1/5
g_1_12 = 1/6
g_1_21 = 1/7
g_1_22 = 1/8

# Create the G matrix using NumPy
G = np.array([
    [g_0_11 + A * B * g_1_11 + A * B * g_1_12 + g_0_21, 0.0, -(B * g_1_11), -(B * g_1_12), -g_0_11],
    [0.0, g_0_22 + A * B * g_1_21 + A * B * g_1_22 + g_0_12, -B * g_1_21, -B * g_1_22, -g_0_12],
    [-A * g_1_11, -A * g_1_21, g_1_11 + g_1_21, 0.0, 0.0],
    [-A * g_1_12, -A * g_1_22, 0.0, g_1_12 + g_1_22, 0.0],
    [g_0_11, g_0_12, 0.0, 0.0, -g_0_11 - g_0_12]
])

# Define the nudge constraint
I_nudge = 0

# Extract G_ff and the constraint row G_fv
G_ff = G[:-1, :-1]
G_fv = G[:-1, -1]
G_vf = G[-1, :-1]  # Last row except the fixed variable column

# The constraint we want to enforce is G[3, :] @ e = I_nudge
constraint_row = G[3, :-1]

# Combine G_ff and the constraint
K = np.vstack([np.hstack([2 * G_ff, constraint_row.reshape(-1, 1)]),
               np.hstack([constraint_row.reshape(1, -1), np.array([[0]])])])

# Right-hand side vector
b = np.hstack([-2 * G_fv * 3.0, I_nudge])  # Here, 3.0 is the fixed value for e_5

# Solve the linear system
solution = np.linalg.solve(K, b)

# Extract the solution
e_f_opt = solution[:-1]
lambda_opt = solution[-1]

# Combine the free variables and fixed variable to get the full solution
e_opt = np.append(e_f_opt, 3.0)

# Calculate the minimum value of the quadratic form
min_value = e_opt.T @ G @ e_opt

print("Optimal values of free variables (e_f):", e_f_opt)
print("Full solution (e):", e_opt)
print("Minimum value of the quadratic form:", min_value)
print("Optimal lambda:", lambda_opt)
