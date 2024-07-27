import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve

# Define symbols
x1, x2, x3, x4, g11, g12, g21, g22, g31, g32, g41, g42, vx = symbols('x1 x2 x3 x4 g11 g12 g21 g22 g31 g32 g41 g42 vx')

# Define a function to solve the system numerically for given values and return x1, x3
def solve_system_x1_x3(A_value, B_value, g11_value, i_n_value, i_nn_value):
    eq1 = Eq((-g11_value - A_value*B_value*g31 - A_value*B_value*g32 - g21)*x1 + B_value*g31*x3 + B_value*g41*x4, -vx*g11_value)
    eq2 = Eq((-g22 - A_value*B_value*g41 - A_value*B_value*g42 - g12)*x2 + B_value*g41*x3 + B_value*g42*x4, -vx*g12)
    eq3 = Eq(A_value*g31*x1 + A_value*g41*x2 - (g31 + g41)*x3, i_n_value)
    eq4 = Eq(A_value*g32*x1 + A_value*g42*x2 - (g32 + g42)*x4, i_nn_value)

    # Substitute constants for all remaining variables
    random_values = {
        g12: 3, g21: 4, g22: 5,
        g31: 1, g32: 2, g41: 3, g42: 4,
        vx: 1
    }
    
    eq1 = eq1.subs(random_values)
    eq2 = eq2.subs(random_values)
    eq3 = eq3.subs(random_values)
    eq4 = eq4.subs(random_values)
    
    solution = solve((eq1, eq2, eq3, eq4), (x1, x2, x3, x4))
    return solution[x1], solution[x3]

# Define a function to solve the system numerically for given values and return x3
def solve_system_x3(A_value, B_value, g11_value, i_n_value, i_nn_value):
    eq1 = Eq((-g11_value - A_value*B_value*g31 - A_value*B_value*g32 - g21)*x1 + B_value*g31*x3 + B_value*g41*x4, -vx*g11_value)
    eq2 = Eq((-g22 - A_value*B_value*g41 - A_value*B_value*g42 - g12)*x2 + B_value*g41*x3 + B_value*g42*x4, -vx*g12)
    eq3 = Eq(A_value*g31*x1 + A_value*g41*x2 - (g31 + g41)*x3, i_n_value)
    eq4 = Eq(A_value*g32*x1 + A_value*g42*x2 - (g32 + g42)*x4, i_nn_value)

    # Substitute constants for all remaining variables
    random_values = {
        g12: 3, g21: 4, g22: 5,
        g31: 1, g32: 2, g41: 3, g42: 4,
        vx: 1
    }
    
    eq1 = eq1.subs(random_values)
    eq2 = eq2.subs(random_values)
    eq3 = eq3.subs(random_values)
    eq4 = eq4.subs(random_values)
    
    solution = solve((eq1, eq2, eq3, eq4), (x1, x2, x3, x4))
    return solution[x3]

# Parameters
A_value = 1
B_value = 1
g11_value = 2
vx_value = 1
h = 1e-6  # increment for finite difference

# Compute the derivative d(x_3^2)/dg_11
x3_g11 = solve_system_x3(A_value, B_value, g11_value, 0, 0)
x3_g11_h = solve_system_x3(A_value, B_value, g11_value + h, 0, 0)
derivative_x3_squared = (x3_g11_h**2 - x3_g11**2) / h

# Range of beta values
beta_values = np.linspace(1e-8, 1e-4, 100)
expression_differences = []

# Compute the difference for each beta
for beta in beta_values:
    # Calculate x3 with i_n = A * beta * x3 and i_nn = 0
    _, x3_zero = solve_system_x1_x3(A_value, B_value, g11_value, 0, 0)
    x1_vn, _ = solve_system_x1_x3(A_value, B_value, g11_value, A_value * beta * x3_zero, 0)
    Vn = vx_value - x1_vn

    # Calculate Vf with i_n = 0 and i_nn = 0
    x1_vf, _ = solve_system_x1_x3(A_value, B_value, g11_value, 0, 0)
    Vf = vx_value - x1_vf

    # Evaluate the expression 1/beta * (Vn^2 - Vf^2)
    expression_value = (Vn**2 - Vf**2) / beta

    # Compute the difference
    difference = expression_value - derivative_x3_squared
    expression_differences.append(difference)

# Plotting the differences
plt.figure(figsize=(10, 6))
plt.plot(beta_values, expression_differences, label='Difference')
plt.xlabel('Beta values')
plt.ylabel('Difference between expressions')
plt.title('Difference between 1/beta (Vn^2 - Vf^2) and d(x3^2)/dg11')
plt.legend()
plt.grid(True)
plt.show()
