from sympy import symbols, Eq, solve, simplify, collect

# Define symbols
x1, x2, x3, x4, g11, g12, g21, g22, g31, g32, g41, g42, B, A, vx, i_n, i_nn = symbols('x1 x2 x3 x4 g11 g12 g21 g22 g31 g32 g41 g42 B A vx i_n i_nn')

# Define the equations
eq1 = Eq((-g11 - A*B*g31 - A*B*g32 - g21)*x1 + B*g31*x3 + B*g41*x4, -vx*g11)
eq2 = Eq((-g22 - A*B*g41 - A*B*g42 - g12)*x2 + B*g41*x3 + B*g42*x4, -vx*g12)
eq3 = Eq(A*g31*x1 + A*g41*x2 - (g31 + g41)*x3, i_n)
eq4 = Eq(A*g32*x1 + A*g42*x2 - (g32 + g42)*x4, i_nn)

random_values = {
    
    g11: 2, g12: 3, g21: 4, g22: 5,
    g31: 1, g32: 2, g41: 3, g42: 4,
    A: 4, B: 1/4, vx: 1, i_n: 0, i_nn: 0
}

# Substitute the values into the simplified solutions
solution = solve((eq1, eq2, eq3, eq4), (x1, x2, x3, x4))
simplified_solution = {var: simplify(expr) for var, expr in solution.items()}

numeric_solution = {var: expr.subs(random_values) for var, expr in simplified_solution.items()}

print(numeric_solution)

# Collect terms based on i_n, i_nn, and vx in x1 solution
x1_solution = simplified_solution[x1]
terms_i_n = collect(x1_solution, i_n, evaluate=False)
terms_i_nn = collect(x1_solution, i_nn, evaluate=False)
terms_vx = collect(x1_solution, vx, evaluate=False)

# Output results
print("Simplified Solutions:", simplified_solution)
print("Terms involving i_n in x1:", terms_i_n)
print("Terms involving i_nn in x1:", terms_i_nn)
print("Terms involving vx in x1:", terms_vx)