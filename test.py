import torch
import matplotlib.pyplot as plt
# Initialize the parameter g^(1)_12 with requires_grad=True
g_1_12 = torch.tensor(1.0, requires_grad=True)  # Changed initial value
g_0_11 = torch.tensor(1.0, requires_grad=True)

# Initialize other constants like A, B, v_x with different values
A = torch.tensor(2.0)  # Different example value
B = torch.tensor(1.0) # Different example value



v_x = torch.tensor(3.0)
i_test = torch.tensor(1.0)  # Different example value

# Initialize g^(0) components as tensors with different example values
g_0_12 = torch.tensor(1/2)
g_0_21 = torch.tensor(1/3)
g_0_22 = torch.tensor(1/4)

# Initialize other g^(1) components as tensors with different example values
g_1_11 = torch.tensor(3)
g_1_21 = torch.tensor(4)
g_1_22 = torch.tensor(1/8)


# Construct the matrix G
# Construct the G matrix with the signs of the top left 4x4 submatrix changed
G = torch.stack([
    torch.stack(
        [g_0_11 + A * B * g_1_11 + A * B * g_1_12 + g_0_21, torch.tensor(0.0), -(B * g_1_11), -(B * g_1_12), -g_0_11]),
    torch.stack(
        [torch.tensor(0.0), g_0_22 + A * B * g_1_21 + A * B * g_1_22 + g_0_12, -B * g_1_21, -B * g_1_22, -g_0_12]),
    torch.stack([-A * g_1_11, -A * g_1_21, (g_1_11 + g_1_21), torch.tensor(0.0), torch.tensor(0.0)]),
    torch.stack([-A * g_1_12, -A * g_1_22, torch.tensor(0.0), (g_1_12 + g_1_22), torch.tensor(0.0)]),
    torch.stack([-g_0_11, -g_0_12, torch.tensor(0.0), torch.tensor(0.0), +g_0_11 + g_0_12])
])


third_row = torch.stack([-A * g_1_11, -A * g_1_21, (g_1_11 + g_1_21), torch.tensor(0.0), torch.tensor(0.0)]) * (1 / A)
fourth_row = torch.stack([-A * g_1_12, -A * g_1_22, torch.tensor(0.0), (g_1_12 + g_1_22), torch.tensor(0.0)]) * (1 / A)

# Creating the new symmetric matrix
G_b = torch.stack([
    torch.stack(
        [g_0_11 + A * B * g_1_11 + A * B * g_1_12 + g_0_21, torch.tensor(0.0), -(B * g_1_11), -(B * g_1_12), -g_0_11]),
    torch.stack(
        [torch.tensor(0.0), g_0_22 + A * B * g_1_21 + A * B * g_1_22 + g_0_12, -B * g_1_21, -B * g_1_22, -g_0_12]),
    third_row,
    fourth_row,
    torch.stack([-g_0_11, -g_0_12, torch.tensor(0.0), torch.tensor(0.0), +g_0_11 + g_0_12])
])





G_inv = torch.inverse(G)
G_b_inv = torch.inverse(G_b)

def compute_geq(G):
    # Define the vector i_s using the given beta
    i_s = torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), i_test])

    # Solve for x using Gx = i_s
    x = torch.linalg.solve(G, i_s)
    G_eq = i_test / x[4]
    return G_eq

def compute_x(beta, G, i_nudge, i_eq):
    # Define the vector i_s using the given beta
    i_s = torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), i_nudge, i_eq])
    # Solve for x using Gx = i_s
    x = torch.linalg.solve(G, i_s)
    return x

# Compute the gradient of 1/2 * x[3]^2 with respect to g_1_12
beta = torch.tensor(0.0)  # Free phase
loss = torch.tensor(0.0)
G_eq = compute_geq(G)  # The equivalent resistance of the circuit when the nudge current is off

i_nudge = torch.tensor(0.0)
i_eq = v_x * G_eq
x_free = compute_x(beta, G_b, i_nudge, i_eq)
#Vf = (x_free[3] - A * x_free[0]) ** 2
#Vf2 = -A * B * (x_free[0] ** 2) + B * x_free[0] * x_free[3] + A * x_free[0] * x_free[3] - (x_free[3] ** 2)
expression_lhs = 0.5 * x_free[3] ** 2

# Reset gradients
if g_1_12.grad or g_0_11.grad is not None:
    g_1_12.grad.zero_()
    g_0_11.grad.zero_()
# First backward pass for the first expression
expression_lhs.backward()
gradient12 = g_1_12.grad.item()
gradient11 = g_0_11.grad.item()

# Compute Vn with beta=10e-6 (as a float)
beta_nudge = 10e-7
i_nudge = - A * beta_nudge * x_free[3]
i_equivalent = (v_x-G_inv[4,3]*i_nudge)/G_inv[4,4]

x_beta_new = compute_x(beta, G, i_nudge, i_equivalent)


Vf = (x_free[3]/torch.sqrt(A) - x_free[0]*torch.sqrt(A)) ** 2
Vn = (x_beta_new[3]/torch.sqrt(A) - x_beta_new[0]*torch.sqrt(A)) ** 2


Vf2 = (v_x - x_free[0]) ** 2
Vn2 = (v_x - x_beta_new[0]) ** 2
#Vf2 = - A * B * (x_free[0] ** 2) + B * x_free[0] * x_free[3] + A * x_free[0] * x_free[3] - (x_free[3] ** 2)
#Vn2 = - A * B * (x_beta_new[0] ** 2) + B * x_beta_new[0] * x_beta_new[3] + A * x_beta_new[0] * x_beta_new[3] - (x_beta_new[3] ** 2)
# Compute the expression 1/beta_nudge * (Vn - Vf)
expression_rhs = 1 / (2*beta_nudge) * (Vn - Vf)
expression_rhs2 = 1 / (2*beta_nudge) * (Vn2 - Vf2)
# Output the gradient from the first expression and the approximation from the second
print("Gradient of 1/2 * x[3]^2 with respect to g_0_11:", gradient11)
print("Approximation of gradient from 1/beta * (Vn/A^2 - Vf/A^2)^2:", expression_rhs.item())
#print("Approximation of gradient from 1/beta * (Vn - Vf)^2:", expression_rhs2.item())

beta_nudge_values = torch.logspace(-1, -9, steps=100)
gradient_approximations_11 = []
gradient_approximations_12 = []

for beta_nudge in beta_nudge_values:
    i_nudge = - A * beta_nudge * x_free[3]
    i_equivalent = (v_x-G_inv[4,3]*i_nudge)/G_inv[4,4]
    x_beta_new = compute_x(beta, G, i_nudge, i_equivalent)
    Vf2 = (x_free[3]/torch.sqrt(A) - x_free[0]*torch.sqrt(A)) ** 2
    Vn2 = (x_beta_new[3]/torch.sqrt(A) - x_beta_new[0]*torch.sqrt(A)) ** 2
    
    Vf = (v_x - x_free[0]) ** 2
    Vn = (v_x - x_beta_new[0]) ** 2
    
    expression_rhs = 1 / (2*beta_nudge) * (Vn - Vf)
    expression_rhs2 = 1 / (2*beta_nudge) * (Vn2 - Vf2)
    
    
    gradient_approximations_11.append(expression_rhs.item())
    gradient_approximations_12.append(expression_rhs2.item())
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(beta_nudge_values.numpy(), gradient_approximations_11, marker='o')
plt.axhline(y=gradient11, color='r', linestyle='--', label="True Gradient")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Beta Nudge')
plt.ylabel('Gradient 11 Approximation')
plt.title('Gradient Approximation  11 vs. Beta Nudge')
plt.grid(True)
plt.figure(figsize=(10, 6))
plt.plot(beta_nudge_values.numpy(), gradient_approximations_12, marker='o')
plt.axhline(y=gradient12, color='r', linestyle='--', label="True Gradient")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Beta Nudge')
plt.ylabel('Gradient 12 Approximation')
plt.title('Gradient Approximation 12 vs. Beta Nudge')
plt.grid(True)
plt.show()
