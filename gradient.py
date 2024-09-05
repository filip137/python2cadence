import torch
import matplotlib.pyplot as plt

# Initialize the parameter g^(1)_12 with requires_grad=True
g_1_12 = torch.tensor(10.0, requires_grad=True)
g_0_11 = torch.tensor(4.0, requires_grad=True)

# Initialize other constants
A_values = torch.linspace(1, 10, steps=100)
gradient11_values = []
gradient12_values = []

g_0_12 = torch.tensor(1/2)
g_0_21 = torch.tensor(1/3)
g_0_22 = torch.tensor(1/4)

g_1_11 = torch.tensor(100)
g_1_21 = torch.tensor(17)
g_1_22 = torch.tensor(1/8)

v_x = torch.tensor(30.0)
i_test = torch.tensor(1.0)

def compute_geq(G):
    i_s = torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), i_test])
    x = torch.linalg.solve(G, i_s)
    G_eq = i_test / x[4]
    return G_eq

def compute_x(G, i_nudge, i_eq):
    i_s = torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), i_nudge, i_eq])
    x = torch.linalg.solve(G, i_s)
    return x

for A in A_values:
    B = 1 / A
    
    G = torch.stack([
        torch.stack([g_0_11 + A * B * g_1_11 + A * B * g_1_12 + g_0_21, torch.tensor(0.0), -(B * g_1_11), -(B * g_1_12), -g_0_11]),
        torch.stack([torch.tensor(0.0), g_0_22 + A * B * g_1_21 + A * B * g_1_22 + g_0_12, -B * g_1_21, -B * g_1_22, -g_0_12]),
        torch.stack([-A * g_1_11, -A * g_1_21, (g_1_11 + g_1_21), torch.tensor(0.0), torch.tensor(0.0)]),
        torch.stack([-A * g_1_12, -A * g_1_22, torch.tensor(0.0), (g_1_12 + g_1_22), torch.tensor(0.0)]),
        torch.stack([-g_0_11, -g_0_12, torch.tensor(0.0), torch.tensor(0.0), +g_0_11 + g_0_12])
    ])
    
    G_eq = compute_geq(G)
    i_eq = v_x * G_eq
    x_free = compute_x(G, torch.tensor(0.0), i_eq)
    
    expression_lhs = 0.5 * x_free[3] ** 2
    
    # Reset gradients
    if g_1_12.grad is not None:
        g_1_12.grad.zero_()
    if g_0_11.grad is not None:
        g_0_11.grad.zero_()
    
    # Backward pass
    expression_lhs.backward()
    
    gradient11_values.append(g_0_11.grad.item())
    gradient12_values.append(g_1_12.grad.item())

# Plotting the results with separate y scales

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting the gradient with respect to g_0_11 on the left y-axis
ax1.plot(A_values.numpy(), gradient11_values, 'b-', label="Gradient with respect to g_0_11")
ax1.set_xlabel('A')
ax1.set_ylabel('Gradient with respect to g_0_11', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis to plot the gradient with respect to g_1_12
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(A_values.numpy(), gradient12_values, 'r-', label="Gradient with respect to g_1_12")
ax2.set_ylabel('Gradient with respect to g_1_12', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Title and grid
plt.title('Gradient Values vs. A with Separate Y Scales')
plt.grid(True)
plt.show()