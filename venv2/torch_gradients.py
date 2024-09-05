import torch

# Define the parameter g^(1)_12 with requires_grad=True
g_1_12 = torch.tensor(1.0, requires_grad=True)

# Define other constants like A, B, I_nudge1, I_nudge2, v_x
A = torch.tensor(1.0)  # Example value, modify as needed
B = torch.tensor(1.0)  # Example value, modify as needed
I_nudge1 = torch.tensor(1.0)  # Example value, modify as needed
I_nudge2 = torch.tensor(1.0)  # Example value, modify as needed
v_x = torch.tensor(1.0)  # Example value, modify as needed

# Define g^(0) components as tensors (example values)
g_0_11 = torch.tensor(1.0)
g_0_12 = torch.tensor(1.0)
g_0_21 = torch.tensor(1.0)
g_0_22 = torch.tensor(1.0)

# Define other g^(1) components as tensors (example values)
g_1_11 = torch.tensor(1.0)
g_1_21 = torch.tensor(1.0)
g_1_22 = torch.tensor(1.0)

# Define Z as a function of g^(1)_ij
Z = A * B * ((g_1_11 + g_1_21) * (g_1_22 + g_1_12)) / (
    g_1_11 * g_1_21 * (g_1_22 + g_1_12) + g_1_22 * g_1_12 * (g_1_11 + g_1_21)
)

# Define G_eq as a function of g^(0)_ij and Z
numerator = Z * (g_0_11 * g_0_12 * g_0_21 +
                 g_0_11 * g_0_12 * g_0_22 +
                 g_0_11 * g_0_21 * g_0_22 +
                 g_0_12 * g_0_21 * g_0_22) + (
    g_0_11 * g_0_21 + g_0_11 * g_0_22 + g_0_12 * g_0_21 + g_0_12 * g_0_22)

denominator = Z * (g_0_11 * g_0_12 + g_0_11 * g_0_22 +
                   g_0_12 * g_0_21 + g_0_21 * g_0_22) + (
    g_0_11 + g_0_12 + g_0_21 + g_0_22)

G_eq = numerator / denominator

# Construct the matrix G
G = torch.tensor([
    [-g_0_11 - A * B * g_1_11 - A * B * g_1_12 - g_0_21, 0, B * g_1_11, B * g_1_12, -g_0_11],
    [0, -g_0_22 - A * B * g_1_21 - A * B * g_1_22 - g_0_12, B * g_1_21, B * g_1_22, -g_0_12],
    [A * g_1_11, A * g_1_21, -(g_1_11 + g_1_21), 0, 0],
    [A * g_1_12, A * g_1_22, 0, -(g_1_12 + g_1_22), 0],
    [-g_0_11, -g_0_12, 0, 0, g_0_11 + g_0_12]
])

# Define the vector y (or i_s)
y = torch.tensor([0, 0, I_nudge1, I_nudge2, v_x * G_eq])

# Solve for x using Gx = y
x = torch.linalg.solve(G, y)

# Compute the gradient of x_3 with respect to g^(1)_12
x[2].backward()

# Output the gradient
print("The gradient of x_3 with respect to g_1_12 is:", g_1_12.grad)
