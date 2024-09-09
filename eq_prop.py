import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from PIL import Image
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Moons Dataset')
plt.show()

def plot_decision_boundary(model, X, y, grid_step=0.1, cmap=plt.cm.Paired):
    # Set the model to evaluation mode
    model.eval()

    # Generate a grid of points covering the feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))

    # Convert the grid to a tensor and pass it through the model
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid)
        _, Z = torch.max(Z, 1)
        Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)

    # Plot the original data points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=cmap)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()  







# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
# Create TensorDataset objects
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


# Parameters for the network
dt = 0.5
T = 10  # free phase time
Kmax = 5  # nudge phase time
beta = 0.2  # nudging parameter
fcLayers = [2, 8, 2] # [output, hidden, input]
act = 'hardsigm'

# Define activation function and its derivatives
if act =='sigmoid':
    def rho(x):
        return 1/(1+np.exp(-(4*(x-0.5))))
    def rhop(x):
        return 4*np.matmul(1/(1+np.exp(-(4*(x-0.5)))),1-1/(1+np.exp(-(4*(x-0.5)))))

if act =='hardsigm':
    def rho(x):
        return np.clip(x,0,1)
    def rhop(x):
        return (x>=0)&(x<=1)

if act == 'tanh':
    def rho(x):
        return np.tanh(x)
    def rhop(x):
        return 1 - np.power(np.tanh(x),2)

# Create the network
class mlp_eqprop(nn.Module):
    def __init__(self, fcLayers, dt, T, Kmax, beta, loss):
        super(mlp_eqprop, self).__init__()
        self.fcLayers = fcLayers
        self.dt = dt
        self.T = T
        self.Kmax = Kmax
        self.beta = beta
        self.loss = loss
        if loss == 'MSE':
            self.softmax_output = False
        elif loss == 'Cross-entropy':
            self.softmax_output = True
            
        W = nn.ModuleList(None)
        for i in range(len(fcLayers)-1):
            W.extend([nn.Linear(fcLayers[i+1], fcLayers[i], bias=True)])
        self.W = W

    def stepper_softmax(self, s, target=None, beta= None):

        if len(s) < 3:
            raise ValueError("Input list 's' must haave at least three elements for softmax-readout.")
    
        # Separate 'h' elements and 'y'
        h = s[1:]  # All but the first element are considered 'h' # at the start s has only the inputs
        y = F.softmax(self.W[0](rho(h[0])), dim=1) #maybe different activation function for the last layer?
    
        dhdt = [-h[0] + rhop(h[0]) *self.W[1](rho(h[1]))] #the update for the first hidden layer from the input
        
        if target is not None and beta is not None:
            dhdt[0] = dhdt[0] + beta * torch.mm((target-y), self.W[0].weight) #nudge of the output layer
    
        for layer in range(1, len(h) - 1):
            dhdt.append(-h[layer] + rhop(h[layer]) * (self.W[layer+1](rho(h[layer+1]))
                                                               + torch.mm(rho(h[layer - 1]),self.W[layer].weight)))
        # update h
        for (layer, dhdt_item) in enumerate(dhdt):
                h[layer] = h[layer] + self.dt * dhdt_item
                h[layer] = h[layer].clamp(0, 1)
            
        return [y] + h
        
    def stepper_c(self, s, target=None, beta=None):
            """
            stepper function for energy-based dynamics of EP
            """
            if len(s) < 2:
                raise ValueError("Input list 's' must have at least two elements.")
            #this is dsdt for the output layer
            dsdt = [-s[0] + (rhop(s[0])*(self.W[0](rho(s[1]))))]
            #in the nudge phase the output layer is clamped
            if beta is not None and target is not None:
                dsdt[0] = dsdt[0] + beta*(target-s[0])
                    #here calculate the updates layer by layer
            for layer in range(1, len(s)-1):  # start at the first hidden layer and then to the before last hidden layer
                dsdt.append(-s[layer] + rhop(s[layer])*(self.W[layer](rho(s[layer+1])) + torch.mm(rho(s[layer-1]), self.W[layer-1].weight)))
    
            for (layer, dsdt_item) in enumerate(dsdt):
                s[layer] = s[layer] + self.dt*dsdt_item
                s[layer] = s[layer].clamp(0, 1)
    
            return s

    
    def forward(self, s, beta=None, target=None, tracking=False):
        #update parameters for all time steps
        T, Kmax = self.T, self.Kmax
        if beta is None and target is None:
            q, y = torch.empty((s[1].size(1),T)), torch.empty((s[0].size(1), T))

        else:
            q, y = torch.empty((s[1].size(1),Kmax)), torch.empty((s[0].size(1), Kmax))

        with torch.no_grad():
            # continuous time EP
            if beta is None and target is None:
                # free phase
                if self.softmax_output:
                    for t in range(T):
                        s = self.stepper_softmax(s, target=target, beta=beta)
                        if tracking:
                            q[:,t] = s[1][0,:]
                            y[:,t] = s[0][0,:]
                else:
                    for t in range(T):
                        s = self.stepper_c(s, target=target, beta=beta)
                        if tracking:
                            q[:,t] = s[1][0,:]
                            y[:,t] = s[0][0,:]
            else:
                # nudged phase
                if self.softmax_output:
                    for t in range(Kmax):
                        s = self.stepper_softmax(s, target=target, beta=beta)
                        if tracking:
                            q[:,t] = s[1][0,:]
                            y[:,t] = s[0][0,:]
                else:
                    for t in range(Kmax):
                        s = self.stepper_c(s, target=target, beta=beta)
                        if tracking:
                            q[:,t] = s[1][0,:]
                            y[:,t] = s[0][0,:]

        return s, q, y


    def compute_gradients_ep(self, s, seq, target=None):
        """
        Compute EQ gradient to update the synaptic weight 
        """
        batch_size = s[0].size(0)
        # learning rate should be the 1/beta of the BP learning rate
        # in this way the learning rate is corresponded with the sign of beta
        coef = 1 / (self.beta * batch_size)

        gradW, gradBias = [], []

        with torch.no_grad():
            #calculate the gradients of the output layer
            if self.softmax_output:
                gradW.append(
                    -(0.5 / batch_size) * (torch.mm(torch.transpose((s[0] - target), 0, 1), rho(s[1])) +
                                           torch.mm(torch.transpose((seq[0] - target), 0, 1),
                                                    rho(seq[1]))))
                gradBias.append(-(0.5 / batch_size) * (s[0] + seq[0] - 2 * target).sum(0))
            else:
                gradW.append(coef * (torch.mm(torch.transpose(rho(s[0]), 0, 1), rho(s[1]))
                                     - torch.mm(torch.transpose(rho(seq[0]), 0, 1),
                                                rho(seq[1]))))
                gradBias.append(coef * (rho(s[0]) - rho(seq[0])).sum(0))
            # calculate the gradients of the other layers 
            for layer in range(1, len(s) - 1):
                gradW.append(coef * (torch.mm(torch.transpose(rho(s[layer]), 0, 1), rho(s[layer+1]))
                                     - torch.mm(torch.transpose(rho(seq[layer]), 0, 1),
                                                rho(seq[layer+1]))))
                gradBias.append(coef * (rho(s[layer]) - rho(seq[layer])).sum(0))

        for (i, param) in enumerate(self.W):
            param.weight.grad = -gradW[i]
            param.bias.grad = -gradBias[i]
    
    def init_state(self, data):
        """
        Init the state of the network
        State if a dict, each layer is state["S_layer"]
        """
        state = []
        size = data.size(0)
        for layer in range(len(self.fcLayers) - 1): #set everything to zero except the last layer with the inputs
            state.append(torch.zeros(size, self.fcLayers[layer], requires_grad=False))

        state.append(data.float())

        return state
    

def defineOptimizer(net, lr, type):
    net_params = []
    for i in range(len(net.W)):
        net_params += [{'params': [net.W[i].weight], 'lr': lr[i]}]
        net_params += [{'params': [net.W[i].bias], 'lr': lr[i]}]
    if type == 'SGD':
        optimizer = torch.optim.SGD(net_params)
    elif type == 'Adam':
        optimizer = torch.optim.Adam(net_params)
    else:
        raise ValueError("{} type of Optimizer is not defined ".format(type))

    return net_params, optimizer


loss = 'Cross-entropy'  # 'MSE' or 'Cross-entropy' 
lr = [0.01, 0.03]  
net = mlp_eqprop(fcLayers, dt, T, Kmax, beta, loss)
net_params, optimizer = defineOptimizer(net, lr, 'Adam')
epoch = 10

for rep in range(epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        invert_targ = 1 - targets
        targets = torch.stack((targets, invert_targ), dim =1)
        s = net.init_state(data) #write the inputs
        s, m, n = net.forward(s, tracking=True) #do the free phase propagation, m and n are the neurons
        if batch_idx == 0:
            train_hidden = m
            train_output = n
        else:
            train_hidden = torch.cat((train_hidden, m), 1)
            train_output = torch.cat((train_output, n), 1)
        seq = s.copy() #store the equilibrum value
        s, m, n = net.forward(s, beta=beta, target=targets, tracking=True)
        train_hidden = torch.cat((train_hidden, m), 1) #store the value of neurons
        train_output = torch.cat((train_output, n), 1)#store the value of the outputs
        # update weight
        net.compute_gradients_ep(s, seq, targets)# compute gradients
        optimizer.step()

    # Test
    test_number = 0
    correct_number = 0
    for batch_idx, (data, targets) in enumerate(test_loader):
        invert_targ = 1 - targets
        targets = torch.stack((targets, invert_targ), dim =1)
        s = net.init_state(data)
        s, m, n = net.forward(s, tracking=True)
        if batch_idx ==0:
            test_hidden = m
            test_output = n
        else:
            test_hidden = torch.cat((test_hidden, m), 1)
            test_output = torch.cat((test_output, n), 1)
        # calculate the accuracy
        predictions = torch.argmax(s[0])
        test_number += len(targets)
        correct_number += (prediction==targets).sum().float()
        #plot_decision_boundary(net, X, y)
    print('The accuracy is:', float(correct_number/test_number))

# Draw the evolution figure for the neurons
def plot_evolution(x):
    fig, ax = plt.subplots()
    for out in range(x.size(0)):
        ax.plot(x[out,:], label=f'Neuron{out}')
    ax.set_ylabel('Neuron state')
    ax.set_xlabel('Time')
    ax.legend(bbox_to_anchor=(1.01, 1.01), loc='upper left', ncol=1, fontsize='large', frameon=False)
# Draw the weights 
def imshow_weight(w):
    fig,ax = plt.subplots(1, len(w))
    for i in range(len(w)):
        ax[i].imshow(w[i].weight.detach().numpy(), cmap=cm.coolwarm)
        ax[i].set_ylabel(f'Layer {i}')
    fig.suptitle('Weight values')
    plt.tight_layout()
    plt.subplots_adjust(top=1.2)

  

plot_evolution(train_output)
plot_decision_boundary(net, X, y)

