from estimating_causal_powers import Analysis
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
import sys

if sys.argv[1] == "GPU":
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# assumptions
prior = 0
noise_bounds = torch.tensor([[0,0.1]], device=device)

# get posterior at zero noise
ana = Analysis(noise_bounds, optimize=True)
base, _, _ = ana.run_analysis(torch.tensor([0.], dtype=torch.float32, requires_grad=True, device=device))


class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, noise_bounds : torch.tensor, noise_ndim = 1):
        
        super().__init__()
        
        if sys.argv[1] == "GPU":
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        weights = []
        for i in range(noise_bounds.shape[0]):
            weights.append(torch.distributions.Uniform(noise_bounds[i][0], noise_bounds[i][1]).sample((1,)))
        # make weights torch parameters
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float, device=self.device, requires_grad=True))        

        # weights = torch.distributions.Uniform(0, 0.1).sample((1,))
        # # make weights torch parameters
        # self.weights = nn.Parameter(weights)        
        
    def forward(self):
        e, _, _ = ana.run_analysis(self.weights)
        return e
    
def training_loop(model, optimizer, n=1000):
    "Training loop for torch model."
    losses = []
    weights = []
    distances = []

    for i in range(n):
        preds = model()
        loss = 1 / torch.abs(base - preds)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        # record progress
        losses.append(loss.item())
        weights.append(m.weights.item())  
        distances.append(torch.abs(base - preds).item())
    return losses, weights, distances


# Instantiate optimizer
max_distances = []
max_noises = []

# k is the number of restart
k = 3
for i in range(k):
    # instantiate model
    m = Model(noise_bounds)

    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    losses, noises, distances = training_loop(m, opt, n=100)
    if (len(max_distances) == 0 or max(distances) > max(max_distances)):
        max_distances, max_noises = distances, noises

plt.figure(figsize=(14, 7))
plt.plot(max_noises, max_distances)
print("---------------------------------")
print("Base(expectation with 0 eps): ", base)
print("Max ED1 is :", max(max_distances))
print("Reached at eps: ", max_noises[max_distances.index(max(max_distances))])
print("---------------------------------")
plt.show()