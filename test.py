import random
import torch
import torch.nn as nn
import numpy as np
import math
from math import sin
from math import cos
import matplotlib.pyplot as plt

x = np.arange(0, 10)
y = np.arange(1,11)
a = np.vstack([x[i] for i in range(len(x))])
b = np.vstack([y[i] for i in range(len(y))])

a = torch.from_numpy(a).float()
b = torch.from_numpy(b).float()
# print(a)
# print(b)
def loss(a, b):
  loss_function = torch.nn.MSELoss(reduce=True, size_average=True)
  standard = torch.autograd.Variable(a)
  generated = torch.autograd.Variable(b)
  loss = loss_function(standard.float(), generated.float())
  return loss

G = nn.Sequential(  # Generator
    nn.Linear(1, 128),  
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),  
)

D = nn.Sequential(  # Discriminator
    nn.Linear(1, 128),  
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),  
)

LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)


D_loss_history = []
G_loss_history = []
MES_loss_history = []

for step in range(5000):
    
    G_fakes = G(a)
    prob_0 = D(b)  # D try to increase this prob
    prob_1 = D(G_fakes)  # D try to reduce this prob

    D_loss = - torch.mean(torch.log(prob_0) + torch.log(1. - prob_1))
    G_loss = torch.mean(torch.log(1. - prob_1))
    MSE_loss = loss(b, G_fakes)

    D_loss_history.append(D_loss)
    G_loss_history.append(G_loss)
    MES_loss_history.append(MSE_loss)
    

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)  # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 100 == 0:
    	print(G_fakes[-1])

plt.plot(MES_loss_history)
plt.show()
