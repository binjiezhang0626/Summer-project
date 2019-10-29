import random
import copy
import torch
import torch.nn as nn
import numpy as np

from math import sin
from math import cos
import matplotlib.pyplot as plt

def loss(a, b):
  loss_function = torch.nn.MSELoss(reduce=True, size_average=True)
  standard = torch.autograd.Variable(a)
  generated = torch.autograd.Variable(b)
  loss = loss_function(standard.float(), generated.float())
  return loss

x1 = np.arange(0, 4*np.pi, 0.01)
y1 = np.sin(x1)
z1 = np.zeros((len(x1),1))
z1[:,0] = y1


G = nn.Sequential(  # Generator
    nn.Linear(1, 128),  
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),  
)

D = nn.Sequential(  # Discriminator
    nn.Linear(2, 128),  
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 1),
    nn.Sigmoid(),  
)

LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()
D_loss_history = []
G_loss_history = []
MSE_loss_history = []

for step in range(2000):

  # start = random.randint(0,len(x1)-600)
  # end = start + 600
  start = 0
  end = len(x1)
  G_sin_input = z1[start:end-1]
  D_sin_input = z1[start+1:end]
  G_input = torch.from_numpy(G_sin_input).float()
  D_input = torch.from_numpy(D_sin_input).float()
  D_real = torch.zeros((end - start -1),2)
  D_feak = torch.zeros((end - start -1),2)

  G_output = G(G_input)

  for i in range((end - start -2)):
    D_real[i] = torch.cat([G_input[i],G_input[i+1]])
  for i in range((end - start -1)):
    D_feak[i] = torch.cat([G_input[i],G_output[i]])
  prob_0 = D(D_real)  # D try to increase this prob
  prob_1 = D(D_feak)  # D try to reduce this prob

  D_loss = - torch.mean(torch.log(prob_0) + torch.log(1. - prob_1))
  G_loss = torch.mean(torch.log(1. - prob_1))
  MSE_loss = loss(D_input, G_output)
  D_loss_history.append(D_loss)
  G_loss_history.append(G_loss)
  MSE_loss_history.append(MSE_loss)
    
  opt_D.zero_grad()
  D_loss.backward(retain_graph=True)  # reusing computational graph
  opt_D.step()

  opt_G.zero_grad()
  G_loss.backward()
  opt_G.step()

  if step % 50 == 0:
        plt.cla()
        plt.plot(G_sin_input, c='#FF0000', lw=1, label='Standard sin')
        plt.plot(G_output.data.numpy(), lw=1, label='Generated sin', )         
        plt.legend(loc='upper right', fontsize=10)
        plt.ylim((-1.5, 1.5))
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()

plt.plot(MSE_loss_history)
plt.show()
torch.save(G,'Sin_Generator.plk')
print('Network Saved!')
