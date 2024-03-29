import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import sin
from math import cos
from math import acos
from scipy.integrate import odeint

def pendulum_state(w, t, l):
    th, v = w
    dth = v
    dv  = - g/l * sin(th)
    return dth, dv

def loss(a, b):
  loss_function = torch.nn.MSELoss(reduce=True, size_average=True)
  standard = torch.autograd.Variable(a)
  generated = torch.autograd.Variable(b)
  loss = loss_function(standard.float(), generated.float())
  return loss

g = 9.8
t1 = np.arange(0,2,0.01)
th = 1.0
state = odeint(pendulum_state, (1.0, 0), t1, args=(1.0,))

z = np.zeros((len(t1),3))
z[:,0] = t1
z[:,1] = state[:,0]
z[:,2] = state[:,1]

G_input = z[0:len(t1)-1]
D_input = z[1:len(t1)]

G_input = torch.from_numpy(G_input).float()
D_input = torch.from_numpy(D_input).float()

G = nn.Sequential(  # Generator
    nn.Linear(3, 128),  
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 3),  
)


D = nn.Sequential(  # Discriminator
    nn.Linear(3, 128),  
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 1), 
)

LR_G = 0.00005  # learning rate for generator
LR_D = 0.00005  # learning rate for discriminator
weight_cliping_limit = 0.01
n_critic = 5
opt_D = torch.optim.RMSprop(D.parameters(), lr=LR_D)
opt_G = torch.optim.RMSprop(G.parameters(), lr=LR_G)

plt.ion()
D_loss_history = []
G_loss_history = []
MSE_loss_history = []

for step in range(10000):

    for n in range(n_critic):

        for p in D.parameters():
            p.data.clamp_(-weight_cliping_limit,weight_cliping_limit)

        
        G_output = G(G_input)
        prob_0 = D(D_input)  # D try to increase this prob
        prob_1 = D(G_output)  # D try to reduce this prob

        D_loss = - torch.mean((prob_0) + (1. - prob_1))
        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)  # reusing computational graph
        opt_D.step()

    opt_G.zero_grad()
    G_output = G(G_input)
    prob_1 = D(G_output)
    G_loss = torch.mean((1. - prob_1))
    MES_loss = loss(D_input, G_output)

    G_loss.backward()
    opt_G.step() 
    
    D_loss_history.append(D_loss)
    G_loss_history.append(G_loss)
    MSE_loss_history.append(MES_loss)

    if step % 100 == 0:
        plt.cla()
        plt.plot(state[:,0], c='#FF0000', lw=1, label='Standard th1')
        plt.plot(state[:,1], c='#0000FF', lw=1, label='Standard v', )
        plt.plot(G_output.data.numpy()[:,1], lw=1, label='Generated th1', )
        plt.plot(G_output.data.numpy()[:,2], lw=1, label='Generated v', )
        plt.legend(loc='upper right', fontsize=10)
        plt.ylim((-6, 6))
        plt.draw()
        plt.pause(0.01)
        
plt.ioff()
plt.show()

plt.plot(MSE_loss_history)
plt.show()

torch.save(G,'Generator_wgan.plk')
torch.save(D,'Discriminator_wgan.plk')
print("network saved")