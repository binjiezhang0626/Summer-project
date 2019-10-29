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
BATCH_SIZE = 64  
lr_G = 0.0001  # learning rate for generator
lr_D = 0.0001  # learning rate for discriminator

t = np.arange(0, 10, 0.1)
th = 1.0
track = odeint(pendulum_state, (1.0, 0), t, args=(1.0,))
th1,  v= track[:, 0], track[:, 1] 

states = np.array([th1,v]).flatten()
standard_states = np.vstack([states for _ in range(BATCH_SIZE)])
standard_states = torch.from_numpy(standard_states).float()

Generator = nn.Sequential(  
    nn.Linear(200, 128),  
    nn.ReLU(),
    nn.Linear(128, 200),  
)

Discriminator = nn.Sequential(  
    nn.Linear(200, 128),  
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),  
)

opt_D = torch.optim.Adam(Discriminator.parameters(), lr=lr_D)
opt_G = torch.optim.Adam(Generator.parameters(), lr=lr_G)

plt.ion()  # something about continuous plotting

D_loss_history = []
G_loss_history = []
MSE_loss_history = []

for step in range(10000):
    G_inputs = torch.randn(BATCH_SIZE, 200) 
    G_generates = Generator(G_inputs) 

    prob_0 = Discriminator(standard_states)  # D try to increase this prob
    prob_1 = Discriminator(G_generates)  # D try to reduce this prob

    D_loss = - torch.mean(torch.log(prob_0) + torch.log(1. - prob_1))
    G_loss = torch.mean(torch.log(1. - prob_1))
    MSE_loss = loss(standard_states, G_generates)
    

    D_loss_history.append(D_loss)
    G_loss_history.append(G_loss)
    MSE_loss_history.append(MSE_loss)
    

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)  # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 50 == 0:  # plotting
      
        plt.cla()
        plt.plot(t, th1, c='#FF0000', lw=1, label='Standard th1')
        plt.plot(t, v, c='#0000FF', lw=1, label='Standard v', )
        plt.plot(t, G_generates.data.numpy()[0][0:100], lw=1, label='Generated th1', )         
        plt.plot(t, G_generates.data.numpy()[0][100:200], lw=1, label='Generated v', )
        # print(MSE_loss.data)
        plt.legend(loc='upper right', fontsize=10);
        plt.ylim((-8, 8));
        plt.draw();
        plt.pause(0.01)
       
plt.ioff()
plt.show()


np.savetxt('loss.csv', MSE_loss_history, delimiter = ',')
plt.plot(MSE_loss_history)
plt.show()