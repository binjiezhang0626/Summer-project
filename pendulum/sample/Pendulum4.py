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
t1 = np.arange(0,10,0.01)
th = 1.0
state = odeint(pendulum_state, (1.0, 0), t1, args=(1.0,))

G = nn.Sequential(  # Generator
    nn.Linear(2, 128),  
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 2),  
)

D = nn.Sequential(  # Discriminator
    nn.Linear(4, 128),  
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 4),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(4, 1),
    nn.Sigmoid(),  
)

LR_G = 0.0001  
LR_D = 0.0001  
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()
D_loss_history = []
G_loss_history = []
MSE_loss_history = []

for step in range(2000):

    time_start = random.randint(0,len(t1)-200)
    time_end = time_start +200
    G_state_input = state[time_start:time_end-1]
    D_state_input = state[time_start+1:time_end]

    G_input = torch.from_numpy(G_state_input).float()
    D_input = torch.from_numpy(D_state_input).float()
    D_real = torch.zeros((time_end - time_start -1),4)
    D_feak = torch.zeros((time_end - time_start -1),4)
    
    G_output = G(G_input)

    for i in range((time_end - time_start -2)):
        D_real[i] = torch.cat([G_input[i],G_input[i+1]])
  
    for i in range((time_end - time_start -1)):
        D_feak[i] = torch.cat([G_input[i],G_output[i]])

    prob_0 = D(D_real)  # D try to increase this prob
    prob_1 = D(D_feak)  # D try to reduce this prob

    D_loss = - torch.mean(torch.log(prob_0) + torch.log(1. - prob_1))
    G_loss = torch.mean(torch.log(1. - prob_1))
    MES_loss = loss(D_input, G_output)
    D_loss_history.append(D_loss)
    G_loss_history.append(G_loss)
    

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)  # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 40 == 0:
        MES_loss = loss(D_input, G_output)
        MSE_loss_history.append(MES_loss)
        # plt.cla()
        # plt.plot(G_state_input[:,0], c='#FF0000', lw=1, label='Standard th1')
        # plt.plot(G_state_input[:,1], c='#0000FF', lw=1, label='Standard v', )
        # plt.plot(G_output.data.numpy()[:,0], lw=1, label='Generated th1', )
        # plt.plot(G_output.data.numpy()[:,1], lw=1, label='Generated v', )
        # plt.legend(loc='upper right', fontsize=10)
        # plt.ylim((-6, 6))
        # plt.draw()
        # plt.pause(0.01)
              
plt.ioff()
plt.show()

plt.plot(MSE_loss_history)
plt.show()
np.savetxt('mse_20.csv',MSE_loss_history,delimiter=',')
print("csv saved")
# torch.save(G,'Sample_pendulum_Generator1000.plk')
# print("network saved")