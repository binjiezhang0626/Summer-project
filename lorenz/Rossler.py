import random
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def rossler(w, t, a, b, c):
    x, y, z = w
    return np.array([(-y-z), x+a*y, b+(x-c)*z])

def loss(a, b):
    loss_function = torch.nn.MSELoss(reduce=True, size_average=True)
    standard = torch.autograd.Variable(a)
    generated = torch.autograd.Variable(b)
    loss = loss_function(standard.float(), generated.float())
    loss = np.log(loss)
    return loss

t = np.arange(0, 40, 0.01)
track = odeint(rossler, (0.0, 2.0, 0.0), t, args=(0.2, 0.2, 5.7))

G = nn.Sequential(  # Generator
    nn.Linear(3, 128),  
    nn.ReLU(),
    nn.Linear(128, 128),
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
    nn.Linear(6, 128),  
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 1),
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
fig = plt.figure()

for step in range(10000):

    time_start = random.randint(0,len(t)-500)
    time_end = time_start + 500
    G_state_input = track[time_start:time_end-1]
    D_state_input = track[time_start+1:time_end]
    G_input = torch.from_numpy(G_state_input).float()
    D_input = torch.from_numpy(D_state_input).float()

    D_real = torch.zeros((time_end - time_start -1),6)
    D_feak = torch.zeros((time_end - time_start -1),6)
    
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
    MSE_loss_history.append(MES_loss)

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)  # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    
    if step % 100 == 0:
        plt.cla()
        ax = Axes3D(fig)
        ax.plot(G_state_input[:,0], G_state_input[:,1], G_state_input[:,2],c='#FF0000', lw=1, label='Standard Rossler')
        ax.plot(G_output.data.numpy()[:,0],G_output.data.numpy()[:,1],G_output.data.numpy()[:,2],c='#0000FF', lw=1, label='Generated Rossler')
        plt.show()
        plt.pause(0.01)
              
plt.ioff()
plt.show()

plt.plot(MSE_loss_history)
plt.show()
torch.save(G,'Rossler_Generator.plk')
print("network saved")