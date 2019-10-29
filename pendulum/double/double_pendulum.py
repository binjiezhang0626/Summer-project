import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos
import numpy as np
from scipy.integrate import odeint

g = 9.8
class DoublePendulum(object):
    def __init__(self, m1, m2, l1, l2):
        self.m1, self.m2, self.l1, self.l2 = m1, m2, l1, l2
        self.init_status = np.array([0.0,0.0,0.0,0.0])
        
    def equations(self, w, t):

        m1, m2, l1, l2 = self.m1, self.m2, self.l1, self.l2
        th1, th2, v1, v2 = w
        dth1 = v1
        dth2 = v2
        
        #eq of th1
        a = l1*l1*(m1+m2)  # dv1 parameter
        b = l1*m2*l2*cos(th1-th2) # dv2 paramter
        c = l1*(m2*l2*sin(th1-th2)*dth2*dth2 + (m1+m2)*g*sin(th1))
        
        #eq of th2
        d = m2*l2*l1*cos(th1-th2) # dv1 parameter
        e = m2*l2*l2 # dv2 parameter
        f = m2*l2*(-l1*sin(th1-th2)*dth1*dth1 + g*sin(th2))
        
        dv1, dv2 = np.linalg.solve([[a,b],[d,e]], [-c,-f])
        
        return np.array([dth1, dth2, dv1, dv2])

def double_pendulum_odeint(pendulum, t):
    track = odeint(pendulum.equations, pendulum.init_status, t)
    pendulum.init_status = track[-1,:].copy() 
    return track

def loss(a, b):
    loss_function = torch.nn.MSELoss(reduce=True, size_average=True)
    standard = torch.autograd.Variable(a)
    generated = torch.autograd.Variable(b)
    loss = loss_function(standard.float(), generated.float())
    return loss
  
pendulum = DoublePendulum(1.0, 2.0, 1.0, 1.0)
t = np.arange(0, 10, 0.01) 
th1, th2 = 1.0, 1.0
pendulum.init_status[:2] = th1, th2
track = double_pendulum_odeint(pendulum, t)

G = nn.Sequential(  # Generator
    nn.Linear(4, 256),  
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 4),  
)

D = nn.Sequential(  # Discriminator
    nn.Linear(8, 256),  
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 1),
    nn.Sigmoid(),  
)

LR_G = 0.00001  # learning rate for generator
LR_D = 0.00001  # learning rate for discriminator
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()
D_loss_history = []
G_loss_history = []
MSE_loss_history = []

for step in range(10000):

    time_start = random.randint(0,len(t)-300)
    time_end = time_start +300
    G_track_input = track[time_start:time_end-1]
    D_track_input = track[time_start+1:time_end]

    G_input = torch.from_numpy(G_track_input).float()
    D_input = torch.from_numpy(D_track_input).float()
    D_real = torch.zeros((time_end - time_start -1),8)
    D_feak = torch.zeros((time_end - time_start -1),8)
    G_output = G(G_input)

    for i in range((time_end - time_start -2)):
        D_real[i] = torch.cat([G_input[i],G_input[i+1]])
  
    for i in range((time_end - time_start -1)):
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

    if step == 1000:
      torch.save(G,'Double_pendulum_Generator1000.plk')
    if step == 2000:
      torch.save(G,'Double_pendulum_Generator2000.plk')
    if step == 3000:
      torch.save(G,'Double_pendulum_Generator3000.plk')
    if step == 4000:
      torch.save(G,'Double_pendulum_Generator4000.plk')
    if step == 5000:
      torch.save(G,'Double_pendulum_Generator5000.plk')
    if step == 6000:
      torch.save(G,'Double_pendulum_Generator6000.plk')  
    if step == 7000:
      torch.save(G,'Double_pendulum_Generator7000.plk')  
    if step == 8000:
      torch.save(G,'Double_pendulum_Generator8000.plk')  
    if step == 9000:
      torch.save(G,'Double_pendulum_Generator9000.plk')  
    if step == 9999:
      torch.save(G,'Double_pendulum_Generator10000.plk')  

    if step % 100 == 0:
        plt.cla()
        plt.plot(G_track_input[:,0], c='#FF0000', lw=1, label='Standard th1')
        plt.plot(G_track_input[:,1], c='#0000FF', lw=1, label='Standard th2')
        # plt.plot(G_track_input[:,2], c='#008000', lw=1, label='Standard v1')
        # plt.plot(G_track_input[:,3], c='#FFFF00', lw=1, label='Standard v2')
        plt.plot(G_output.data.numpy()[:,0], c='#FF00FF', lw=1, label='Generated th1', )         
        plt.plot(G_output.data.numpy()[:,1], c='#00BFFF', lw=1, label='Generated th2', )
        # plt.plot(G_output.data.numpy()[:,2], c='#7CFC00', lw=1, label='Generated v1', )         
        # plt.plot(G_output.data.numpy()[:,3], c='#FFA500', lw=1, label='Generated v2', )
        plt.legend(loc='upper right', fontsize=10)
        plt.ylim((-6, 6))
        plt.draw()
        plt.pause(0.01)
        

plt.ioff()
plt.show()

plt.plot(MSE_loss_history)
plt.show()
# torch.save(G,'Double_pendulum_Generator1.plk')
# print("network saved")