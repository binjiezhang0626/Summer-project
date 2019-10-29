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
l = 1.0
t1 = np.arange(0,10,0.01)
th = 1.0
state = odeint(pendulum_state, (1.0, 0), t1, args=(1.0,))

G2 = torch.load('Sample_pendulum_Generator.plk')

input = state[0]
G_input = torch.from_numpy(input).float()
G_output = [[1,0]]

for i in range(len(t1)-1):
    output = G2(G_input)
    output = output.data
    G_input = output
    output_numpy = output.numpy()
    output_list = output_numpy.tolist()
    G_output.append(output_list)

G_output= np.array(G_output)


print(loss(torch.from_numpy(state),torch.from_numpy(G_output)))
plt.plot(G_output)
plt.plot(state)
plt.show()
