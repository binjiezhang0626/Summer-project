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

G2 = torch.load('Sample_pendulum_Generator2000.plk')

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


# print(loss(torch.from_numpy(state),torch.from_numpy(G_output)))
plt.figure()
plt.plot(state[:,0], label ='Standard Pendulum angular')
plt.plot(state[:,1], label ='Standard Pendulum velocity')
plt.plot(G_output[:,0], label = 'Generated Pendulum angular')
plt.plot(G_output[:,1], label = 'Generated Pendulum velocity')
plt.xlabel('Iteration')
plt.legend(loc='upper right', fontsize=10)
# plt.title('Standard pendulum track')
plt.ylim([-7,7])
plt.show()

# th1 = state[:,0]
# th2 = G_output[:,0]
# x1 = []
# x2 = []
# y1 = []
# y2 = []
# for i in range(len(t1)):
#     xi = l*sin(th1[i])   
#     yi = - l*cos(th1[i])
#     xj = l*sin(th2[i])   
#     yj = - l*cos(th2[i])
#     x1.append(xi)
#     y1.append(yi)
#     x2.append(xj)
#     y2.append(yj)

# for i in range(len(t1)):
#     plt.cla()
#     plt.figure(1)
#     plt.plot([0,x1[i]],[0,y1[i]],color = 'black')
#     plt.plot(x1[i],y1[i],marker='o', markevery=15, color='red',label = 'Standard Pendulum')
#     plt.plot([0,x2[i]],[0,y2[i]],color = 'black')
#     plt.plot(x2[i],y2[i],marker='o', markevery=15, color='blue',label = 'Generated Pendulum')
#     plt.xlim((-1, 1))
#     plt.ylim((-1.1, 0))
#     plt.title('Pendulum') 
#     plt.legend(loc='upper right', fontsize=10)
#     plt.pause(0.00001)
# plt.show()
