import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import sin
from math import cos
from math import acos
from scipy.integrate import odeint

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
g = 9.8
t = np.arange(0, 10, 0.01) 
th1, th2 = 1.0, 1.0
pendulum.init_status[:2] = th1, th2
track = double_pendulum_odeint(pendulum, t)

G3 = torch.load('Double_pendulum_Generator10000.plk')

input = track[0]
G_input = torch.from_numpy(input).float()
G_output = [[1,1,0,0]]

for i in range(len(t)-1):
    output = G3(G_input)
    output = output.data
    G_input = output
    output_numpy = output.numpy()
    output_list = output_numpy.tolist()
    G_output.append(output_list)

G_output= np.array(G_output)

# print(loss(torch.from_numpy(track),torch.from_numpy(G_output)))

plt.figure(1)
plt.plot(track[:,0],label = 'th1')
plt.plot(track[:,1],label = 'th2')
plt.plot(track[:,2],label = 'v1')
plt.plot(track[:,3],label = 'v2')
plt.xlabel('Iteration')
plt.title('Standard double pendulum track')
plt.legend(loc='upper right', fontsize=10)

plt.figure(2)
plt.plot(G_output[:,0],label = 'th1')
plt.plot(G_output[:,1],label = 'th2')
plt.plot(G_output[:,2],label = 'v1')
plt.plot(G_output[:,3],label = 'v2')
plt.xlabel('Iteration')
plt.title('Generated double pendulum track')
plt.legend(loc='upper right', fontsize=10)
plt.show()
# th1 = track[:,0]
# th2 = track[:,1]
# th3 = G_output[:,0]
# th4 = G_output[:,1]

# l1 = 1.0
# l2 = 1.0
# x1 = []
# y1 = []
# x2 = []
# y2 = []
# x3 = []
# y3 = []
# x4 = []
# y4 = []
# for i in range(len(t)):
#     xa = l1*sin(th1[i])   
#     ya = - l1*cos(th1[i])
#     xb = xa + l2*sin(th2[i])   
#     yb = ya - l2*cos(th2[i]) 
#     xc = l1*sin(th3[i])   
#     yc = - l1*cos(th3[i])
#     xd = xc + l2*sin(th4[i])   
#     yd = yc - l2*cos(th4[i])
#     x1.append(xa)
#     y1.append(ya)
#     x2.append(xb)
#     y2.append(yb)
#     x3.append(xc)
#     y3.append(yc)
#     x4.append(xd)
#     y4.append(yd)

# for i in range(len(t)):
#     plt.cla()
#     plt.plot([0,x1[i]],[0,y1[i]],color = 'black')
#     plt.plot(x1[i],y1[i],marker='o', markevery=15, color='red', label = 'Standard Double Pendulum')
#     plt.plot([x1[i],x2[i]],[y1[i],y2[i]],color = 'black')
#     plt.plot(x2[i],y2[i],marker='o', markevery=15, color='red')
#     plt.plot([0,x3[i]],[0,y3[i]],color = 'black')
#     plt.plot(x3[i],y3[i],marker='o', markevery=15, color='blue', label = 'Generated Double Pendulum')
#     plt.plot([x3[i],x4[i]],[y3[i],y4[i]],color = 'black')
#     plt.plot(x4[i],y4[i],marker='o', markevery=15, color='blue')
#     plt.xlim((-2.5, 2.5))
#     plt.ylim((-2, 0.5))
#     plt.title('Double Pendulum') 
#     plt.legend(loc='upper right', fontsize=10)
#     plt.draw()
#     plt.pause(0.01)

# plt.show()