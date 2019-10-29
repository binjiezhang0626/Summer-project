import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos
import numpy as np
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
   
pendulum = DoublePendulum(1.0, 2.0, 1.0, 1.0)
g = 9.8
t = np.arange(0, 10, 0.01) 
th1, th2 = 1.0, 1.0
pendulum.init_status[:2] = th1, th2
track = double_pendulum_odeint(pendulum, t)

th1 = track[:,0]
th2 = track[:,1]
print(th1,th2)

l1 = 1.0
l2 = 1.0
x1 = []
y1 = []
x2 = []
y2 = []
for i in range(len(t)):
    xi = l1*sin(th1[i])   
    yi = - l1*cos(th1[i])
    xj = xi + l2*sin(th2[i])   
    yj = yi - l2*cos(th2[i])
    x1.append(xi)
    y1.append(yi)
    x2.append(xj)
    y2.append(yj)

for i in range(len(t)):
    plt.cla()
    plt.plot([0,x1[i]],[0,y1[i]],color = 'black')
    plt.plot(x1[i],y1[i],marker='o', markevery=15, color='red')
    plt.plot([x1[i],x2[i]],[y1[i],y2[i]],color = 'black')
    plt.plot(x2[i],y2[i],marker='o', markevery=15, color='blue')
    plt.xlim((-2.5, 2.5))
    plt.ylim((-2, 1))
    plt.draw()
    plt.pause(0.01)

plt.show()

