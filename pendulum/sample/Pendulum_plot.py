import numpy as np
from math import sin
from math import cos
from math import acos
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pendulum_state(w, t, l):
    th, v = w
    dth = v
    dv  = - g/l * sin(th)
    return dth, dv

g = 9.8
t1 = np.arange(0,10,0.01)
th = 1.0
state = odeint(pendulum_state, (1.0, 0), t1, args=(1.0,))

th1 = state[:,0]

l = 1.0
x = []
y = []
for i in range(len(th1)):
    xi = l*sin(th1[i])   
    yi = - l*cos(th1[i])
    x.append(xi)
    y.append(yi)


for i in range(len(th1)):
    plt.cla()
    plt.plot([0,x[i]],[0,y[i]],color = 'black')
    plt.plot(x[i],y[i],marker='o', markevery=15, color='red')
    plt.xlim((-1, 1))
    plt.ylim((-1.1, 0))
    plt.draw()
    plt.pause(0.001)

plt.show()
