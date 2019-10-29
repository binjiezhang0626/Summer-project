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

t = np.arange(0, 40, 0.01)
track = odeint(rossler, (0.0, 2.0, 0.0), t, args=(0.2, 0.2, 5.7))

G2 = torch.load('Rossler_Generator.plk')

input = track[0]
G_input = torch.from_numpy(input).float()
G_output = [[0,2,0]]

for i in range(4000):
    output = G2(G_input)
    output = output.data
    G_input = output
    output_numpy = output.numpy()
    output_list = output_numpy.tolist()
    G_output.append(output_list)

G_output= np.array(G_output)
# plt.figure(1)
# plt.plot(track[:,0])
# plt.plot(track[:,1])
# plt.plot(track[:,2])
# plt.figure(2)
# plt.plot(G_output[:,0])
# plt.plot(G_output[:,1])
# plt.plot(G_output[:,2])
# plt.show()
x1 = track[:,0]
y1 = track[:,1]
z1 = track[:,2]
x2 = G_output[:,0]
y2 = G_output[:,1]
z2 = G_output[:,2]
print('1')
fig = plt.figure()
ax = Axes3D(fig)
for i in range (len(t)):
    ax.plot(x1[i],y1[i],z1[i],marker='o', markevery=15, color='red',)
    ax.plot(x2[i],y2[i],z2[i])
    plt.pause(0.01)

# ax.plot(track[:,0], track[:,1], track[:,2])
# ax.plot(G_output[:,0], G_output[:,1], G_output[:,2])
plt.show()