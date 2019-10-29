import random
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def lorenz(w, t, p, r, b):
    x, y, z = w
    return np.array([p*(y-x), x*(r-z)-y, x*y-b*z])

t = np.arange(0, 2, 0.01) 
track = odeint(lorenz, (0.0, 1.00, 0.0), t, args=(10.0, 28.0, 3.0))

G2 = torch.load('lorenz_Generator.plk')

input = track[0]
print(input)
G_input = torch.from_numpy(input).float()
G_output = [[0,1,0]]

for i in range(200):
    output = G2(G_input)
    output = output.data
    G_input = output
    output_numpy = output.numpy()
    output_list = output_numpy.tolist()
    G_output.append(output_list)

G_output= np.array(G_output)
plt.figure(1)
plt.plot(track[:,0])
plt.plot(track[:,1])
plt.plot(track[:,2])
plt.figure(2)
plt.plot(G_output[:,0])
plt.plot(G_output[:,1])
plt.plot(G_output[:,2])
plt.show()
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot(track[:,0], track[:,1], track[:,2])
# ax.plot(G_output[:,0], G_output[:,1], G_output[:,2])
# plt.show()