import random
import copy
import torch
import torch.nn as nn
import numpy as np

from math import sin
from math import cos
import matplotlib.pyplot as plt

x1 = np.arange(0, 2*np.pi, 0.01)
y1 = np.sin(x1)
z1 = np.zeros((len(x1),1))
z1[:,0] = y1

G2 = torch.load('Sin_Generator.plk')

input = z1[0]
G_input = torch.from_numpy(input).float()
G_output = [[0]]
for i in range(300):
    output = G2(G_input)
    output = output.data
    G_input = output
    output_numpy = output.numpy()
    output_list = output_numpy.tolist()
    G_output.append(output_list)

G_output= np.array(G_output)

plt.plot(z1)

plt.plot(G_output)
plt.show()