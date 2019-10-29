import random
import numpy as np
from math import cos
import matplotlib.pyplot as plt

th = np.arange(0,2*np.pi, 0.01)
x = np.cos(th)
y = np.sin(th)

l = 1
z = np.zeros((len(th),2))
z[:,0] = x
z[:,1] = y
plt.plot(z)
plt.show()
