from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import random
import numpy as np


A = np.ones([3,3])

sz = np.shape(A)
sz_border = sz + np.array([2,2])

B = np.zeros(sz_border)
B[1:-1, 1:-1] = A

print(A)
print(B)