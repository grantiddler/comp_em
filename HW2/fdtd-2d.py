import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



class FDTD_1D:
    def __init__(self, x0, x1, dx, y0, y1, dy, time_oversample = 1.5, forcing_type=[], forcing_function=[]):
        self.permittivity = 8.8541878188e-12
        self.permeability = 1.25663706127e-6
        self.conductivity = 0

        self.c = 1 / np.sqrt(self.permittivity * self.permeability)

        self.x0 = x0
        self.x1 = x1
        self.dx = dx

        self.y0 = y0
        self.y1 = y1
        self.dy = dy

        self.dt = 1 / (time_oversample * self.c * np.sqrt((1/ np.pow(dx, 2)) + (1/ np.pow(dy, 2))))

        self.t = 0 #
        
        self.nx = 2 * int((x1 - x0) / dx)
        self.ny = 2 * int((y1 - y0) / dy)

        self.E = np.zeros([self.nx, self.ny]) 
        self.H_x_offset = np.zeros([self.nx + 1, self.ny]) # H is offset -1/2 of an index from E in each direction
        self.H_y_offset = np.zeros([self.nx, self.ny + 1]) # H is offset -1/2 of an index from E in each direction

        self.E_last = self.E
        self.H_x_offset_last = self.H_x_offset
        self.H_y_offset_last = self.H_y_offset

        self.f_type = forcing_type
        self.f_func = forcing_function

        self.sz_E = np.shape(self.E)
        self.sz_H_x = np.shape(self.H_x_offset)
        self.sz_H_y = np.shape(self.H_y_offset)

        self.sz_E_x_border = self.sz_E + np.array([1,1])
        self.sz_E_y_border = self.sz_E + np.array([0,1])
        self.sz_H_x_border = self.sz_H_x + np.array([1,0])
        self.sz_H_y_border = self.sz_H_y + np.array([0,1])

        self.diff_coeff_y = self.dt / (self.permeability * self.dy)
        self.diff_coeff_x = self.dt / (self.permeability * self.dx)

        self.alpha = 1
        self.beta = 1


    def update_H(self):
        E_su = np.zeros(self.sz_E_y_border)
        E_sd = np.zeros(self.sz_E_y_border)

        E_sl = np.zeros(self.sz_E_x_border)
        E_sr = np.zeros(self.sz_E_x_border)

        E_sd[:,1:] = self.E
        E_su[:,0:-1] = self.E 

        E_sr[1:,:] = self.E
        E_sl[0:-1,:] = self.E

        self.H_x_offset = self.Hxy_offset - self.diff_coeff_x * (E_sl - E_sr)
        self.H_y_offset = self.H_y_offset - self.diff_coeff_y * (E_su - E_sd)


        # DO H BOUNDART CONDITIONS HERE

        self.E = (self.alpha * self.E + (self.H_x_offset[0:-1,:] - self.H_x_offset[1:,:]) / self.dx + (self.H_y_offset[:,0:-1] - self.H_y_offset[:,:1:]) / self.dy) / self.beta
 
        # DO E BOUNDART AND FORCING HERE

    