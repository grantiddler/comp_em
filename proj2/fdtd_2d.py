import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



class FDTD_2D:
    def __init__(self, x0, x1, dx, y0, y1, dy, time_oversample = 1.5, forcing_type=[], forcing_function=[]):
        self.epsilon_0 = 8.8541878188e-12
        self.permeability = 1.25663706127e-6
        self.conductivity = 0

        self.c = 1 / np.sqrt(self.epsilon_0 * self.permeability)

        self.x0 = x0
        self.x1 = x1
        self.dx = dx

        self.y0 = y0
        self.y1 = y1
        self.dy = dy

        self.dt = 1 / (time_oversample * self.c * np.sqrt((1/ np.pow(dx, 2)) + (1/ np.pow(dy, 2))))

        self.t = 0 #

        self.f_type = forcing_type
        self.f_func = forcing_function
        
        self.nx = 1 + int((x1 - x0) / dx)
        self.ny = 1 + int((y1 - y0) / dy)

        self.E = np.zeros([self.nx, self.ny]) 
        self.H_y = np.zeros([self.nx - 1, self.ny]) # H is offset -1/2 of an index from E in each direction
        self.H_x = np.zeros([self.nx, self.ny - 1]) # H is offset -1/2 of an index from E in each direction

        self.epsilon_r = np.ones([self.nx, self.ny]) 
        self.conductivity = np.zeros([self.nx, self.ny]) 

        self.epsilon_r[0:int(self.nx/ 3), :] = 3

        self.permittivity = self.epsilon_0 * self.epsilon_r

        self.E_last = self.E
        self.H_x = self.H_x
        self.H_y = self.H_y

        self.sz_E = np.shape(self.E)
        self.sz_H_x = np.shape(self.H_x)
        self.sz_H_y = np.shape(self.H_y)


        self.diff_coeff_x = self.dt / (self.permeability * self.dy)
        self.diff_coeff_y = self.dt / (self.permeability * self.dx)

        self.alpha = self.permittivity / self.dt - self.conductivity / 2
        self.beta = self.permittivity / self.dt + self.conductivity / 2



    def update(self):
        # E_yp = np.zeros([self.nx, self.ny + 1])
        # E_ym = np.zeros([self.nx, self.ny + 1])
        # E_xm = np.zeros([self.nx + 1, self.ny])
        # E_xp = np.zeros([self.nx + 1, self.ny])

        # E_yp[:,0:-1] = self.E
        # E_ym[:,1:] = self.E

        # E_xp[0:-1,:] = self.E
        # E_xm[1:,:] = self.E

        # print(E_down)
        # print(E_up)
        # print(E_left)
        # print(E_right)
        self.H_x = self.H_x + self.diff_coeff_x * (self.E[:,0:-1] - self.E[:,1:])
        self.H_y = self.H_y - self.diff_coeff_y * (self.E[0:-1, :] - self.E[1:,:])

        H_y_plus = np.zeros(np.shape(self.E))
        H_y_minus = np.zeros(np.shape(self.E))

        H_x_plus = np.zeros(np.shape(self.E))
        H_x_minus = np.zeros(np.shape(self.E))
        
        H_y_plus[0:-1,:] = self.H_y
        H_y_minus[1:,:] = self.H_y

        H_x_plus[:, 0:-1] = self.H_x
        H_x_minus[:, 1:] = self.H_x

        self.E = (self.alpha * self.E + (H_y_plus - H_y_minus) / self.dx - (H_x_plus - H_x_minus) / self.dy)

        # E Boundary conditions

        self.E[int(self.nx / 2), int(self.nx / 2)] -= self.source_profile((self.t + 0.5) * self.dt, 200, 1e7)
        
        self.E /= self.beta
        # self.E[:, int(self.ny / 3)] = 0

        self.t += 1

    def source_profile(self, t, omega, scale):
        # return 1 - np.exp(-t * scale)
        return (1 - np.exp(- scale * t)) * np.cos(t * omega * scale) * np.exp(- np.pow( t * scale - 3, 2))


# sim = FDTD_2D(0, 50, 1e-1, 0, 50, 1e-1)
# while True:
#     for i in range(20):
#         sim.update()

#     print(sim.source_profile((sim.t + 0.5) * sim.dt, 10, 1e7))
#     print(np.max(np.abs(sim.E)))

#     plt.imshow(np.transpose(sim.E), cmap=plt.colormaps["PuOr"], vmin=-0.25, vmax=0.25)
#     plt.colorbar()
#     plt.show()