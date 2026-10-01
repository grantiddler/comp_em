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

        # self.epsilon_r[0:int(self.nx/ 3), :] = 4


        self.E_last = self.E
        self.H_x = self.H_x
        self.H_y = self.H_y

        self.sz_E = np.shape(self.E)
        self.sz_H_x = np.shape(self.H_x)
        self.sz_H_y = np.shape(self.H_y)

        self.x =  np.arange(x0, x1 + self.dx, self.dx) * np.ones([self.ny, 1])
        self.y =  np.transpose(np.transpose(np.arange(y0, y1 + self.dy, self.dy)) * np.ones([self.nx, 1]))


        self.diff_coeff_x = self.dt / (self.permeability * self.dy)
        self.diff_coeff_y = self.dt / (self.permeability * self.dx)

        self.epsilon_r += 8 * (np.sqrt(np.pow(self.x - .125,2) + np.pow(self.y - .2,2)) < 0.0075)
        self.PEC_mask = (np.sqrt(np.pow(self.x - .125,2) + np.pow(self.y - .055,2)) < 0.0075)
        self.permittivity = self.epsilon_0 * self.epsilon_r
        

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

        # print(E_down)F
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

        self.E_next = (self.alpha * self.E + (H_y_plus - H_y_minus) / self.dx - (H_x_plus - H_x_minus) / self.dy)

        # E Boundary conditions

        self.E_next[int(self.nx / 2), int(self.nx / 2)] -= self.source_profile((self.t + 0.5) * self.dt, 10, 4e10) / (self.dx * self.dy * 1000) 
        # self.E_next -= self.plane_wave((self.t + 0.5) * self.dt,0 , 5e8)
        self.E_next /= self.beta
        





        self.ABC(25, -1, "x")
        self.ABC(self.nx - 25, 1, "x")
        self.ABC(25, -1, "y")
        self.ABC(self.ny - 25, 1, "y")
        # self.ABC(25, -1, "y")
        # self.ABC(475, 1, "y")

        self.E_last = self.E
        self.E = self.E_next

        # self.E -= self.E * self.PEC_mask

        # self.plane_wave(self.t * self.dt, np.pi/3, 1e8)
        # plt.imshow(self.plane_wave(self.t * self.dt, 0, 1e8))
        # plt.show()
        


        # self.E[:, int(self.ny / 3)] = 0

        self.t += 1


    def ABC(self, M, d, ax):

        
        if(ax == "x"):

            self.E_next[M, :] = self.E[M-d, :] + (self.E[M, :] - self.E_next[M-d, :]) * (self.dx - self.c * self.dt) / (self.dx + self.c * self.dt)
            self.E_next[M+d,:] = 0 #self.E[-2,:]

        if(ax == "y"):

            self.E_next[:,M] = self.E[:,M-d] + (self.E[:,M] - self.E_next[:,M-d]) * (self.dx - self.c * self.dt) / (self.dx + self.c * self.dt)
            self.E_next[:,M+d] = 0 #self.E[-2,:]



        # B = 1
        # self.E_next[M,1:-1] += ((self.E_next[M - B,1:-1] - self.E[M - B,1:-1]) / (self.dx * self.dt))
        # self.E_next[M,1:-1] += (self.E[M,1:-1] * ((1/(self.dx * self.dt)) + (2/(self.c * np.pow(self.dt, 2))) - (self.c/(np.pow(self.dy, 2)))))
        # self.E_next[M,1:-1] -= (self.E_last[M,1:-1]/(self.c * (np.pow(self.dt, 2)))) 
        # self.E_next[M,1:-1] += (self.c/(2* np.pow(self.dy, 2))) * (self.E[M,0:-2] - self.E[M,2:])
        # self.E_next[M,1:-1] /= ((1/(self.dx * self.dt)) + (1/(self.c * np.pow(self.dy, 2))))
        # self.E_next[A,:] = .1
        
        
        # self.E_next[:,0] = 0 #self.E[-2,:]
        # self.E_next[:,-1] = 0 #self.E[-2,:]

    def source_profile(self, t, omega, scale):
        # return 1 - np.exp(-t * scale)
        
        return (1 - np.exp(- scale * t)) * np.sin(t * omega * scale) * np.exp(- np.pow( t * scale - 3, 2)) / 5

    def source_profile2(self, t, a, b):
   
        return (t > 0) * np.sin(t * 1e11) * (1 - np.exp( - t * 1e8)) / 5


    def plane_wave(self, t, angle, frequency):
        x0 = 100
        x1 = 400
        y0 = 100
        y1 = 400

        k = frequency / self.c
        kx = np.cos(angle)
        ky = np.sin(angle)

        
        J = np.zeros(self.sz_E)

        J[x0:x1, y0] = np.sin(t * frequency)
        J[x0:x1, y1] = np.sin(t * frequency + k * (y1 - y0) * self.dy )

        y = np.arange(y0, y1, 1)
        x = np.arange(x0, x1, 1)

        # J[x0, y0:y1] = np.sin(t * frequency + k * y * self.dy )
        # J[x1, y0:y1] = np.sin(t * frequency + k * y * self.dy )

        # plt.imshow(J)
        # plt.show()
        
        # self.E[x0:x1, y0] = self.source_profile(t * frequency + kx * y0 * self.dy + ky * x * self.dx, 5, 1e8)
        # self.E[x0:x1, y1] = self.source_profile(t * frequency + kx * y1 * self.dy + ky * x * self.dx, 5, 1e8)
        # self.E[x0, y0:y1] = self.source_profile(t * frequency + kx * y * self.dy + ky * x0 * self.dx, 5, 1e8)
        # self.E[x1, y0:y1] = self.source_profile(t * frequency + kx * y * self.dy + ky * x1 * self.dx, 5, 1e8)

        # self.E[x0:x1, y0:y1] = 0

        self.E[x0:x1, y0] = self.source_profile(t - (y0 * ky * self.dy + x * kx * self.dy) / self.c, 4, 1e8)
        self.E[x0:x1, y1] = self.source_profile(t - (y1 * ky * self.dy + x * kx * self.dy) / self.c, 4, 1e8)
        self.E[x0, y0:y1] = self.source_profile(t - (y * ky * self.dy + x0 * kx * self.dy) / self.c, 4, 1e8)
        self.E[x1, y0:y1] = self.source_profile(t - (y * ky * self.dy + x1 * kx * self.dy) / self.c, 4, 1e8)

        # plt.imshow(self.E)

        # plt.show()
        # plt.imshow(self.E)

        # plt.show()

        # plt.imshow(self.E)
# 
        # return J * (1 - np.exp(-1e6 * t)) * .01




        



# sim = FDTD_2D(0, 50, 1e-1, 0, 50, 1e-1)
# while True:
#     for i in range(20):
#         sim.update()

#     print(sim.source_profile((sim.t + 0.5) * sim.dt, 10, 1e7))
#     print(np.max(np.abs(sim.E)))

#     plt.imshow(np.transpose(sim.E), cmap=plt.colormaps["PuOr"], vmin=-0.25, vmax=0.25)
#     plt.colorbar()
#     plt.show()