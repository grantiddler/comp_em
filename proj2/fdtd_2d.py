import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



class FDTD_2D:
    def __init__(self, x0, x1, dx, y0, y1, dy, time_oversample = 1.5, forcing_type=[], forcing_function=[]):
        self.epsilon_0 = 8.8541878188e-12
        self.permeability = 1.25663706127e-6

        self.Z0 = np.sqrt(self.permeability / self.epsilon_0)

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

        self.PEC_mask = np.zeros([self.nx, self.ny]) 


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

        self.PEC_mask = (np.sqrt(np.pow(self.x - .0625,2) + np.pow(self.y - .0625,2)) < 0.015)
        # self.epsilon_r += 8 * (np.sqrt(np.pow(self.x - .0625,2) + np.pow(self.y - .0625,2)) < 0.015)
        self.permittivity = self.epsilon_0 * self.epsilon_r
        

        self.alpha = self.permittivity / self.dt - self.conductivity / 2
        self.beta = self.permittivity / self.dt + self.conductivity / 2

        # scattering stuff
        self.huygens_x0 = 250
        self.huygens_x1 = self.nx - 250
        self.huygens_y0 = 250
        self.huygens_y1 = self.ny - 250

        self.wave_origin_x = 0
        self.wave_origin_y = 0
        self.angle = np.pi/3





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
        self.H_x_next = self.H_x + self.diff_coeff_x * (self.E[:,0:-1] - self.E[:,1:])
        self.H_y_next = self.H_y - self.diff_coeff_y * (self.E[0:-1, :] - self.E[1:,:])

        self.plane_wave_pt1(self.t * self.dt, self.angle, 1e8)

        self.H_x = self.H_x_next
        self.H_y = self.H_y_next

        H_y_plus = np.zeros(np.shape(self.E))
        H_y_minus = np.zeros(np.shape(self.E))

        H_x_plus = np.zeros(np.shape(self.E))
        H_x_minus = np.zeros(np.shape(self.E))
        
        H_y_plus[0:-1,:] = self.H_y
        H_y_minus[1:,:] = self.H_y

        H_x_plus[:, 0:-1] = self.H_x
        H_x_minus[:, 1:] = self.H_x

        self.E_next = (self.alpha * self.E + (H_y_plus - H_y_minus) / self.dx - (H_x_plus - H_x_minus) / self.dy)

        


        self.plane_wave_pt2(self.t * self.dt, self.angle, 1e8)

        # E Boundary conditions

        # self.E_next[int(self.nx / 2), int(self.nx / 2)] -= self.source_profile2((self.t + 0.5) * self.dt, 10, 4e10) / (self.dx * self.dy * 1000) 
        # self.E_next -= self.plane_wave((self.t + 0.5) * self.dt,0 , 5e8)
        self.E_next /= self.beta

        self.ABC(25, -1, "x")
        self.ABC(self.nx - 25, 1, "x")
        self.ABC(25, -1, "y")
        self.ABC(self.ny - 25, 1, "y")

        self.E_last = self.E
        self.E = self.E_next

        self.E -= self.E * self.PEC_mask




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



    def source_profile(self, t, omega, scale):
        # return 1 - np.exp(-t * scale)
        omega = 10
        scale = 1e10
        return (t > 0) * (1 - np.exp(- scale * t)) * np.cos(t * omega * scale) * np.exp(- np.pow( t * scale - 3, 2)) / 5

    def source_profile2(self, t, a, b):

        return (t > 0) * np.sin(t * 1e11) * (1 - np.exp( - t * 1e8)) / 5


    def plane_wave_pt1(self, t, angle, frequency):
        x0 = self.huygens_x0 
        x1 = self.huygens_x1 
        y0 = self.huygens_y0 
        y1 = self.huygens_y1 

        k = frequency / self.c
        kx = np.cos(angle)
        ky = np.sin(angle)

        y = np.arange(y0, y1 + 1, 1) * self.dy
        x = np.arange(x0, x1 + 1, 1) * self.dx

        # [x0, y0:y1]


        E_inc = self.source_profile(t - ((y - self.wave_origin_y) * ky + (x0 * self.dx - self.wave_origin_x) * kx) / self.c, 0, 1e10)
        self.H_x_next[x0 - 1, y0:y1] = self.H_x[x0 - 1, y0:y1] + self.diff_coeff_x * ((self.E[x0 - 1, y0:y1] - E_inc[0:-1]) - (self.E[x0 - 1, y0+1:y1+1] - E_inc[1:]))
        self.H_y_next[x0 - 1, y0:y1] = self.H_y[x0 - 1, y0:y1] - self.diff_coeff_y * ((self.E[x0 - 1, y0:y1] - E_inc[0:-1]) - self.E[x0, y0:y1])


        E_inc = self.source_profile(t - ((y - self.wave_origin_y) * ky + (x1 * self.dx - self.wave_origin_x) * kx) / self.c, 10, 1e10)
        self.H_x_next[x1, y0:y1] = self.H_x[x1, y0:y1] + self.diff_coeff_x * ((self.E[x1, y0:y1] - E_inc[0:-1]) - (self.E[x1, y0+1:y1+1] - E_inc[1:]))
        self.H_y_next[x1, y0:y1] = self.H_y[x1, y0:y1] - self.diff_coeff_y * ((self.E[x1, y0:y1]) - (self.E[x1 + 1, y0:y1] - E_inc[0:-1]))


        E_inc = self.source_profile(t - ((y0 * self.dy - self.wave_origin_y) * ky + (x - self.wave_origin_x) * kx) / self.c, 10, 1e10)
        self.H_x_next[x0:x1, y0] = self.H_x[x0:x1, y0] + self.diff_coeff_x * ((self.E[x0:x1, y0] - E_inc[0:-1]) - self.E[x0:x1, y0 + 1])
        self.H_y_next[x0:x1, y0] = self.H_y[x0:x1, y0] - self.diff_coeff_y * ((self.E[x0:x1, y0] - E_inc[0:-1]) - (self.E[x0+1:x1+1, y0] - E_inc[1:]))


        E_inc = self.source_profile(t - ((y1 * self.dy - self.wave_origin_y) * ky + (x - self.wave_origin_x) * kx) / self.c, 10, 1e10)
        self.H_x_next[x0:x1, y1 - 1] = self.H_x[x0:x1, y1 - 1] + self.diff_coeff_x * ((self.E[x0:x1, y1 - 1]) - (self.E[x0:x1, y1]  - E_inc[0:-1]))
        self.H_y_next[x0:x1, y1] = self.H_y[x0:x1, y1] - self.diff_coeff_y * ((self.E[x0:x1, y1] - E_inc[0:-1]) - (self.E[x0+1:x1+1, y1] - E_inc[1:]))
        
        


    
    def plane_wave_pt2(self, t, angle, frequency):
        x0 = self.huygens_x0 
        x1 = self.huygens_x1 
        y0 = self.huygens_y0 
        y1 = self.huygens_y1 

        k = frequency / self.c
        kx = np.cos(angle)
        ky = np.sin(angle)

        y = np.arange(y0, y1, 1) * self.dy
        x = np.arange(x0, x1, 1) * self.dx


        lazy = self.E_next[x0-1:x0+1, y0]
        # WHEN H<- is on surf.
        # [x0, y0:y1]
        H_y_plus = self.H_y[x0, y0:y1]
        H_y_minus = self.H_y[x0 - 1, y0:y1]
        H_y = H_y_minus + (kx > 0) * kx * self.source_profile(t - ((y - self.wave_origin_y) * ky + ((x0- 1/2) * self.dx - self.wave_origin_x) * kx) / self.c, 10, 1e10) / (self.Z0 )

        H_x_plus = self.H_x[x0, y0:y1]
        H_x_minus = self.H_x[x0, y0-1:y1-1]

        self.E_next[x0, y0:y1] = (self.alpha[x0, y0:y1] * self.E[x0, y0:y1] + (H_y_plus - H_y) / self.dx - (H_x_plus - H_x_minus) / self.dy) 
        

        # WHEN H<- is on surf.
        # [x0, y0:y1]
        H_y_plus = self.H_y[x1, y0:y1]
        H_y_minus = self.H_y[x1 - 1, y0:y1]
        H_y = H_y_plus + kx * self.source_profile(t - ((y - self.wave_origin_y) * ky + ((x1 + 1/2) * self.dx - self.wave_origin_x) * kx) / self.c, 10, 1e10) / (self.Z0 )

        H_x_plus = self.H_x[x1, y0:y1]
        H_x_minus = self.H_x[x1, y0-1:y1-1]

        self.E_next[x1, y0:y1] = (self.alpha[x1, y0:y1] * self.E[x1, y0:y1] + (H_y - H_y_minus) / self.dx - (H_x_plus - H_x_minus) / self.dy) 
        
        # [x0:x1, y0]

        H_y_plus = self.H_y[x0:x1, y0]
        H_y_minus = self.H_y[x0-1:x1-1, y0]

        H_x_plus = self.H_x[x0:x1, y0]
        H_x_minus = self.H_x[x0:x1, y0-1]
        H_x = H_x_minus - ky * self.source_profile(t - (((y0 - 1/2) * self.dy - self.wave_origin_y) * ky + (x - self.wave_origin_x) * kx) / self.c, 10, 1e10) / (self.Z0 )

        self.E_next[x0:x1, y0] = (self.alpha[x0:x1, y0] * self.E[x0:x1, y0] + (H_y_plus - H_y_minus) / self.dx - (H_x_plus - H_x) / self.dy)

        H_y_plus = self.H_y[x0:x1, y1]
        H_y_minus = self.H_y[x0-1:x1-1, y1]

        H_x_plus = self.H_x[x0:x1, y1]
        H_x_minus = self.H_x[x0:x1, y1-1]
        H_x = H_x_plus - ky * self.source_profile(t - (((y1 + 1/2) * self.dy - self.wave_origin_y) * ky + (x - self.wave_origin_x) * kx) / self.c, 10, 1e10) / (self.Z0 )

        self.E_next[x0:x1, y1] = (self.alpha[x0:x1, y1] * self.E[x0:x1, y1] + (H_y_plus - H_y_minus) / self.dx - (H_x - H_x_minus) / self.dy)

        self.E_next[x0-1:x0+1, y0] = lazy #lazy workaround because these 2 points are getting updated here but should not be and I am too tired to figrue out the indexing
        
        # self.E_next[x0:x1, y0] = -self.source_profile(t - ((y0 * self.dy - self.wave_origin_y) * ky + (x - self.wave_origin_x) * kx) / self.c, 10, 1e10)
        # # self.E_next[x0, y0:y1] = -self.source_profile(t - ((y - self.wave_origin_y) * ky + (x0 * self.dx - self.wave_origin_x) * kx) / self.c, 10, 1e10)
        # self.E_next[x0:x1, y1] = -self.source_profile(t - ((y1 * self.dy - self.wave_origin_y) * ky + (x - self.wave_origin_x) * kx) / self.c, 10, 1e10)
        # self.E_next[x1, y0:y1] = -self.source_profile(t - ((y - self.wave_origin_y) * ky + (x1 * self.dx - self.wave_origin_x) * kx) / self.c, 10, 1e10)

        # self.E = self.source_profile(t - ((self.y - self.wave_origin_y) * ky + (self.x - self.wave_origin_x) * kx) / self.c, 10, 1e10)

        # self.E = self.source_profile( t - ((self.y - self.wave_origin_y) * ky * self.dy +  * kx * self.dx) / self.c, 10, 1e10)