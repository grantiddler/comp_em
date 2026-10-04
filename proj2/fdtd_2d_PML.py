import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import jv
from scipy.special import hankel2
from scipy.special import jvp
from scipy.special import h2vp
from matplotlib.patches import Circle, Rectangle



class FDTD_2D:
    def __init__(self, scatterer, x0, x1, dx, y0, y1, dy, time_oversample = 1.5, pml_cells = 10, sigma_max = 0.07, p = 3, forcing_type=[], forcing_function=[]):
        self.scatterer = scatterer
        self.epsilon_0 = 8.8541878188e-12
        self.permeability = 1.25663706127e-6
        self.conductivity = 0
        self.c = 1 / np.sqrt(self.epsilon_0 * self.permeability)
        self.probe_history = [] #probe test
        self.E0 = 1

        self.Z0 = np.sqrt(self.permeability / self.epsilon_0)



        #----Cylinder Variables----#
        self.f0 =  10 * 10**9
        self.a = 0.015 # cylinder radius in m
        self.epsilon_rc = 9 # e_r of cylinder
        self.lamda0 = self.c/self.f0
        self.k0 = (2 * np.pi) / self.lamda0
        self.kd = self.k0 * np.sqrt(self.epsilon_rc)


        self.phi = np.linspace(0, 2*np.pi, 361)
        n = np.arange(-50, 51) # 50 azimuthal harmonics


        self.nphi = np.outer(n, self.phi)

        # dielectric A_n
        

        self.x0 = x0
        self.x1 = x1
        self.dx = dx

        self.y0 = y0
        self.y1 = y1
        self.dy = dy


        #cylcinder center
        self.xc = self.x0+(self.x1-self.x0)/2
        self.yc = self.y0+(self.y1-self.y0)/2

        self.dt = 1 / (time_oversample * self.c * np.sqrt((1/ np.pow(dx, 2)) + (1/ np.pow(dy, 2))))

        self.t = 0 

        self.f_type = forcing_type
        self.f_func = forcing_function
        
        self.nx = 1 + int((x1 - x0) / dx) #sets grid points for x
        self.ny = 1 + int((y1 - y0) / dy) #sets grid points for y

        self.x = np.linspace(x0, x1, self.nx)# 1D x cord array
        self.y = np.linspace(y0, y1, self.ny)# 1D y cord array

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij') #arrays with shape (nx, ny)

        self.rho = np.sqrt((self.X-self.xc)**2+(self.Y-self.yc)**2) # the distance of each grid point from the cylinder center

        self.PEC_mask = self.rho <= self.a 

        self.pml_cells = pml_cells #variable for outer cell thickness of pml

        #describes the loss of the PML in x and y directions
        self.sigma_x = np.zeros([self.nx, self.ny])
        self.sigma_y = np.zeros([self.nx, self.ny])

        self.epsilon_r = np.ones([self.nx, self.ny]) 
        self.conductivity = np.zeros([self.nx, self.ny]) 

        #self.epsilon_r[0:int(self.nx/ 3), :] = 3

        self.permittivity = self.epsilon_0 * self.epsilon_r

        for i in range(self.pml_cells):
            depth = (self.pml_cells-i)/self.pml_cells #gradient of depth from edges in
            sigma = sigma_max * depth**p

            self.sigma_x[i, :] = sigma #left most x cells
            self.sigma_x[-i-1, :] =  sigma #right most x cells

            self.sigma_y[:, i] = sigma #bottom y cells
            self.sigma_y[:, -i-1] = sigma #top y cells

        #spacial angle for 2D
        self.phi_grid = np.arctan2(self.Y-self.yc, self.X-self.xc) 

        # masks for rho<a and rho>=a for when inside and outside the cylinder
        outside = self.rho >= self.a
        inside = self.rho < self.a
        
        #incident feild everywhere
        self.E_i = self.E0*np.exp(-1j*self.k0*self.X) 

        # Create empty scattered-field array
        self.E_s = np.zeros_like(self.E_i, dtype=complex)

        # Harmonic index for masked 1-D spatial arrays
        n_col = n[:, None]



        if self.scatterer == 'PEC':
            self.A_n = jv(n, self.k0*self.a)/hankel2(n, self.k0*self.a) #PEC A_n
            self.E_s[inside] = -self.E_i[inside] # E tot will become 0 inside PEC
        elif self.scatterer == 'dielectric':
            self.A_n = ((jvp(n, self.k0*self.a)*jv(n, self.kd*self.a)) - (np.sqrt(self.epsilon_rc)*jv(n, self.k0*self.a)*jvp(n, self.kd*self.a))
                                ) / ((h2vp(n, self.k0*self.a)*jv(n, self.kd*self.a))- (np.sqrt(self.epsilon_rc)*hankel2(n, self.k0*self.a)*jvp(n, self.kd*self.a))) 
        else:
            print('Invalid Input')
            return


            
        self.sigma_2D = 4/self.k0 * np.abs(np.sum(self.A_n[:, None] * np.exp(1j*self.nphi), axis = 0))**2 # axis = 0 gives same number of elements as self.phi


        # Scattered feild
        self.E_s[outside] = -self.E0*(np.sum((-1j)**n_col * self.A_n[:, None] * hankel2(n_col, self.k0*self.rho[outside])*np.exp(1j*n_col*self.phi_grid[outside]), axis=0)) #scattering wave

        



        self.E_tot = self.E_i +self.E_s



        self.E = np.zeros([self.nx, self.ny]) 
        self.E_zx = np.zeros([self.nx, self.ny])
        self.E_zy = np.zeros([self.nx, self.ny])

        # avergaes neighbor values to find halfway point between E values to find where H lives 
        #sigma is defined on E grid so define sigma_H bewteen E values
        self.sigma_x_Hy = 0.5 * (self.sigma_x[:-1, :] + self.sigma_x[1:, :])
        self.sigma_y_Hx = 0.5 * (self.sigma_y[:, :-1] + self.sigma_y[:, 1:])

        #----Y variables for H----
        # if sigma_x_Hy = 0 there is no PML damping;
        # if sigma_x_Hy > 0 inside the PML, H_y is damped        
        self.A_Hy = (1 - self.sigma_x_Hy * self.dt / (2 * self.epsilon_0)
                     ) / (1 + self.sigma_x_Hy * self.dt / (2 * self.epsilon_0)) # turns off in interior and on at PML boundary as sigma=0

        self.B_Hy = (self.dt / self.permeability
                     ) / (1 + self.sigma_x_Hy * self.dt / (2 * self.epsilon_0)) # turns off in interior and on at PML boundary

        # X variables for H
        self.A_Hx = (1 - self.sigma_y_Hx * self.dt / (2 * self.epsilon_0)
                     ) / (1 + self.sigma_y_Hx * self.dt / (2 * self.epsilon_0)) # turns off in interior and on at PML boundary

        self.B_Hx = (self.dt / self.permeability         #B_Hx equation
                     ) / (1 + self.sigma_y_Hx * self.dt / (2 * self.epsilon_0)) # turns off in interior and on at PML boundary
        
        #----X variables for E----
        #sigma defined on E grid
        self.A_Ex = (1 - self.sigma_x * self.dt / (2*self.permittivity) # 1 in interior and <1 (damping) in PML boundary
                     ) / (1 + self.sigma_x * self.dt / (2*self.permittivity)) 

        self.B_Ex = (self.dt / (self.permittivity)
                     ) / (1 + self.sigma_x * self.dt / (2*self.permittivity)) 

        #Y variables for E
        self.A_Ey = (1- self.sigma_y * self.dt / (2*self.permittivity)
                     ) / (1 + self.sigma_y * self.dt / (2*self.permittivity)) 

        self.B_Ey = (self.dt / (self.permittivity)
                     ) / (1 + self.sigma_y * self.dt / (2*self.permittivity)) 


        self.H_y = np.zeros([self.nx - 1, self.ny]) # H is offset -1/2 of an index from E in each direction
        self.H_x = np.zeros([self.nx, self.ny - 1]) # H is offset -1/2 of an index from E in each direction


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


        self.huygens_x0 = 50
        self.huygens_x1 = self.nx - 50
        self.huygens_y0 = 50
        self.huygens_y1 = self.ny - 50

        self.wave_origin_x = 0
        self.wave_origin_y = 0
        self.angle = np.pi/4

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
        self.H_x_next = (self.A_Hx * self.H_x + self.B_Hx * (self.E[:, :-1] - self.E[:, 1:]) / self.dy)
        self.H_y_next = (self.A_Hy * self.H_y + self.B_Hy * (self.E[1:, :] - self.E[:-1, :]) / self.dx)

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

        self.E_zx_next = (self.A_Ex * self.E_zx + self.B_Ex * ( H_y_plus -  H_y_minus) / self.dx)
        self.E_zy_next = (self.A_Ey * self.E_zy - self.B_Ey * ( H_x_plus -  H_x_minus) / self.dy)

        self.plane_wave_pt2(self.t * self.dt, self.angle, 1e8)

        self.E_zx = self.E_zx_next
        self.E_zy = self.E_zy_next

        

        #test source
        source = self.source_profile(
            (self.t + 0.5) * self.dt
        )

        ix = int(self.nx / 4)
        iy = int(self.ny / 2)

        # self.E_zx[ix, iy] -= 0.5 * source
        # self.E_zy[ix, iy] -= 0.5 * source




        self.E = self.E_zx + self.E_zy # combine split feilds
        # self.E = self.E_next

        if self.scatterer == 'PEC':
            self.E -= self.E * self.PEC_mask
            

        elif self.scatterer == 'dielectric':
            return
        else:
            print('not valid input')


        self.t += 1

    def source_profile(self, t):
        f0 = self.f0       # 300 MHz
        t0 = 0.5e-9      # pulse centered at 3 ns
        tau = 0.15e-9     # pulse width

        return (
            np.sin(2 * np.pi * f0 * t)
            * np.exp(-((t - t0) / tau)**2)
        )
    

    def ABC(self, M, d, ax):

        
        if(ax == "x"):

            self.E_next[M, :] = self.E[M-d, :] + (self.E[M, :] - self.E_next[M-d, :]) * (self.dx - self.c * self.dt) / (self.dx + self.c * self.dt)
            self.E_next[M+d,:] = 0 #self.E[-2,:]

        if(ax == "y"):

            self.E_next[:,M] = self.E[:,M-d] + (self.E[:,M] - self.E_next[:,M-d]) * (self.dx - self.c * self.dt) / (self.dx + self.c * self.dt)
            self.E_next[:,M+d] = 0 #self.E[-2,:]


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

        # self.H_x_next = (self.A_Hx * self.H_x + self.B_Hx * (self.E[:, :-1] - self.E[:, 1:]) / self.dy)
        # self.H_y_next = (self.A_Hy * self.H_y + self.B_Hy * (self.E[1:, :] - self.E[:-1, :]) / self.dx)

        E_inc = self.source_profile(t - ((y - self.wave_origin_y) * ky + (x0 * self.dx - self.wave_origin_x) * kx) / self.c)
        self.H_x_next[x0 - 1, y0:y1] = self.H_x[x0 - 1, y0:y1] + self.diff_coeff_x * ((self.E[x0 - 1, y0:y1] - E_inc[0:-1]) - (self.E[x0 - 1, y0+1:y1+1] - E_inc[1:]))
        self.H_y_next[x0 - 1, y0:y1] = self.H_y[x0 - 1, y0:y1] - self.diff_coeff_y * ((self.E[x0 - 1, y0:y1] - E_inc[0:-1]) - self.E[x0, y0:y1])


        E_inc = self.source_profile(t - ((y - self.wave_origin_y) * ky + (x1 * self.dx - self.wave_origin_x) * kx) / self.c)
        self.H_x_next[x1, y0:y1] = self.H_x[x1, y0:y1] + self.diff_coeff_x * ((self.E[x1, y0:y1] - E_inc[0:-1]) - (self.E[x1, y0+1:y1+1] - E_inc[1:]))
        self.H_y_next[x1, y0:y1] = self.H_y[x1, y0:y1] - self.diff_coeff_y * ((self.E[x1, y0:y1]) - (self.E[x1 + 1, y0:y1] - E_inc[0:-1]))


        E_inc = self.source_profile(t - ((y0 * self.dy - self.wave_origin_y) * ky + (x - self.wave_origin_x) * kx) / self.c)
        self.H_x_next[x0:x1, y0] = self.H_x[x0:x1, y0] + self.diff_coeff_x * ((self.E[x0:x1, y0] - E_inc[0:-1]) - self.E[x0:x1, y0 + 1])
        self.H_y_next[x0:x1, y0] = self.H_y[x0:x1, y0] - self.diff_coeff_y * ((self.E[x0:x1, y0] - E_inc[0:-1]) - (self.E[x0+1:x1+1, y0] - E_inc[1:]))


        E_inc = self.source_profile(t - ((y1 * self.dy - self.wave_origin_y) * ky + (x - self.wave_origin_x) * kx) / self.c)
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


        # [x0, y0:y1]
        
        H_y_plus = self.H_y[x0, y0:y1]
        H_y_minus = self.H_y[x0-1, y0:y1]

        H_x_plus = self.H_x[x0, y0:y1]
        H_x_minus = self.H_x[x0, y0-1:y1-1]

        H_y = H_y_minus + (kx > 0) * kx * self.source_profile(t - ((y - self.wave_origin_y) * ky + ((x0- 1/2) * self.dx - self.wave_origin_x) * kx) / self.c) / (self.Z0 )

        self.E_zx_next[x0, y0:y1] = (self.A_Ex[x0, y0:y1] * self.E_zx[x0, y0:y1] + self.B_Ex[x0, y0:y1] * ( H_y_plus -  H_y) / self.dx)
        self.E_zy_next[x0, y0:y1] = (self.A_Ey[x0, y0:y1] * self.E_zy[x0, y0:y1] - self.B_Ey[x0, y0:y1] * ( H_x_plus -  H_x_minus) / self.dy)



        H_y_plus = np.zeros(np.shape(self.E))
        H_y_minus = np.zeros(np.shape(self.E))

        H_x_plus = np.zeros(np.shape(self.E))
        H_x_minus = np.zeros(np.shape(self.E))
        
        
        H_y_plus = self.H_y[x0:x1, y0]
        H_y_minus = self.H_y[x0-1:x1-1, y0]

        H_x_plus = self.H_x[x0:x1, y0]
        H_x_minus = self.H_x[x0:x1, y0-1]
        

        # [x0:x1, y0]
        self.E_zx_next[x0:x1, y0] = (self.A_Ex[x0:x1, y0] * self.E_zx[x0:x1, y0] + self.B_Ex[x0:x1, y0] * (H_y_plus - H_y_minus) / self.dx)
        self.E_zy_next[x0:x1, y0] = (self.A_Ey[x0:x1, y0] * self.E_zy[x0:x1, y0] - self.B_Ey[x0:x1, y0] * (H_x_plus - H_x_minus) / self.dy)
       



# sim = FDTD_2D('PEC', 0, 0.25, 1e-4, 0, 0.25, 1e-4, pml_cells=10, sigma_max=1)

# fig, ax = plt.subplots()

# im = ax.imshow(
#     sim.E.T,
#     origin="lower",
#     cmap=plt.colormaps["RdBu_r"],
#     vmin=-0.25,
#     vmax=0.25,
#     animated=True,
#     interpolation="bilinear"
# )

# pec_circle = Circle(
#     (sim.nx/2, sim.ny/2),
#     sim.a/sim.dx,
#     fill=False,
#     edgecolor='black',
#     linewidth=1
# )
# ax.add_patch(pec_circle)

# pml_rect = Rectangle(
#     (sim.pml_cells, sim.pml_cells),
#     sim.nx - 2*sim.pml_cells,
#     sim.ny - 2*sim.pml_cells,
#     fill=False,
#     edgecolor='black',
#     linestyle='--',
#     linewidth=1.5
# )

# ax.add_patch(pml_rect)


# plt.colorbar(im, ax=ax)

# def animate(frame):
#     for _ in range(1):
#         sim.update()

#         probe_x = int(sim.nx / 2) + 5
#         probe_y = int(sim.ny / 2)

#         sim.probe_history.append(sim.E[probe_x, probe_y])

#     im.set_array(sim.E.T)
#     return im, pec_circle, pml_rect

# ani = FuncAnimation(
#     fig,
#     animate,
#     frames=500,
#     interval=10,
#     blit=True
# )

# plt.show()

# # plt.imshow(sim.sigma_x.T, origin='lower')
# # plt.colorbar(label='sigma_x')
# # plt.title('PML sigma_x')
# # plt.show()

# # plt.imshow(sim.sigma_y.T, origin='lower')
# # plt.colorbar(label='sigma_y')
# # plt.title('PML sigma_y')
# # plt.show()

# # sim = FDTD_2D(0, 50, 1e-1, 0, 50, 1e-1)
# # while True:
# #     for i in range(20):
# #         sim.update()

# #     print(sim.source_profile((sim.t + 0.5) * sim.dt, 10, 1e7))
# #     print(np.max(np.abs(sim.E)))

# #     plt.imshow(np.transpose(sim.E), cmap=plt.colormaps["PuOr"], vmin=-0.25, vmax=0.25)
# #     plt.colorbar()
# #     plt.show()

# # sim_pml = FDTD_2D(
# #     0, 5, 0.1,
# #     0, 5, 0.1,
# #     pml_cells=10,
# #     sigma_max=0.07
# # )

# # sim_no_pml = FDTD_2D(
# #     0, 5, 0.1,
# #     0, 5, 0.1,
# #     pml_cells=10,
# #     sigma_max=0.0
# # )

# # probe_x = int(sim_pml.nx / 2) + 5
# # probe_y = int(sim_pml.ny / 2)

# # for n in range(500):
# #     sim_pml.update()
# #     sim_no_pml.update()

# #     sim_pml.probe_history.append(
# #         sim_pml.E[probe_x, probe_y]
# #     )

# #     sim_no_pml.probe_history.append(
# #         sim_no_pml.E[probe_x, probe_y]
# #     )

# # plt.figure()

# # plt.plot(
# #     sim_pml.probe_history,
# #     label="PML"
# # )

# # plt.plot(
# #     sim_no_pml.probe_history,
# #     label="No PML"
# # )

# # plt.xlabel("Time step")
# # plt.ylabel("Ez at probe")
# # plt.title("PML vs No PML")
# # plt.legend()
# # plt.grid()

# # plt.show()