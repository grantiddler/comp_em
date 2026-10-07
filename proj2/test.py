# from fdtd_2d import FDTD_2D
from fdtd_2d_PML import FDTD_2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle, Rectangle
from scipy.integrate import simpson

from scipy.fft import rfft



sim = FDTD_2D('PEC', 0, .125, 5e-4, 0, .125, 5e-4,1, pml_cells=20, sigma_max=1)
# sim = FDTD_2D(0, .125, 1e-4, 0, .125, 1e-4,1)
# sim = FDTD_2D(0, .25, 5e-4, 0, .25, 5e-4,1)
# sim = FDTD_2D('dielectric', 0, .125, 1e-4, 0, .125, 1e-4,1, pml_cells=50, sigma_max=1)

# sim = FDTD_2D(0, .25, 5e-4, 0, .25, 5e-4,1)
# sim = FDTD_2D(0, 50, 1e-1, 0, 50, 1e-1,1)
Es = []
Hy = []
Hx = []

frames = 500
steps_per_frame = 2

#simulation time samples
Nt = frames * steps_per_frame
t = np.arange(Nt) * sim.dt


#source waveform
source_time = sim.source_profile(t) 

source_ft = rfft(source_time)

ts = []


x0 = 25
x1 = sim.nx - 25
y0 = 25
y1 = sim.ny - 25


xs = np.linspace(x0, x1, x1-x0) * sim.dx
ys = np.linspace(y0, y1, y1-y0) * sim.dy



M_x_l = np.zeros([y1 - y0, frames * steps_per_frame])
M_x_u = np.zeros([x1 - x0, frames * steps_per_frame])
M_x_r = np.zeros([y1 - y0, frames * steps_per_frame])
M_x_d = np.zeros([x1 - x0, frames * steps_per_frame])
J_x_l = np.zeros([y1 - y0, frames * steps_per_frame])
J_x_u = np.zeros([x1 - x0, frames * steps_per_frame])
J_x_r = np.zeros([y1 - y0, frames * steps_per_frame])
J_x_d = np.zeros([x1 - x0, frames * steps_per_frame])


for j in range(frames):
    for i in range(steps_per_frame):
        sim.update()

        M_x_l[:, steps_per_frame * j + i] = (-sim.E[x0,y0:y1]) # y directed
        M_x_u[:, steps_per_frame * j + i] = (-sim.E[x0:x1,y1]) # x directed
        M_x_r[:, steps_per_frame * j + i] = (sim.E[x1,y0:y1]) # y directed
        M_x_d[:, steps_per_frame * j + i] = (sim.E[x0:x1,y0]) # x directed

        J_x_l[:, steps_per_frame * j + i] = (-sim.H_y[x0,y0:y1]) # z directed
        J_x_u[:, steps_per_frame * j + i] = (-sim.H_x[x0:x1,y1]) # z directed
        J_x_r[:, steps_per_frame * j + i] = (sim.H_y[x1,y0:y1]) # z directed
        J_x_d[:, steps_per_frame * j + i] = (sim.H_x[x0:x1,y0]) # z directed

    # print(sim.source_profile((sim.t + 0.5) * sim.dt, 10, 1e7))
    # print(np.max(np.abs(sim.E)))



    Es.append(np.transpose(sim.E))
    Hx.append(sim.H_x)
    Hy.append(sim.H_y)
    
    print(sim.t / (frames * steps_per_frame))
    
    # plt.imshow(np.transpose(sim.E), cmap=plt.colormaps["PuOr"], vmin=-0.25, vmax=0.25)
    # plt.colorbar()
    # plt.show()


M_x_l_ft = rfft(M_x_l)
M_x_u_ft = rfft(M_x_u)
M_x_r_ft = rfft(M_x_r)
M_x_d_ft = rfft(M_x_d)

J_x_l_ft = rfft(J_x_l)
J_x_u_ft = rfft(J_x_u)
J_x_r_ft = rfft(J_x_r)
J_x_d_ft = rfft(J_x_d)

# Find the FFT bin index corresponding to the source frequency f0 = 10 GHz
f0 = int((2 * sim.f0 * sim.dt * np.shape(M_x_l_ft)[1]))


# Wavenumber used for spatial phase term in the far-field calculation
k = sim.k0



theta = np.atleast_2d(np.linspace(0,np.pi * 2, 100))
rx = np.cos(theta)
ry = np.sin(theta)


exponential_l = np.exp(1j * k * (rx.T * np.atleast_2d(x0 * sim.dx - 0.0625) + ry.T * np.atleast_2d(ys - 0.0625))) 
exponential_r = np.exp(1j * k * (rx.T * np.atleast_2d(x1 * sim.dx - 0.0625) + ry.T * np.atleast_2d(ys - 0.0625))) 
exponential_d = np.exp(1j * k * (rx.T * np.atleast_2d(xs - 0.0625) + ry.T * np.atleast_2d(y0 * sim.dy - 0.0625))) 
exponential_u = np.exp(1j * k * (rx.T * np.atleast_2d(xs - 0.0625) + ry.T * np.atleast_2d(y1 * sim.dy - 0.0625))) 


# N is Z directed
print(np.shape(J_x_l_ft))
print(np.shape(exponential_l))
N  = (exponential_l @ J_x_l_ft) * sim.dy
N += (exponential_r @ J_x_r_ft) * sim.dy
N += (exponential_u @ J_x_u_ft) * sim.dx
N += (exponential_d @ J_x_d_ft) * sim.dx



L_y  = (exponential_l @ M_x_l_ft) * sim.dy
L_y += (exponential_r @ M_x_r_ft) * sim.dy
L_x  = (exponential_u @ M_x_u_ft) * sim.dx
L_x += (exponential_d @ M_x_d_ft) * sim.dx



theta =  (np.ones([(np.shape(M_x_l_ft)[1]),1]) @ theta ).T

phi_x = np.sin(theta)
phi_y = -np.cos(theta)

L = L_x * phi_x + L_y * phi_y



print(sim.k0)
mag_E_f0 = L[:, f0] + sim.Z0 * N[:, f0]
#echo_width = np.abs(mag_E) ** 2

echo_width = (sim.k0 / (np.pi*4)) * np.abs(mag_E_f0 / source_ft[f0])**2

phi_fdtd = theta[:, 0]

#matches sample size
sigma_2D_interp = np.interp(phi_fdtd, sim.phi, sim.sigma_2D)

#numerator
num = np.sqrt(simpson((echo_width - sigma_2D_interp)**2, x=theta[:,0]) / (theta[-1,0] - theta[0,0]))   #can increase angular samples for better acuracy

# denominator
den = np.sqrt(simpson(sigma_2D_interp**2, x=phi_fdtd))

rmse = num/den

print("Normalized RMSE =", rmse)
print("Normalized RMSE (%) =", rmse * 100)

f = np.linspace(0, np.shape(M_x_l_ft)[1], np.shape(M_x_l_ft)[1]) * 1 / (np.shape(M_x_l_ft)[1] * 2 * sim.dt)

f0 = round((2 * sim.f0 * sim.dt * np.shape(M_x_l_ft)[1]))

plt.polar(sim.phi, sim.sigma_2D)
plt.title("Analytical Bistatic Echo Width")
plt.show()

plt.polar(theta[:,0],echo_width)
plt.title("Numerical Bistatic Echo Width")
plt.show()

plt.polar(sim.phi, sim.sigma_2D, label="Analytical Solution")
plt.polar(theta[:,0],echo_width, label="Numerical Solution")
plt.legend()
plt.title("Analytical and FDTD Bistatic Echo Width")
plt.show()
# plt.title("phase")
# plt.plot(np.angle(1j * mag_E[:,f0]))
# plt.show()

analytical_norm = sim.sigma_2D / np.max(sim.sigma_2D)

numerical_sigma = echo_width
numerical_norm = numerical_sigma / np.max(numerical_sigma)

# plt.polar(sim.phi, sim.sigma_2D, label="Analytical")
# plt.polar(theta[:,0], echo_width, label="FDTD")
# plt.title("Analytical and FDTD Bistatic Echo Width")
# plt.legend()
# plt.show()
# print("source FFT at f0 =", abs(source_ft[f0]))
# print("Analytical max =", np.max(sim.sigma_2D))
# print("FDTD max =", np.max(echo_width))

# print("Source FFT at 10 GHz =", np.abs(source_ft[f0]))
# print("max |L| =", np.max(np.abs(L[:, f0])))
# print("max |Z0*N| =", np.max(np.abs(sim.Z0 * N[:, f0])))
# print("max |L+Z0*N| =", np.max(np.abs(L[:, f0] + sim.Z0 * N[:, f0])))

# f=k/n


theta = np.linspace(0, np.pi * 2, 100)
x = 0.0625 + np.sin(theta) * .015
y = 0.0625 + np.cos(theta) * .015
print("Starting animation")
print("Total simulated time:", frames * steps_per_frame * sim.dt)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title(r"TM$_z$ Scattering from a PEC Cylinder")
ax.plot(x,y, linestyle=(0, (2,3)), linewidth=1)

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



ax.plot([sim.pml_cells* sim.dx, (sim.nx-sim.pml_cells) * sim.dx, (sim.nx-sim.pml_cells) * sim.dx, sim.pml_cells * sim.dx, sim.pml_cells * sim.dx], [sim.pml_cells * sim.dy, sim.pml_cells * sim.dy, (sim.ny-sim.pml_cells) * sim.dy, (sim.ny-sim.pml_cells) * sim.dy, sim.pml_cells * sim.dy], linestyle=(0, (2,3)), linewidth=1)
ax.plot([sim.huygens_x0 * sim.dx, sim.huygens_x1 * sim.dx, sim.huygens_x1 * sim.dx, sim.huygens_x0 * sim.dx, sim.huygens_x0 * sim.dx], [sim.huygens_y0 * sim.dy, sim.huygens_y0 * sim.dy, sim.huygens_y1 * sim.dy, sim.huygens_y1 * sim.dy, sim.huygens_y0 * sim.dy], linestyle=(0, (2,3)), linewidth=1)
ax.plot([x0 * sim.dx, x1 * sim.dx, x1 * sim.dx, x0 * sim.dx, x0 * sim.dx], [y0 * sim.dy, y0 * sim.dy, y1 * sim.dy, y1 * sim.dy, y0 * sim.dy], linestyle=(0, (2,3)), linewidth=1)


div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

cv0 = Es[0]
cf = ax.imshow(cv0, cmap=plt.colormaps["seismic"], vmin=-0.25, vmax=0.25, origin="lower", extent=(sim.x0,sim.x1,sim.y0,sim.y1))
cb = fig.colorbar(cf, cax=cax)
# tx = ax.set_title('Frame 0')

def animate(i, Es):
    #print(i)
    arr = Es[i]
    cf.set_data(arr)
    cax.cla()
    fig.colorbar(cf, cax=cax)
    # tx.set_text('Frame {0}'.format(i))

ani = animation.FuncAnimation(fig, animate, frames=frames,interval=10, fargs=[Es], repeat=False)
plt.show()


# ani.save(filename="2.mp4", writer="ffmpeg", fps=30)


