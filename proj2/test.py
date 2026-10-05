from fdtd_2d_PML import FDTD_2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle, Rectangle



# sim = FDTD_2D('PEC', 0, .25, 2e-4, 0, .25, 2e-4,1, pml_cells=25, sigma_max=1)
sim = FDTD_2D('dielectric', 0, .25, 5e-4, 0, .25, 5e-4,1, pml_cells=50, sigma_max=1)

# sim = FDTD_2D(0, .25, 5e-4, 0, .25, 5e-4,1)
# sim = FDTD_2D(0, 50, 1e-1, 0, 50, 1e-1,1)
Es = []
Hy = []
Hx = []

frames = 200
steps_per_frame = 5
for j in range(frames):
    for i in range(steps_per_frame):
        sim.update()

    # print(sim.source_profile((sim.t + 0.5) * sim.dt, 10, 1e7))
    # print(np.max(np.abs(sim.E)))

    Es.append(np.transpose(sim.E))
    Hx.append(sim.H_x)
    Hy.append(sim.H_y)
    
    print(sim.t / (frames * steps_per_frame))
    
    # plt.imshow(np.transpose(sim.E), cmap=plt.colormaps["PuOr"], vmin=-0.25, vmax=0.25)
    # plt.colorbar()
    # plt.show()



theta = np.linspace(0, np.pi * 2, 100)
x = 0.125 + np.sin(theta) * .015
y = 0.125 + np.cos(theta) * .015

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y, linestyle=(0, (2,3)), linewidth=1)

pml_rect = Rectangle(
    (sim.pml_cells, sim.pml_cells),
    sim.nx - 2*sim.pml_cells,
    sim.ny - 2*sim.pml_cells,
    fill=False,
    edgecolor='black',
    linestyle='--',
    linewidth=1.5
)

ax.add_patch(pml_rect)



ax.plot([sim.pml_cells* sim.dx, (sim.nx-sim.pml_cells) * sim.dx, (sim.nx-sim.pml_cells) * sim.dx, sim.pml_cells * sim.dx, sim.pml_cells * sim.dx], [sim.pml_cells * sim.dy, sim.pml_cells * sim.dy, (sim.ny-sim.pml_cells) * sim.dy, (sim.ny-sim.pml_cells) * sim.dy, sim.pml_cells * sim.dy], linestyle=(0, (2,3)), linewidth=1)
ax.plot([sim.huygens_x0 * sim.dx, sim.huygens_x1 * sim.dx, sim.huygens_x1 * sim.dx, sim.huygens_x0 * sim.dx, sim.huygens_x0 * sim.dx], [sim.huygens_y0 * sim.dy, sim.huygens_y0 * sim.dy, sim.huygens_y1 * sim.dy, sim.huygens_y1 * sim.dy, sim.huygens_y0 * sim.dy], linestyle=(0, (2,3)), linewidth=1)


div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

cv0 = Es[0]
cf = ax.imshow(cv0, cmap=plt.colormaps["seismic"], vmin=-0.25, vmax=0.25, origin="lower", extent=(sim.x0,sim.x1,sim.y0,sim.y1))
cb = fig.colorbar(cf, cax=cax)
# tx = ax.set_title('Frame 0')

def animate(i, Es):
    print(i)
    arr = Es[i]
    cf.set_data(arr)
    cax.cla()
    fig.colorbar(cf, cax=cax)
    # tx.set_text('Frame {0}'.format(i))

ani = animation.FuncAnimation(fig, animate, frames=frames,interval=10, fargs=[Es])
plt.show()


# ani.save(filename="2.mp4", writer="ffmpeg", fps=30)


