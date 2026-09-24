from fdtd_2d import FDTD_2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable



sim = FDTD_2D(0, 50, 1e-1, 0, 50, 1e-1)
Es = []
Hy = []
Hx = []

frames = 500
steps_per_frame = 20
for j in range(frames):
    for i in range(steps_per_frame):
        sim.update()

    # print(sim.source_profile((sim.t + 0.5) * sim.dt, 10, 1e7))
    # print(np.max(np.abs(sim.E)))

    Es.append(sim.E)
    Hx.append(sim.H_x)
    Hy.append(sim.H_y)
    
    print(sim.t / (frames * steps_per_frame))
    
    # plt.imshow(np.transpose(sim.E), cmap=plt.colormaps["PuOr"], vmin=-0.25, vmax=0.25)
    # plt.colorbar()
    # plt.show()



fig = plt.figure()
ax = fig.add_subplot(111)

# I like to position my colorbars this way, but you don't have to
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

cv0 = Es[0]
cf = ax.imshow(cv0, cmap=plt.colormaps["seismic"], vmin=-0.25, vmax=0.25)
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

ani.save(filename="E.mp4", writer="ffmpeg", fps=60)
