# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
from IPython.display import display, HTML


class FDTD_2D:
    def __init__(
        self,
        x0,
        x1,
        dx,
        y0,
        y1,
        dy,
        time_oversample=1.5,
        forcing_type=[],
        forcing_function=[]
    ):
        self.permittivity = 8.8541878188e-12
        self.permeability = 1.25663706127e-6
        self.conductivity = 0

        self.c = 1 / np.sqrt(
            self.permittivity * self.permeability
        )

        self.x0 = x0
        self.x1 = x1
        self.dx = dx

        self.y0 = y0
        self.y1 = y1
        self.dy = dy

        self.dt = 1 / (
            time_oversample
            * self.c
            * np.sqrt(
                (1 / np.pow(dx, 2))
                + (1 / np.pow(dy, 2))
            )
        )

        self.t = 0

        self.nx = int((x1 - x0) / dx)
        self.ny = int((y1 - y0) / dy)

        # Electric field grid
        self.E = np.zeros([self.nx, self.ny])

        # Magnetic fields are offset from E on the Yee grid
        self.H_x_offset = np.zeros([self.nx + 1, self.ny])
        self.H_y_offset = np.zeros([self.nx, self.ny + 1])

        self.E_last = self.E.copy()
        self.H_x_offset_last = self.H_x_offset.copy()
        self.H_y_offset_last = self.H_y_offset.copy()

        self.f_type = forcing_type
        self.f_func = forcing_function

        self.sz_E = np.shape(self.E)
        self.sz_H_x = np.shape(self.H_x_offset)
        self.sz_H_y = np.shape(self.H_y_offset)

        # Arrays used for finite differences
        self.sz_E_x_border = self.sz_E + np.array([1, 0])
        self.sz_E_y_border = self.sz_E + np.array([0, 1])

        self.sz_H_x_border = self.sz_H_x + np.array([1, 0])
        self.sz_H_y_border = self.sz_H_y + np.array([0, 1])

        self.diff_coeff_y = (
            self.dt / (self.permeability * self.dy)
        )

        self.diff_coeff_x = (
            self.dt / (self.permeability * self.dx)
        )

        self.alpha = 1
        self.beta = 1


    def update(self):

        # Save previous fields
        self.E_last = self.E.copy()
        self.H_x_offset_last = self.H_x_offset.copy()
        self.H_y_offset_last = self.H_y_offset.copy()

        # Extra-sized E arrays for finite differences
        E_su = np.zeros(self.sz_E_y_border)
        E_sd = np.zeros(self.sz_E_y_border)

        E_sl = np.zeros(self.sz_E_x_border)
        E_sr = np.zeros(self.sz_E_x_border)

        # Shift E in the y direction
        E_sd[:, 1:] = self.E
        E_su[:, 0:-1] = self.E

        # Shift E in the x direction
        E_sr[1:, :] = self.E
        E_sl[0:-1, :] = self.E

        # Update magnetic fields
        self.H_x_offset = (
            self.H_x_offset
            - self.diff_coeff_x * (E_sl - E_sr)
        )

        self.H_y_offset = (
            self.H_y_offset
            - self.diff_coeff_y * (E_su - E_sd)
        )

        # -----------------------------------
        # DO H BOUNDARY CONDITIONS HERE
        # -----------------------------------

        # Update electric field
        self.E = (
            self.alpha * self.E
            + (
                self.H_x_offset[0:-1, :]
                - self.H_x_offset[1:, :]
            ) / self.dx
            + (
                self.H_y_offset[:, 0:-1]
                - self.H_y_offset[:, 1:]
            ) / self.dy
        ) / self.beta

        # -----------------------------------
        # DO E BOUNDARY AND FORCING HERE
        # -----------------------------------

        self.t += self.dt


# %%
sim = FDTD_2D(
    0,
    1,
    1e-3,
    0,
    1,
    1e-3
)

Es = []


# %%
# Temporary animation test
# This is not yet plotting the FDTD fields.

fig, ax = plt.subplots()

line1, = ax.plot([], [], 'ro')


def init():
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)

    return line1,


def update_animation(frame, ln, x, y):

    x.append(frame)
    y.append(np.sin(frame))

    ln.set_data(x, y)

    return ln,


ani = FuncAnimation(
    fig,
    partial(
        update_animation,
        ln=line1,
        x=[],
        y=[]
    ),
    frames=np.linspace(
        0,
        2 * np.pi,
        128
    ),
    init_func=init,
    blit=False
)

plt.show()