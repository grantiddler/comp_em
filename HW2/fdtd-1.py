import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



class FDTD_1D:
    def __init__(self, x0, x1, dx, time_oversample = 1.5):
        self.x0 = x0
        self.x1 = x1
        self.dx = dx
        self.n = int((x1 - x0) / dx)

        self.x = np.arange(x0, x1, dx)

        self.time_oversample = time_oversample

        self.permittivity = 8.8541878188e-12
        self.permeability = 1.25663706127e-6

        self.c = 1 / np.sqrt(self.permittivity * self.permeability)

        self.dt = self.dx / ( self.c * self.time_oversample)

        self.E = np.zeros(int((x1 - x0) / dx))
        self.E_last = self.E

        self.r = np.pow(self.c * self.dt / self.dx, 2)
        print(self.r)

        self.t = 0

    def source_profile(self, t, omega, scale):
        return (1 - np.exp(- scale * t)) * np.cos(t * omega * scale) * np.exp(- np.pow( t * scale - 3, 2))

    def forcing(self, offset):
        z = np.zeros(self.n + 2)
        I = self.source_profile(self.t + offset, 10, 1e7)
        # print(f"I = {I}")
        z[int(self.n / 2)] = I
        # print(self.source_profile(self.t + offset, 10, 1e5))
        return z

    def update(self):
        z = np.zeros(1)
        E2 = np.append(z, self.E)
        E = np.append(E2, z)
        E_last = np.append(np.append(z, self.E_last), z)


        E_sl = np.append(np.append(self.E, z), z)
        E_sr = np.append(z, E2)

        # print(E_sl)
        # print(E_sr)
        # print(E)

        E_next = 2 * E - E_last + self.r * (E_sl + E_sr - 2 * E) - (self.dt / (2 * self.permittivity)) * (self.forcing(self.dt) - self.forcing(-self.dt))



        # plt.title("cross_coeff")
        # plt.plot(E)
        # plt.plot(E_last)
        # plt.plot(self.r * (E_sl + E_sr - 2 * E))
        # plt.plot(2 * E - E_last + self.r * (E_sl + E_sr - 2 * E))
        # plt.show()
        
        self.E_last = self.E
        self.E = E_next[1:-1]

        self.t += self.dt

    def plot(self):

        # plt.plot(self.forcing(0))
        plt.plot(self.x, self.E)
        plt.ylim(-3,3)
        plt.show()

    def animate(self, frame, graph, I):
        for i in range(int(1e2)):
            self.update()
        # updating the data
        x = self.x
        y = self.E

        # creating a new graph or updating the graph
        graph.set_xdata(x)
        graph.set_ydata(y)

        I.set_ydata([0, self.source_profile(self.t, 10, 1e7)])

        print(np.max(np.abs(y)))
        print(self.t / (self.dt * 10000))

        # plt.ylim(-3,3)
        plt.xlim(0, x[-1])
        plt.ylim(-4, 4)


sim = FDTD_1D(0, 500, 1e-2, 4)

fig, ax = plt.subplots()
graph = ax.plot([],[])[0]
I = ax.plot([250, 250],[0, 0])[0]


anim = FuncAnimation(fig, sim.animate, frames = None, save_count=50000,fargs=[graph, I])
plt.show()

anim.save(filename="ffmpeg_example.mp4", writer="ffmpeg", fps=60)
