import numpy as np
import matplotlib.pyplot as plt

def TE_reflection(theta, e1, e2):
    return ( np.sqrt(e1) * np.cos(theta) - np.sqrt(e2) * np.sqrt(1 - (e1/e2) * np.pow(np.sin(theta), 2))) / ( np.sqrt(e1) * np.cos(theta) + np.sqrt(e2) * np.sqrt(1 - (e1/e2) * np.pow(np.sin(theta), 2)))

def TE_transmission(theta, e1, e2):
    return 1 + TE_reflection(theta, e1, e2)

def TM_reflection(theta, e1, e2):
    return ( np.sqrt(e2) * np.cos(theta) - np.sqrt(e1) * np.sqrt(1 - (e1/e2) * np.pow(np.sin(theta), 2))) / ( np.sqrt(e2) * np.cos(theta) + np.sqrt(e1) * np.sqrt(1 - (e1/e2) * np.pow(np.sin(theta), 2)))

def TM_transmission(theta, e1, e2):
    return ( 2 * np.sqrt(e1) * np.cos(theta) ) / ( np.sqrt(e2) * np.cos(theta) + np.sqrt(e1) * np.sqrt(1 - (e1/e2) * np.pow(np.sin(theta), 2)))


def snells(theta, e1, e2):
    return np.arcsin(np.sin(theta) * np.sqrt(e1/e2))

angle = np.pi * 50/180 +0j
# print(TE_reflection(angle, 8, 4))

e1 = 8
e2 = 4
theta = np.linspace(0, np.pi / 2, 200) + 0j
deg = theta * 180 / np.pi





plt.plot(deg, np.pow(np.abs((TE_reflection(theta, e1,e2))), 2), color="red", label="TE power reflection")
plt.plot(deg, 1 - np.pow(np.abs((TE_reflection(theta, e1,e2))), 2), '--', color="red", label="TE power transmission")


plt.plot(deg, np.pow(np.abs((TM_reflection(theta, e1,e2))), 2), color="blue", label="TM power reflection")
plt.plot(deg, 1 - np.pow(np.abs((TM_reflection(theta, e1,e2))), 2), '--', color="blue", label="TM power transmission")
plt.legend()
plt.xlabel("angle (degrees)")
plt.ylabel("Power coefficient")
plt.title("Fresnel power coefficients")


plt.show()




pts = 200
wavelength = 1
k = np.pi * 2/wavelength

# plt.rcParams['text.usetex'] = True

angle2 = snells(angle, e1, e2)
print(angle2 * 180 / np.pi)

data = np.zeros([pts, pts]) + 0j
print(data)

for i in range(pts):
    x = wavelength * (2 * i / pts - 1)
    for j in range(pts):
        y = wavelength * (2 * j / pts - 1)

        if(x < 0):
            data[i, j] = np.exp(-1j * np.sqrt(e1) * k * (x * np.cos(angle) + y * np.sin(angle)))
            data[i, j] += TE_reflection(angle, e1, e2) * np.exp(-1j * np.sqrt(e1) * k * (-x * np.cos(angle) + y * np.sin(angle)))


        else:
            data[i,j] = TE_transmission(angle, e1, e2) * np.exp(-1j * np.sqrt(e2) * k * (x * np.cos(angle2) + y * np.sin(angle2)))


plt.imshow(np.transpose(np.real(data)), extent=[-1,1,-1,1])
plt.title("Real component of E field")
plt.xlabel("x (Wavelengths)")
plt.ylabel("z (Wavelengths)")
bar = plt.colorbar()
bar.set_label("Re[Ey/E0] magnitude")

plt.show()

plt.imshow(np.transpose(np.log(np.abs(data))) , extent=[-1,1,-1,1])

plt.title("Complex magnitude of E field")
plt.xlabel("x (Wavelengths)")
plt.ylabel("z (Wavelengths)")
bar = plt.colorbar()
bar.set_label("Ey/E0 magnitude (logarithmic)")


plt.show()
