import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.special import hankel1, hankel2

### EDGE GENERATION

radius = 0.015

fineness = 0.001 # maximum edge length

num_points = int(max(3, np.ceil(2 * np.pi * radius / fineness) ))

print(num_points)
angles = np.linspace(0, np.pi * 2,num_points + 1)[0:-1]
points = radius * np.array([np.cos(angles), np.sin(angles)])

edges = np.array([np.arange(0, num_points), np.arange(1, num_points+1)])
edges[1,-1] = 0


midpoints = (points[:, edges[0,:]] + points[:, edges[1,:]]) / 2


# for i in range(num_points):
#     plt.plot([points[0, edges[0, i]], points[0, edges[1, i]]], [points[1, edges[0, i]], points[1, edges[1, i]]], color="black")
#     plt.plot(midpoints[0,i], midpoints[1,i],'o')

# plt.show()



#### MATRIX BUILDING

# basic electromagnetic constants
c0 = 299792458
f = 10e9
k0 = 2 * np.pi * f / c0
Z0 = 377
gamma = np.exp(1) - 1 # TODO: Where does this come from

scatterer = "dielectric"

G0 = lambda r: hankel2(0, k0 * np.abs(r)) / 4j


Z = np.zeros([num_points, num_points], np.dtype(np.complex128))
Sm = np.zeros(num_points)


if scatterer == "PEC":
    Gi = lambda r: 0
    
    for m in range(num_points): 
        pt_a1 = edges[0,m]
        pt_a2 = edges[1,m]

        len_edge = norm(points[:,pt_a1] - points[:,pt_a2])
        Sm[m] = len_edge
        for n in range(m+1):




            if m == n:
                Z[m,n] =  k0 * Z0 * len_edge * len_edge * ( 1 - 2j * np.log(k0 * gamma * len_edge / (4 * np.exp(1))) / np.pi)  # exp can maybe be removed, turned to -1, don't know if that will make problem conditioning worse.
            else:
                pt_b1 = edges[0,n]
                pt_b2 = edges[1,n]

                len_edge2 = norm(points[:,pt_b1] - points[:,pt_b2])

                Z[m,n] = k0 * Z0 * len_edge * len_edge2 * hankel2(0, k0 * norm( midpoints[:,m] - midpoints[:,n])) / 4
                Z[n,m] = Z[m, n]
else :
    er = 9
    mr = 1

    Gi = lambda r:  hankel2(0, k0 * np.abs(r)) / 4j
    # Gi = lambda r: 0


    A0 = np.zeros([num_points, num_points], np.dtype(np.complex128))
    Ai = np.zeros([num_points, num_points], np.dtype(np.complex128))
    B0 = np.zeros([num_points, num_points], np.dtype(np.complex128))
    Bi = np.zeros([num_points, num_points], np.dtype(np.complex128))
    
    for m in range(num_points): 
        pt_a1 = edges[0,m]
        pt_a2 = edges[1,m]

        normal = points[:, pt_a1] - points[:, pt_a2]
        normal = np.array([-normal[1], normal[0]])

        normal = -normal / norm(normal)

        len_edge = norm(points[:,pt_a1] - points[:,pt_a2])
        Sm[m] = len_edge
        for n in range(m+1):

            if m == n:
                A0[m,m] = (len_edge / 2) - len_edge * len_edge * ( 1 - 2j * np.log(k0 * gamma * len_edge / (4 * np.exp(1))) / np.pi) / 4j
                Ai[m,m] = (len_edge / 2) + len_edge * len_edge * ( 1 - 2j * np.log(k0 * np.sqrt(er * mr) * gamma * len_edge / (4 * np.exp(1))) / np.pi) / 4j

                # Z[m,n] =  k0 * Z0 * len_edge * len_edge * ( 1 - 2j * np.log(k0 * gamma * len_edge / (4 * np.exp(1))) / np.pi)  # exp can maybe be removed, turned to -1, don't know if that will make problem conditioning worse.
            else:
                pt_b1 = edges[0,n]
                pt_b2 = edges[1,n]

                len_edge2 = norm(points[:,pt_b1] - points[:,pt_b2])

                A0[m,n] = len_edge * len_edge2 * hankel2(0, k0 * norm( midpoints[:,m] - midpoints[:,n])) / 4j
                Ai[m,n] = len_edge * len_edge2 * hankel2(0, k0 * np.sqrt(er * mr) * norm( midpoints[:,m] - midpoints[:,n])) / 4j

                A0[n, m] = A0[m, n]
                Ai[n, m] = Ai[m, n] 



                NDOT = np.inner(normal, midpoints[:, m] - midpoints[:, n]) / norm(midpoints[:, m] - midpoints[:, n])
                NDOT = np.inner(normal, midpoints[:, m] - midpoints[:, n]) / norm(midpoints[:, m] - midpoints[:, n])


                B0[m,n] = Z0 * k0 * k0 * len_edge * len_edge2 * NDOT *  hankel2(1, k0 * norm( midpoints[:,m] - midpoints[:,n])) / 4
                Bi[m,n] = mr * Z0 * k0 * k0 * np.sqrt(er * mr) * len_edge * len_edge2 * NDOT *  hankel2(1, k0 * np.sqrt(er * mr) * norm( midpoints[:,m] - midpoints[:,n])) / 4
                
                B0[n, m] = B0[m, n]
                Bi[n, m] = Bi[m, n]
    
    Z = 1j * (A0 + B0 @ np.linalg.inv(Bi) @ Ai)


### SOLVE SYSTEM
E_inc = np.exp(1j * k0 * midpoints[0,:])


J = np.linalg.inv(Z) @ (Sm * E_inc)


## DISPLAY THINGS:

plt.polar(angles, np.abs(J), 'o')
# plt.polar(angles, np.angle(J))
plt.show()

sz = 300
a = -0.1
b = 0.1
vals = np.linspace(a,b,sz)
o = np.ones(sz)
x = np.outer(vals,o)
y = np.outer(o,vals)

s = np.zeros([sz,sz], np.dtype(np.complex128))
kr = 1
cylinder_mask = (np.pow(x, 2) + np.pow(y, 2) <= 0.015 ** 2)
for i in range(num_points):
    x_dif = -x - midpoints[0, i]
    y_dif = y - midpoints[1, i]
    r = np.sqrt(np.pow(x_dif, 2)+ np.pow(y_dif, 2))
    

    # plt.imshow(np.abs(r))
    # plt.show()


    s += Sm[i] * ((1j * k0 * Z0 * J[i] * G0(r)) * (1 - cylinder_mask) + (1j * k0 * Z0 * J[i] * Gi(r)) * (cylinder_mask))


# s[cylinder_mask] = 0

plt.imshow(np.abs(s).T, cmap="viridis", extent=[a,b,a,b])
plt.colorbar()
plt.show()

plt.imshow(np.angle(s).T, cmap="twilight", extent=[a,b,a,b])
plt.colorbar()
plt.show()


