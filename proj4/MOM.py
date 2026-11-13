import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.special import hankel1, hankel2

from scipy.special import jv, jvp, h2vp
### EDGE GENERATION
scatterer = "dielectric"

radius = 0.015

fineness = 0.001 # maximum edge length

num_points = int(max(3, np.ceil(2 * np.pi * radius / fineness) ))

print(f"Number of cylinder boundary elements: {num_points}")
angles = np.linspace(0, np.pi * 2,num_points + 1)[0:-1]
points = radius  * np.array([np.cos(angles), np.sin(angles)])

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
gamma = np.exp(np.euler_gamma) # TODO: Where does this come from —— lecture notes use e^euler_gamma which is around the same value as e-1


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

    Gi = lambda r:  hankel2(0, k0 * np.sqrt(er * mr) * np.abs(r)) / 4j
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
        for n in range(m + 1):

            if m == n:
                A0[m,m] = len_edge / 2
                Ai[m,m] = len_edge / 2
                
                B0[m,m] = len_edge * len_edge * k0 * Z0 * ( 1 - 2j * np.log(k0 * gamma * len_edge / (4 * np.exp(1))) / np.pi) / 4
                Bi[m,m] = len_edge * len_edge * k0 * Z0 * ( 1 - 2j * np.log(k0 * np.sqrt(er * mr) * gamma * len_edge / (4 * np.exp(1))) / np.pi) / 4

                # Z[m,n] =  k0 * Z0 * len_edge * len_edge * ( 1 - 2j * np.log(k0 * gamma * len_edge / (4 * np.exp(1))) / np.pi)  # exp can maybe be removed, turned to -1, don't know if that will make problem conditioning worse.
            else:
                pt_b1 = edges[0,n]
                pt_b2 = edges[1,n]

                len_edge2 = norm(points[:,pt_b1] - points[:,pt_b2])

                NDOT = np.inner(normal, midpoints[:, m] - midpoints[:, n]) / norm(midpoints[:, m] - midpoints[:, n])

                A0[m,n] = - k0 * len_edge * len_edge2 * NDOT * hankel2(1, k0 * norm( midpoints[:,m] - midpoints[:,n])) / 4j
                Ai[m,n] = k0 * np.sqrt(er * mr) * len_edge * len_edge2 * NDOT * hankel2(1, k0 * np.sqrt(er * mr) * norm( midpoints[:,m] - midpoints[:,n])) / 4j

                A0[n, m] = A0[m, n]
                Ai[n, m] = Ai[m, n] 




                B0[m,n] = Z0 * k0 * len_edge * len_edge2 * hankel2(0, k0 * norm( midpoints[:,m] - midpoints[:,n])) / 4
                Bi[m,n] = mr * Z0 * k0 * len_edge * len_edge2 *  hankel2(0, k0 * np.sqrt(er * mr) * norm( midpoints[:,m] - midpoints[:,n])) / 4
                
                B0[n, m] = B0[m, n]
                Bi[n, m] = Bi[m, n]
    
    Z = (A0 + B0 @ np.linalg.inv(Bi) @ Ai)
    # Recover the eliminated boundary unknown after solving for J


### SOLVE SYSTEM
E_inc = np.exp(-1j * k0 * midpoints[0,:])


J = np.linalg.solve(Z, Sm * E_inc) #avoids constructing the inverse and more efficient and numerically stable
if scatterer == "dielectric":
    H_boundary = -np.linalg.solve(Bi, Ai @ J) #won't calculate the inverse of Bi
    E_boundary = J.copy() #change name to be more descriptive

### FAR_FIELD CALC
phi = np.linspace(0, 2*np.pi, 361) #evaluates the scattering pattern at 1° increments from 0° to 360°
F_MoM = np.zeros(len(phi), dtype=complex) #far-field scattering amplitude calculated from MoM currents

for i in range(num_points):
    phase = np.exp(1j * k0 * ( midpoints[0, i] * np.cos(phi) + midpoints[1, i] * np.sin(phi)))  #phase factor

    if scatterer == "PEC":
        F_MoM += J[i] * Sm[i] * phase

    else:
        normal = -midpoints[:, i] / norm(midpoints[:, i])

        dot_normal = (normal[0] * np.cos(phi) + normal[1] * np.sin(phi))

        F_MoM += Sm[i] * phase * (1j * k0 * Z0 * H_boundary[i] - 1j * k0 * dot_normal * E_boundary[i])

if scatterer == "PEC":
    F_MoM *= -(k0 * Z0 / 4) * np.sqrt(2 / (np.pi * k0)) * np.exp(1j * np.pi / 4)

else:
    F_MoM *= (1 / (4j)) * np.sqrt(2 / (np.pi * k0)) * np.exp(1j * np.pi / 4)

sigma_2D_num = 2 * np.pi * np.abs(F_MoM)**2

## DISPLAY THINGS:

# plt.polar(angles, np.abs(J))
#plt.polar(angles, np.abs(J))


###Near field testing###

# R_obs = 0.1  # Observation radius (m)
# x_obs = R_obs * np.cos(angles)
# y_obs = R_obs * np.sin(angles)
# E_scat_MoM = np.zeros(len(angles), dtype=complex)

# for i in range(num_points):
#     x_dif = x_obs - midpoints[0, i]
#     y_dif = y_obs - midpoints[1, i]

#     r = np.sqrt(x_dif**2 + y_dif**2)

#     E_scat_MoM += -1j * k0 * Z0 * J[i] * Sm[i] * G0(r)


######
Nh = 50
a = radius

E_scat_analytical = np.zeros(np.size(angles), dtype=complex)

F_analytical = np.zeros(len(phi), dtype=complex)

for n in range(-Nh, Nh + 1):

    if scatterer == "dielectric":
        kd = np.sqrt(er) * k0
        
        an = -((-1j)**n) * (
            jvp(n, k0*a)*jv(n, kd*a)- np.sqrt(er)*jv(n, k0*a)*jvp(n, kd*a)
        ) / (
            h2vp(n, k0*a)*jv(n, kd*a)- np.sqrt(er)*hankel2(n, k0*a)*jvp(n, kd*a))
    
    else:
        an = -((-1j)**n) * (jv(n, k0 * a) / hankel2(n, k0 * a))


    E_scat_analytical += (an * hankel2(n, k0 * a) * np.exp(1j * n * (angles)))
    F_analytical += an * np.exp(1j * n * (phi + np.pi/2))
    #E_scat_analytical += (an * hankel2(n, k0 * R_obs) * np.exp(1j * n * angles))

F_analytical *= np.sqrt(2 / (np.pi * k0)) * np.exp(1j * np.pi / 4)
sigma_2D_analytical = 2 * np.pi * np.abs(F_analytical)**2

### RMSE CALC
numerator = np.trapezoid((sigma_2D_num - sigma_2D_analytical)**2, phi)

denominator = np.trapezoid(sigma_2D_analytical**2, phi)

RMSE = np.sqrt(numerator / denominator)
###

print(f"Normalized Echo-Width RMSE: {RMSE:.6f}")
print(f"Echo-Width RMSE: {RMSE*100:.4f}%")


# plt.plot(angles, np.angle((-1j * J)))
# plt.plot(angles, np.angle((E_scat_analytical)))
# plt.show()


sz = 300
a = -0.1
b = 0.1
vals = np.linspace(a,b,sz)
o = np.ones(sz)
x = np.outer(vals,o)
y = np.outer(o,vals)

s = np.zeros([sz,sz], np.dtype(np.complex128))
kr = 1

E_inc_grid = np.exp(-1j * k0 * x)   #incident electric field only at the midpoints of cylinder's boundary segments

cylinder_mask = (np.pow(x, 2) + np.pow(y, 2) <= 0.015 ** 2)
for i in range(num_points):
    x_dif = x - midpoints[0, i]
    y_dif = y - midpoints[1, i]
    r = np.sqrt(np.pow(x_dif, 2) + np.pow(y_dif, 2))
    

    # plt.imshow(np.abs(r))
    # plt.show()

if scatterer == "PEC":
    s += -1j *  k0 * Z0 * J[i] * Sm[i] * G0(r)
elif scatterer == "dielectric":   #Dielectric scatter-field calc
    # Outward normal vector at segment i
    normal = -midpoints[:, i] / norm(midpoints[:, i])

    # Dot product with source to observation direction
    dot_normal = (x_dif * normal[0] + y_dif * normal[1]) / r 

    # Normal derivative of the Green's function
    dG_dn = (k0 / (4j)) * hankel2(1, k0 * r) * dot_normal

    # Dielectric scattered electric field
    s += Sm[i] * (1j * k0 * Z0 * H_boundary[i] * G0(r) - E_boundary[i] * dG_dn)


E_total = E_inc_grid + s  #total E field

# s[cylinder_mask] = 0

s_plot = s.copy()

if scatterer == "PEC":
    E_total[cylinder_mask] = 0
    s_plot[cylinder_mask] = np.nan # np.nan replaces points inside cylinder with "not a value"

elif scatterer == "dielectric":
    E_total[cylinder_mask] = np.nan
    s_plot[cylinder_mask] = np.nan


plt.figure()

plt.figure(figsize=(9, 5))

plt.plot(np.degrees(phi), sigma_2D_num,
         label="MoM", linewidth=2)

plt.plot(np.degrees(phi), sigma_2D_analytical,
         label="Analytical", linestyle="--", linewidth=2)

plt.xlabel("Scattering Angle (degrees)")
plt.ylabel("Bistatic Echo-Width (m)")
if scatterer == "PEC":
    plt.title("PEC Cylinder Bistatic Echo-Width at 10 GHz")
else:
    plt.title("Dielectric Cylinder Bistatic Echo-Width at 10 GHz")

plt.legend()
plt.grid(True)
plt.xlim(0, 360)
plt.tight_layout()
plt.show()


plt.imshow(np.abs(s_plot).T, cmap="viridis", extent=[a,b,a,b])
plt.colorbar(label="Electric Field Magnitude (V/m)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Scattered Electric Field Magnitude at 10 GHz")
plt.show()

plt.imshow(np.angle(s_plot).T, cmap="twilight", extent=[a,b,a,b])
plt.colorbar(label="Phase (radians)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Scattered Electric Field Phase at 10 GHz")
plt.show()

plt.imshow(np.abs(E_total).T, cmap="viridis", extent=[a,b,a,b])
plt.colorbar(label="Total Electric Field Magnitude (V/m)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Total Electric Field at 10 GHz")
plt.show()

print("END OF SCRIPT")
