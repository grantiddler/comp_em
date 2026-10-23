import pygmsh
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv, jvp, hankel2, h2vp
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.tri as mtri

scatterer = ["Dielectric", "PEC"][0]

e_c = 9

with pygmsh.occ.Geometry() as geom:
    R = 0.1
    a = 0.015
    tol = 1e-5
    c0 = 299792458
    f = 10e9
    k0 = 2 * np.pi * f / c0
    gamma = 1j * k0 + 1 / (2 * R)

    # resolution l_0= 0.03m
    # geom.characteristic_length_min = 0.0015
    geom.characteristic_length_max = 0.000125

    
    disk1 = geom.add_disk([0, 0.0, 0.0], R)
    disk2 = geom.add_disk([0, 0.0, 0.0], a)

    if scatterer == "Dielectric":
        flat = geom.boolean_fragments(disk1, disk2)
    else:
        flat = geom.boolean_difference(disk1, disk2)



    mesh = geom.generate_mesh()

    #----------------#
    # Identify outer ABC edges
    #----------------#

    outer_edges = []

    for block in mesh.cells:
        if block.type != "line":
            continue

        for edge in block.data:
            i, j = edge

            ri = np.linalg.norm(mesh.points[i, :2])
            rj = np.linalg.norm(mesh.points[j, :2])

            if abs(ri - R) < tol and abs(rj - R) < tol:
                outer_edges.append((i, j))

print("Number of outer ABC edges:", len(outer_edges))



#----------------#
# Identify inner boundary of cylinder
#----------------#
N = len(mesh.points)

inner_nodes = set()

if scatterer == "PEC":
    for block in mesh.cells:
        if block.type != "line":
            continue

        for edge in block.data:
            for node in edge:
                r = np.linalg.norm(mesh.points[node, :2])

                if abs(r - a) < tol:
                    inner_nodes.add(int(node))

    print("Number of inner PEC nodes:", len(inner_nodes))



N = len(mesh.points)

K = lil_matrix((N, N), dtype=complex) #makes sparse matrix
B = np.zeros((N, 1), dtype=complex)


for i in mesh.cells[1].data:

    e_r = 1
    if np.sqrt(((mesh.points[i[0]][0] + mesh.points[i[1]][0] + mesh.points[i[2]][0])/3) ** 2 + ((mesh.points[i[0]][1] + mesh.points[i[1]][1] + mesh.points[i[2]][1])/3) ** 2) <= .015:
        e_r = e_c


    Alpha = 1.0
    Beta = -k0**2 * e_r

    i1 = i[0]
    i2 = i[1]
    i3 = i[2]

    p1 = mesh.points[i1]
    p2 = mesh.points[i2]
    p3 = mesh.points[i3]


    #shape-function derivatives
    D = ((p2[0] - p1[0]) * (p3[1] - p1[1])
        - (p3[0] - p1[0]) * (p2[1] - p1[1]))

    Area = abs(D) / 2
    inner_product = Area / 12

    dN1dx = (p2[1] - p3[1]) / D
    dN2dx = (p3[1] - p1[1]) / D
    dN3dx = (p1[1] - p2[1]) / D

    dN1dy = (p3[0] - p2[0]) / D
    dN2dy = (p1[0] - p3[0]) / D
    dN3dy = (p2[0] - p1[0]) / D

    N1N2 = (dN1dx * dN2dx) + (dN1dy * dN2dy)
    N2N3 = (dN2dx * dN3dx) + (dN2dy * dN3dy)
    N3N1 = (dN1dx * dN3dx) + (dN1dy * dN3dy)


    N1N1 = (dN1dx * dN1dx) + (dN1dy * dN1dy)
    N2N2 = (dN2dx * dN2dx) + (dN2dy * dN2dy)
    N3N3 = (dN3dx * dN3dx) + (dN3dy * dN3dy)




    K[i1, i1] += inner_product * 2 * Beta + Area * Alpha * N1N1
    K[i2, i2] += inner_product * 2 * Beta + Area * Alpha * N2N2
    K[i3, i3] += inner_product * 2 * Beta + Area * Alpha * N3N3
    
    
    K[i1, i2] += inner_product * Beta + Area * Alpha * N1N2
    K[i2, i1] += inner_product * Beta + Area * Alpha * N1N2

    K[i2, i3] += inner_product * Beta + Area * Alpha * N2N3
    K[i3, i2] += inner_product * Beta + Area * Alpha * N2N3

    K[i3, i1] += inner_product * Beta + Area * Alpha * N3N1
    K[i1, i3] += inner_product * Beta + Area * Alpha * N3N1
    

    # plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '--',color=color)
    # plt.plot([p2[0], p3[0]], [p2[1], p3[1]], '--',color=color)
    # plt.plot([p3[0], p1[0]], [p3[1], p1[1]], '--',color=color)

for i, j in outer_edges:

    p1 = mesh.points[i, :2]
    p2 = mesh.points[j, :2]

    L = np.linalg.norm(p2 - p1)

    K_edge = gamma * L / 6 * np.array([
        [2, 1],
        [1, 2]
    ])

    # Incident plane wave evaluated at edge midpoint
    midpoint = (p1 + p2) / 2

    x_mid = midpoint[0]
    nx = x_mid / np.linalg.norm(midpoint)

    E_inc = np.exp(-1j * k0 * x_mid)

    g = (gamma - 1j * k0 * nx) * E_inc

    # Midpoint approximation to boundary excitation
    B[i, 0] += g * L / 2
    B[j, 0] += g * L / 2

    nodes = [i, j]

    #adds K_edge to global stiffness matrix K
    for local_i in range(2):
        for local_j in range(2):
            K[nodes[local_i], nodes[local_j]] += K_edge[local_i, local_j]


#----------------#
# Apply PEC Dirichlet conditions
#----------------#

# TMz total field: Ez = 0 on PEC
pec_nodes = np.array(sorted(inner_nodes), dtype=int)

# Keep only unknown (non-PEC) nodes
free_nodes = np.setdiff1d(np.arange(N), pec_nodes)

# Convert sparse matrix from LIL to CSR
K = K.tocsr()

# Apply PEC Dirichlet boundary conditions
K_reduced = K[free_nodes, :][:, free_nodes]

B_reduced = B[free_nodes, 0]

print("Solving sparse FEM system...", flush=True)

E_reduced = spsolve(K_reduced, B_reduced)

print("Sparse matrix solved!", flush=True)


# Reconstruct field at all original mesh nodes using sparse matrix
#spsolve() returns a 1D array
E = np.zeros(N, dtype=complex)
E[free_nodes] = E_reduced
E[pec_nodes] = 0

#calcuate scattered field
E_inc_nodes = np.exp(-1j * k0 * mesh.points[:, 0])

E_scat = E - E_inc_nodes

# print(
#     "Maximum PEC boundary field:",
#     np.max(np.abs(E[pec_nodes]))
# )
outer_nodes = np.unique(np.array(outer_edges).flatten())

#Analytical TMz scattering for PEC cylinder
x = mesh.points[:, 0]
y = mesh.points[:, 1]

rho = np.sqrt(x**2 + y**2)
phi = np.arctan2(y, x)

Nh = 50

E_scat_analytical = np.zeros(N, dtype=complex)

for n in range(-Nh, Nh + 1):

    if scatterer == "Dielectric":
        kd = np.sqrt(e_c) * k0
        
        an = 1j * (1j ** n) * ((jvp(n, k0*a)*jv(n, kd*a)) - (np.sqrt(e_c)*jv(n, k0*a)*jvp(n, kd*a))
                                ) / ((h2vp(n, k0*a)*jv(n, kd*a))- (np.sqrt(e_c)*hankel2(n, k0*a)*jvp(n, kd*a))) 
    
    else:
        an = -((-1j)**n) * (jv(n, k0 * a) / hankel2(n, k0 * a))


    E_scat_analytical += (an * hankel2(n, k0 * rho) * np.exp(1j * n * phi))

E_total_analytical = E_inc_nodes + E_scat_analytical

error = E_scat - E_scat_analytical

# Entire computational domain
relative_rmse_full = np.sqrt(np.sum(np.abs(error)**2) / np.sum(np.abs(E_scat_analytical)**2))

# Fixed comparison region
comparison_region = ((rho >= a - 1e-8) & (rho <= 0.04))

relative_rmse_fixed = np.sqrt(np.sum(np.abs(error[comparison_region])**2) / np.sum(np.abs(E_scat_analytical[comparison_region])**2))

print(f"Full-domain RMSE: {relative_rmse_full:.4%}")
print(f"Fixed-region RMSE: {relative_rmse_fixed:.4%}")

triangles = mesh.get_cells_type("triangle")

fig, axes = plt.subplots(2,3, figsize=(15, 5))
axes = np.reshape(axes, (1,6))

fields = [
    ("Scattered Field Magnitude", np.abs(E_scat), "viridis", "normalized field intensity (|E| / |E0|)"),
    ("Instantanious Scattered Field (phase = 0)", np.real(E_scat), "RdBu_r", "normalized field intensity (E / |E0|)"),
    ("Scattered Field Phase", np.angle(E_scat), "twilight", "field phase (radians)"),

    ("Total Field Magnitude", np.abs(E), "viridis", "normalized field intensity (|E| / |E0|)"),
    ("Instantanious Total Field (phase = 0)", np.real(E), "RdBu_r", "normalized field intensity (E / |E0|)"),
    ("Total Field Phase", np.angle(E), "twilight", "field phase (radians)"),
    
]

theta = np.linspace(0, np.pi * 2, 100)
x_circ = np.sin(theta) * a
y_circ = np.cos(theta) * a


for ax, (title, field, color, clabel) in zip(axes[0], fields):
    values = field

    # is_error = title == "Absolute Error"

    im = ax.tripcolor(
        mesh.points[:, 0],
        mesh.points[:, 1],
        triangles,
        values,
        shading="gouraud",
        cmap=color,
        vmin= -np.pi if ("Phase" in title) else (0 if "Magnitude" in title else -1),
        vmax= np.pi if ("Phase" in title) else (None if "Magnitude" in title else 1)
    )

    fig.colorbar(im, ax=ax, shrink=0.75, label=clabel)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # if scatterer == "Dielectric":
    ax.plot(x_circ,y_circ, '--', color="black")

plt.tight_layout()
plt.show()


# Angular sampling
Nphi = 360
angles = np.linspace(0, 2*np.pi, Nphi, endpoint=False)

# Observation circle inside ABC
r_obs = 0.099

x_obs = r_obs * np.cos(angles)
y_obs = r_obs * np.sin(angles)

# ----------------------------------------
# Interpolate FEM scattered field
# onto observation circle
# ----------------------------------------

triang = mtri.Triangulation(
    mesh.points[:, 0],
    mesh.points[:, 1],
    triangles
)

interp_real = mtri.LinearTriInterpolator(triang, np.real(E_scat))
interp_imag = mtri.LinearTriInterpolator(triang, np.imag(E_scat))

E_obs_real = interp_real(x_obs, y_obs)
E_obs_imag = interp_imag(x_obs, y_obs)

if np.any(np.ma.getmaskarray(E_obs_real)) or \
   np.any(np.ma.getmaskarray(E_obs_imag)):
    raise ValueError("Observation circle extends outside FEM mesh")

E_obs = (
    np.asarray(E_obs_real)
    + 1j * np.asarray(E_obs_imag)
)

# ----------------------------------------
# Extract FEM cylindrical coefficients
# ----------------------------------------

# Limit expansion to resolved angular modes
Nfit = 16
n_modes = np.arange(-Nfit, Nfit + 1)

# Fourier coefficients of field on observation circle
fourier_coeff = np.fft.fft(E_obs) / Nphi

# E_scat(r,phi) = sum c_n H_n^(2)(kr) exp(jn phi)
c_fem = np.array([
    fourier_coeff[n % Nphi] / hankel2(n, k0 * r_obs)
    for n in n_modes
])

# ----------------------------------------
# Analytical coefficients (50 harmonics
# in each direction)
# ----------------------------------------

n_analytical = np.arange(-50, 51)

if scatterer == "PEC":
    c_analytical = np.array([
        -((-1j)**int(n)) *
        jv(n, k0*a) / hankel2(n, k0*a)
        for n in n_analytical
    ])
else:
    c_analytical = np.array([
        (-1j ** n) * ((jvp(n, k0*a)*jv(n, kd*a)) - (np.sqrt(e_c)*jv(n, k0*a)*jvp(n, kd*a))
                                    ) / ((h2vp(n, k0*a)*jv(n, kd*a))- (np.sqrt(e_c)*hankel2(n, k0*a)*jvp(n, kd*a))) 
        for n in n_analytical
    ])

# ----------------------------------------
# Far-field angular amplitudes
# ----------------------------------------

F_fem = np.zeros(Nphi, dtype=complex)
F_analytical = np.zeros(Nphi, dtype=complex)


for n, cn in zip(n_modes, c_fem):
    F_fem += cn * np.exp(1j*n*np.pi/2) * np.exp(1j*n*angles)


if scatterer == "Dielectric":
    angles += np.pi

for n, cn in zip(n_analytical, c_analytical):
    F_analytical += cn * np.exp(1j*n*np.pi/2) * np.exp(1j*n*angles)

if scatterer == "Dielectric":
    angles -= np.pi
# ----------------------------------------
# Bistatic echo-width (meters)
# ----------------------------------------

sigma_fem = (4/k0) * np.abs(F_fem)**2
sigma_analytical = (4/k0) * np.abs(F_analytical)**2

# ----------------------------------------
# Echo-width RMSE
# ----------------------------------------

rmse_echo = np.sqrt(
    np.sum((sigma_fem - sigma_analytical)**2)
    / np.sum(sigma_analytical**2)
)

print(f"Bistatic echo-width RMSE: {rmse_echo:.4%}")

# ----------------------------------------
# Plot results
# ----------------------------------------

angles_deg = np.degrees(angles)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    angles_deg,
    sigma_analytical,
    label="Analytical",
    linewidth=2
)

ax.plot(
    angles_deg,
    sigma_fem,
    "--",
    label="FEM",
    linewidth=1.5
)

ax.set_xlabel("Scattering angle phi (degrees)")
ax.set_ylabel("Bistatic echo-width (m)")
ax.set_title(f"TMz {scatterer} Cylinder - Bistatic Echo-Width at 10 GHz")

ax.set_xlim(0, 360)
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()


plt.polar(
    angles_deg * np.pi / 180,
    sigma_analytical,
)

plt.polar(
    angles_deg * np.pi / 180,
    sigma_fem,
    "--",
)

plt.show()