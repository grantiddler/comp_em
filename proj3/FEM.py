import pygmsh
import matplotlib.pyplot as plt
import numpy as np

with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_min = 0.1
    geom.characteristic_length_max = 5


    disk1 = geom.add_disk([0, 0.0, 0.0], 0.5)
    disk2 = geom.add_disk([0, 0.0, 0.0], 0.015)

    # flat = geom.boolean_fragments(disk1, disk2)
    flat = geom.boolean_difference(disk1, disk2)



    mesh = geom.generate_mesh()





# print(mesh.points)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


colors = ["blue", "purple"]

segments = []
for i in mesh.cells[2].data:
    segments.append(i[0])
seg = -1

boundaries = []
for i in mesh.cells[0].data:
    if(i[0] == seg + 1 and (len(mesh.cells[2]) > 1 or seg == -1)):
        seg += 1
        boundaries.append([])
    boundaries[seg].append(i[0])
    # plt.plot()
    # plt.plot([mesh.points[i[0]][0], mesh.points[i[1]][0]], [mesh.points[i[0]][1], mesh.points[i[1]][1]], color=colors[seg])



Btype=['d', 'd']
Bval=[1, 0]



N = len(mesh.points)

K = np.zeros([N, N])
B = np.zeros([N, 1])


for i in mesh.cells[1].data:
    color = "red"
    e_r = 1
    # if np.sqrt(((mesh.points[i[0]][0] + mesh.points[i[1]][0] + mesh.points[i[2]][0])/3) ** 2 + ((mesh.points[i[0]][1] + mesh.points[i[1]][1] + mesh.points[i[2]][1])/3) ** 2) <= .015:
    #     color="black"
    #     e_r = 9

    B1 = False
    B2 = False
    B3 = False

    boundary_value = 1
    for j in range(len(boundaries)):
        B1 = B1 or i[0] in boundaries[j]
        B2 = B2 or i[1] in boundaries[j]
        B3 = B3 or i[2] in boundaries[j]
        # 
        if Btype[j] == 'd' and (i[0] in boundaries[j] or i[1] in boundaries[j] or i[2] in boundaries[j]):
            color = "purple"
            boundary_value = Bval[j]

        
    D = (mesh.points[i[2]][0] - mesh.points[i[1]][0]) * (mesh.points[i[1]][1] - mesh.points[i[0]][1]) + (mesh.points[i[1]][0] - mesh.points[i[0]][0]) * (mesh.points[i[1]][1] - mesh.points[i[2]][1])
    Area = np.abs(D)/2
    inner_product = Area / 12

    Beta = 0
    Alpha = 1e-12

    i1 = i[0]
    i2 = i[1]
    i3 = i[2]

    p1 = mesh.points[i1]
    p2 = mesh.points[i2]
    p3 = mesh.points[i3]

    dN1dx = (p2[1] - p3[1])
    dN2dx = (p3[1] - p1[1])
    dN3dx = (p1[1] - p2[1])

    dN1dy = (p3[0] - p2[0])
    dN2dy = (p1[0] - p3[0])
    dN3dy = (p2[0] - p1[0])

    N1N2 = (dN1dx * dN2dx) + (dN1dy * dN2dy)
    N2N3 = (dN2dx * dN3dx) + (dN2dy * dN3dy)
    N3N1 = (dN1dx * dN3dx) + (dN1dy * dN3dy)


    N1N1 = (dN1dx * dN1dx) + (dN1dy * dN1dy)
    N2N2 = (dN2dx * dN2dx) + (dN2dy * dN2dy)
    N3N3 = (dN3dx * dN3dx) + (dN3dy * dN3dy)


    if B1:
        B[i1] -= N1N1 * boundary_value * Alpha
        B[i2] -= N1N2 * boundary_value * Alpha
        B[i3] -= N3N1 * boundary_value * Alpha
    if B2:
        B[i1] -= N1N2 * boundary_value * Alpha
        B[i2] -= N2N2 * boundary_value * Alpha
        B[i3] -= N2N3 * boundary_value * Alpha
    if B3:
        B[i1] -= N3N1 * boundary_value * Alpha
        B[i2] -= N2N3 * boundary_value * Alpha
        B[i3] -= N3N3 * boundary_value * Alpha
    


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
 



Nd = 0
pts = mesh.points.copy()


for i in range(len(boundaries)):
    if Btype[i] == 'd':
        for j in range(len(boundaries[i])):
            ax.scatter(mesh.points[boundaries[i][j]][0], mesh.points[boundaries[i][j]][1], Bval[i], 'o', color="black")
        




while True:
    run = False
    m = 0
    for i in range(len(boundaries)):
        if Btype[i] == 'd' and boundaries[i] != []:
            m = max(m, max(boundaries[i]))
            run = True

    for i in range(len(boundaries)):
        if Btype[i] == 'd' and (m in boundaries[i]):
            boundaries[i].remove(m)

    if not run:
        break

    K[m:-1,:] = K[m+1:,:]
    K[:,m:-1] = K[:,m+1:]
    B[m:-1] = B[m+1:]
    pts[m:-1] = pts[m+1:]
    Nd += 1


    





if Nd != 0:
    K = K[0:-Nd, 0:-Nd]
    B = B[0:-Nd]
    pts = pts[0:-Nd]

# for i in mesh.cells[1].data:
#     if(0 in i):
#         print(i)
E = np.linalg.inv(K) @ B
print(E)


# x = []
# I = 0

for i in mesh.points:
    ax.scatter(i[0], i[1], 0,'o', color="red")

for i in range(len(pts)):
    ax.scatter(pts[i][0], pts[i][1], E[i]/4000,'o', color="purple")




plt.show()

# for i in mesh.points:
#     if np.sqrt(i[0] ** 2 + i[1] ** 2) <= .015001:
#         plt.plot(i[0], i[1], 'o', color="black")
#         # plt.plot(i[0], i[1], 'o', color="black")
    # else: