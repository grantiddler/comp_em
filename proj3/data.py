import matplotlib.pyplot as plt
import numpy as np



maximum_phase_error  =  np.array([1.422, 0.3919, 0.064, 0.03642, 0.0162, 0.00406])
characteristic_length = [0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.00025]

phase_err_per_length = maximum_phase_error / .2
plt.loglog(characteristic_length, maximum_phase_error)
plt.loglog(characteristic_length, maximum_phase_error, 'o', color="tab:blue")
plt.title("Mesh fineness vs. Phase error")
plt.ylabel("Phase error (average Rad/m)")
plt.xlabel("Maximum edge length (m)")
plt.show()

## DIELECTRIC
# R= .012
length =            [0.005,    0.0025,   0.001,   0.00075, 0.0005, 0.00025, 0.000125]
err_dielectric =    [209.3802, 124.0203, 16.3398, 12.6177, 7.6774, 2.7003,  1.1627]
err_PEC =           [154.0282,  25.2266, 3.5160,  2.0104,  0.9718,  0.3643,  0.2293]

plt.loglog(length, err_dielectric, label = "Dielectric")
plt.loglog(length, err_PEC, label = "PEC")
plt.legend()
plt.title("Mesh size vs. echo width error")
plt.xlabel("Mesh characteristic length (m)")
plt.ylabel("Bistatic echo width RMSE (%)")

plt.show()

# length = .0005
abc_size =       [0.025,   0.03,    0.04,    0.045,  0.05,   0.06,   0.075,  0.09,    0.1,    0.15,   0.2,    0.25,   0.3]
err_dielectric = [19.1386, 12.7467, 11.8566, 8.5661, 8.6471, 8.3938, 8.1178, 7.9171, 7.4763, 7.5801, 7.2677, 7.4519, 7.5980]
err_PEC =        [6.0736,  8.7603,  3.8808,  3.2240, 2.8393, 1.3884, 0.9916, 0.8944, 0.6381, 1.1578, 1.6634, 1.8359, 2.3691]


plt.loglog(abc_size, err_dielectric, label = "Dielectric")
plt.loglog(abc_size, err_PEC, label = "PEC")
plt.legend()
plt.title("Truncation radius vs. echo width error")
plt.xlabel("ABC surface radius (m)")
plt.ylabel("Bistatic echo width RMSE (%)")


plt.show()

r_obs =          [0.2   , 0.3   , 0.4   , 0.5   , 0.6   , 0.7   , 0.8   , 0.9   , 0.99]
err_dielectric = [7.2349, 6.9093, 7.4763, 6.7988, 6.4891, 7.2899, 6.5185, 6.3624, 7.22152]
err_PEC =        [0.7473, 0.6539, 0.6381, 0.8067, 0.9917, 0.8147, 1.4402, 1.4467, 1.2945]


plt.plot(r_obs, err_dielectric, label = "Dielectric")
plt.plot(r_obs, err_PEC, label = "PEC")
plt.legend()
plt.title("Observation surface radius vs. echo width error")
plt.xlabel("Observation surface radius (m)")
plt.ylabel("Bistatic echo width RMSE (%)")

plt.show()


time_FEM =  [1.35480, 5.3249, 9.2794, 23.304, 112.34]
error_FEM = [25.2266, 3.5160, 2.0104, 0.9718, 0.3643]

time_FDTD = [1.326, 6.131, 34.422, 125.1]
error_FDTD= [12.03, 6.677, 2.6314, 1.9]

plt.loglog(error_FEM, time_FEM, label="FEM")
plt.loglog(error_FDTD, time_FDTD, label="FDTD")

plt.title("Time vs. accuracy")
plt.ylabel("Time")
plt.xlabel("Bistatic echo width RMSE (%)")
plt.legend()

plt.show()

