import matplotlib.pyplot as plt
import numpy as np

## DIELECTRIC
# R= .012
length =            [0.005,    0.0025,   0.001,   0.00075, 0.0005, 0.00025, 0.000125]
err_dielectric =    [209.3802, 124.0203, 16.3398, 12.6177, 7.6774, 2.7003,  1.1627]
err_PEC =           [154.0282,  25.2266,  3.5160,  3.5160, 0.9718, 0.3643,  0.2293]

plt.plot(length, err_dielectric, label = "Dielectric")
plt.plot(length, err_PEC, label = "PEC")
plt.legend()
plt.title("Mesh size vs. echo width error")
plt.xlabel("Mesh characteristic length (m)")
plt.ylabel("Bistatic echo width RMSE (%)")

plt.show()

# length = .0005
abc_size =       [0.025,   0.05,   0.075,  0.1,    0.25]
err_dielectric = [19.1386, 8.6471, 8.1178, 7.4763, 7.4519]
err_PEC =        [6.0736,  2.8393, 0.9916, 0.6381, 1.8359]


plt.plot(abc_size, err_dielectric, label = "Dielectric")
plt.plot(abc_size, err_PEC, label = "PEC")
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

