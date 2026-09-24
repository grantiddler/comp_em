import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(2 * np.pi * x)

def dfdx(x):
    return 2 * np.pi * np.cos(2 * np.pi * x)


def forward_diff(x, d):
    return (f(x + d) - f(x)) / d

def central_diff(x, d):
    return (f(x + d) - f(x - d)) / (2 * d)

def backward_diff(x,d):
    return (f(x) - f(x - d)) / d


h = np.pow(10, np.linspace(0, -10, 200))
print(h)
x = np.pi / 5


fd_err = np.abs( (forward_diff(x, h) - dfdx(x)) / dfdx(x) )
cd_err = np.abs( (central_diff(x, h) - dfdx(x)) / dfdx(x) )
bd_err = np.abs( (backward_diff(x, h) - dfdx(x)) / dfdx(x) )


plt.loglog(1/h, fd_err, label="Forward differencing")
plt.loglog(1/h, cd_err, label="Central differencing")
plt.loglog(1/h, bd_err, label="Backward differencing")
plt.legend()
plt.xlabel("inverse step size")
plt.ylabel("relative error")
plt.title("Inverse step size vs error for differencing methods")
plt.show()