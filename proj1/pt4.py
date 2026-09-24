import numpy as np

def trap(f, a, b):
    d = (b - a)
    return (f(a) + f(b)) * d / 2


def simpsons_13(f,a,b):
    d = (b - a) / 2
    return (f(a) + 4 * f(a + d) + f(b)) * d / 3

def simpsons_38(f,a,b):
    d = (b - a) / 3
    return (f(a) + 3 * f(a + d) + 3 * f(a + 2 * d) + f(b)) * d * (3 / 8)


f = lambda x: np.exp(-x)
g = lambda x: 1 - np.exp(-x)


# print(f"trapezoid: {trap(f, 0, 2)}, error: {(trap(f, 0, 2) - g(2)) / g(2)}")

# print(f"simpsons 1/3 rule: {simpsons_13(f, 0, 2)}, error: {(simpsons_13(f, 0, 2) - g(2)) / g(2)}")
# print(f"trapezoid 2 subdivisions: {trap(f,0,1) + trap(f,1,2)}, error {(trap(f,0,1) + trap(f,1,2) - g(2)) / g(2)}")
# print(f"simpsons 3/8 rule: {simpsons_38(f, 0, 2)}, error: {(simpsons_38(f, 0, 2) - g(2)) / g(2)}")
# print(f"trapezoid, 3 subdivisions: {trap(f,0, 2/3) + trap(f,2/3, 4/3) + trap(f,4/3,2)}, error {(trap(f,0, 2/3) + trap(f,2/3, 4/3) + trap(f,4/3,2) - g(2)) /g(2)}")\
print(g(2))
