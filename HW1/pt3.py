import matplotlib.pyplot as plt
import numpy as np

k = np.array([0, 1, 2, 3, 4, 5])
x_k = np.array([0.5, 1, 4, 8, 9, 10])
y_k = np.array([1.477, 1.456, 1.2347, .675, .7155, 1])

def L(n, k, x_k, x):
    result = 1
    for i in range(n + 1):
        if i != k:
            result *= (x - x_k[i]) / (x_k[k] - x_k[i])
    return result
    
def p(x, y_k, n, x_k):
    result = 0
    for k in range(n + 1):
        result += L(n, k, x_k, x) * y_k[k]
    
    return result

def s(x, x_k, y_k):
    itr = 1
    y = np.zeros(np.shape(x))



    for i in range(len(x)):
        if x_k[itr] < x[i]:
            itr += 1
            
        if itr == len(x_k):
            return y

        y[i] = y_k[itr - 1] + (y_k[itr] - y_k[itr - 1]) * (x[i] - x_k[itr - 1] ) / (x_k[itr] - x_k[itr - 1])

    return y


def hat(x, k, xk, yk):
    ones = np.ones(np.shape(x))

    return yk[k] * ((xk[k] * ones <= x) * (xk[k+1] * ones >= x) * (xk[k+1] - x) / (xk[k+1] - xk[k]) + (xk[k-1] * ones <= x) * (xk[k] * ones >= x) * (xk[k-1] - x) / (xk[k-1] - xk[k]))









########### PART A #############

n = 5
x = np.linspace(0.5,10, 201)





plt.plot(x, L(n, 0, x_k, x), color='red', label="L0")
plt.plot(x_k, L(n, 0, x_k, x_k), 'o', color='red')

plt.plot(x, L(n, 3, x_k, x), color="green", label = "L3")
plt.plot(x_k, L(n, 3, x_k, x_k), 'o', color="green")

plt.plot(x, L(n, 4, x_k, x), color="blue", label = "L4")
plt.plot(x_k, L(n, 4, x_k, x_k), 'o', color="blue")
plt.legend()
plt.title("Lagrange basis polynomials")
plt.show()



########## PART B ##############

plt.plot(x_k, y_k, 'o')
plt.plot(x, p(x, y_k, n, x_k), label="p5(x)")
plt.plot(x, L(n, 1, x_k, x) * y_k[1], label="y1 * l1(x)")
plt.plot(x, L(n, 5, x_k, x) * y_k[5], label="y5 * l5(x)")
plt.legend()
plt.title("Lagrange polynomial interpolation")

plt.show()


############ PART C ##########


y = s(x,x_k,y_k)

plt.plot(x_k,y_k, 'o', label="Discrete data")
plt.plot(x,y, label="Linear vs polynomial interpolation")
plt.plot(x, p(x, y_k, n, x_k), label="Lagrange interpolation")
plt.legend()
plt.title("Linear interpolation")
plt.show()


######## PART D ############

plt.plot(x, p(x, y_k, n, x_k), label="p5(x)")
plt.plot(x,hat(x,2,x_k,y_k), label="y2 Λ2")
plt.plot(x,hat(x,3,x_k,y_k), label="y3 Λ3")
plt.plot(x,hat(x,2,x_k,y_k) + hat(x,3,x_k,y_k), label="y2 Λ2 + y3 Λ3")
plt.legend()
plt.title("Linear interpolation basis functions")

plt.show()