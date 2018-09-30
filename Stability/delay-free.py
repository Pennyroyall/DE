import numpy as np
from scipy.linalg import expm
from scipy.special import lambertw
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import warnings


def kronecker_product(a, b):
    n = np.shape(a)[0]
    s = np.dot(b[0, 0], a)
    for i in range(0, n):
        temp = np.dot(b[0, i], a)
        for j in range(1, n):
            temp = np.concatenate((temp, np.dot(b[j, i], a)), axis=1)
        if i <= 0:
            s = temp
        else:
            s = np.concatenate((s, temp), axis=0)
    return s


def to_matrix(a):
    n = int(np.sqrt(int(np.shape(a)[0])))
    temp = np.zeros([n, n])
    for i in range(0, n):
        for j in range(0, n):
            temp[i, j] = a[i * n + j, 0]
    return temp.T


def to_vector(a):
    n = np.shape(a)[0]
    temp = np.zeros([n * n, 1])
    for i in range(0, n):
        for j in range(0, n):
            temp[i * n + j, 0] = a[j, i]
    return temp


def delayfree_margin(A0, number_of_iterations=50):
    n = np.shape(A0)[0]
    W = 1*np.eye(n)
    l = min(np.linalg.eigvals(W))
    delta = 0

    for i in range(0, number_of_iterations):
        A = A0 + delta*np.eye(n)
        P = kronecker_product(A.T, np.eye(n)) + kronecker_product(np.eye(n), A)
        V = -to_matrix(np.linalg.inv(P) * to_vector(W))
        v = max(np.linalg.eigvals(V))
        delta = delta + l/(2*v)
    return delta


def solver(A0, N, T):
    n = np.shape(A0)[0]
    k = T/N
    X = np.zeros([n, n, N+1])
    X[:, :, 0] = np.eye(n)
    for j in range(0, N):
        R1 = k * (A0.T * X[:, :, j] + X[:, :, j] * A0)
        R2 = k * (A0.T * (X[:, :, j] + (1/2)*R1) + (X[:, :, j] + (1/2)*R1)*A0)
        R3 = k * (A0.T*(X[:, :, j]+(np.sqrt(2)/2-1/2)*R1 + (1-1/np.sqrt(2))*R2) + (X[:, :, j]+(np.sqrt(2)/2-1/2)*R1 + (1-1/np.sqrt(2))*R2)*A0)
        R4 = k * (A0.T*(X[:, :, j] - (1/np.sqrt(2))*R2 + (1+1/np.sqrt(2))*R3) + (X[:, :, j] - (1/np.sqrt(2))*R2 + (1+1/np.sqrt(2))*R3)*A0)
        X[:, :, j+1] = X[:, :, j] + 1/6*R1 + (1/3-1/(3*np.sqrt(2)))*R2 + (1/3+1/(3*np.sqrt(2)))*R3 + 1/6*R4
    return X


def delayfree_gamma(A0, sigma, number_of_iterations=10000, PlotFlag="False"):
    T = 4*3/sigma
    X = solver(A0, number_of_iterations, T)
    f = np.zeros(number_of_iterations)
    F = np.zeros(number_of_iterations)

    for j in range(0, number_of_iterations):
        f[j] = max(np.linalg.eigvals(X[:,:,j]))
        F[j] = np.sqrt(f[j] * np.exp(2*sigma*j*(T/number_of_iterations)))

    if PlotFlag is "True":
        plt.plot(F)
        plt.xlabel('discrete time')
        plt.ylabel('gamma')
        plt.show()
    return max(F)


"""System"""
warnings.simplefilter('ignore')
t = time.time()

# A = np.matrix([[-11, 10],
#                [-1, -9]])
# print(np.linalg.eigvals(A))
#
# sigma = delayfree_margin(A, 50)
# gamma = delayfree_gamma(A, sigma, 1000, "True")


# A = np.matrix([[-17, 1000],
#                [-19, -41]])
# print(np.linalg.eigvals(A))
#
# sigma = delayfree_margin(A, 50)
# gamma = delayfree_gamma(A, sigma, 1000, "True")


h = 1/2
Ah = np.matrix([[-4, 1],
                [0, -4]])
Bh = np.matrix([[0.1, 0],
                [4, 0.1]])
n = np.shape(Ah)[0]

Ah = np.matrix([[-61/9, 16/9, -11/9],
               [1/3, -16/3, -1/3],
               [-19/9, -8/9, -53/9]])
Bh = np.zeros_like(Ah)
n = np.shape(Ah)[0]



A = Ah + Bh
print(np.linalg.eigvals(A))

sigma = delayfree_margin(A, 100)
gamma = delayfree_gamma(A, sigma, 1000, "True")


print("Sigma is:", sigma)
print("Gamma is:", gamma)




print('Time of evaluating  is:', time.time() - t)
