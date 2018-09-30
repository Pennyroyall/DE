import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import pandas as pd


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


def dlti_margin(A0, number_of_iterations=10):
    n = np.shape(A0)[0]
    W = 1*np.eye(n)
    l = min(np.linalg.eigvals(W))
    delta = 0

    for i in range(0, number_of_iterations):
        A = A0*np.exp(delta)
        P = kronecker_product(A.T, A) - kronecker_product(np.eye(n), np.eye(n))
        V = -to_matrix(np.linalg.inv(P)*to_vector(W))
        v = max(np.linalg.eigvals(A.T*V*A))
        delta = delta + (1/2)*np.log(1+l/v)

    return delta, np.exp(-delta)


def dlti_gamma(A, sigma, l=100, PlotFlag="False"):
    n = np.shape(A)[0]
    f = np.zeros(l)
    F = np.zeros(l)
    Y = np.eye(np.shape(A)[0])
    for i in range(0, l):

        f[i] = (np.linalg.norm(Y, 2))
        F[i] = f[i]*np.exp(sigma*i)
        Y = A * Y
        # print("eigs", f[i])
    if PlotFlag is "True":
        plt.plot(F)
        plt.ylabel('some numbers')
        plt.show()
    return max(F)

"""System"""
warnings.simplefilter('ignore')
t = time.time()

# A0 = np.matrix([[0.1, 0.1],
#                 [-0.2, 0.8]])
# print(np.linalg.eigvals(A0))
#
# sigma, q = dlti_margin(A0, 10)
# print("Sigma is:", sigma, "\nMaximum abs of eigval is:", q)
#
# gamma = dlti_gamma(A0, sigma, 10, "True")
# print("Gamma is:", gamma)


# A0 = np.matrix([[0.4611111, 0.0583333],
#                 [0.66667, 0.35]])
# print(A0)
# print(np.linalg.eigvals(A0))
#
# sigma, q = dlti_margin(A0, 10)
# print("Sigma is:", sigma, "\nMaximum abs of eigval is:", q)
#
# gamma = dlti_gamma(A0, sigma, 10, "False")
# print("Gamma is:", gamma)


# h = 1/2
# Ah = np.matrix([[-4, 1],
#                 [0, -4]])
# Bh = np.matrix([[0.1, 0],
#                 [4, 0.1]])
# n = np.shape(Ah)[0]



# h = 1/2
# Ah = np.matrix([[-61/9, -2, -11/9],
#                [2, -16/3, -1/3],
#                [-19/9, -8/9, -53/9]])
# Bh = np.zeros_like(Ah)
# n = np.shape(Ah)[0]


# h = 2/9
# Ah = np.matrix([[-35, 0, 0, 0],
#                 [3, -2.35, -0, 0],
#                 [6, 0, -6, 0],
#                 [0, 0, 0, -2]])
# Bh = np.matrix([[0, 0, 0, 0],
#                 [0, 0, 0, 0],
#                 [0, 0, -0.0887, 0],
#                 [-0, 0, 0.00222, 0]])
#
# n = np.shape(Ah)[0]


h = 2/9
Ah = np.matrix([[-35, 0, 0, 0, 5],
                [3, -2.35, -0, 0, 0],
                [6, 0, -6, 0, 0],
                [0, 0, 0, -2, 0],
                [7, 0, 0, 0, -15]])
Bh = np.matrix([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -0.0887, 0, 0],
                [-0, 0, 0.00222, 0, -0],
                [-0, 0, -0, 0, 0]])

n = np.shape(Ah)[0]


# beta = 0
# Ah = Ah - beta*np.eye(n)
# Bh = np.exp(-beta*h)*Bh

AA = np.linalg.inv(np.eye(n) - h*Ah) * (np.eye(n) + h*Bh)
# print("AA is:", AA)
# print("Eigvals of AA is:", np.max(np.absolute(np.linalg.eigvals(AA.T))))

# sigma, q = dlti_margin(AA, 1500)
# print("Sigma is:", sigma, "\nMaximum abs of eigval is:", q)

gamma = dlti_gamma(AA, -np.log(np.max(np.absolute(np.linalg.eigvals(AA.T)))), 25, "False")
print("Gamma is:", gamma)


# Ah = np.matrix([[-4, 1],
#                 [0, -4]])
# Bh = np.matrix([[0.1, 0],
#                 [4, 0.1]])
# n = np.shape(Ah)[0]
#
# N = 125
# gammas = np.zeros(N)
# hpl = np.zeros(N)
# for i in range(0, N):
#     h = 1/100 + i*0.01
#     hpl[i] = h
#     AA = np.linalg.inv(np.eye(2) - h*Ah) * (np.eye(2) + h*Bh)
#     sigma, q = dlti_margin(AA, 25)
#
#     gamma = dlti_gamma(AA, sigma, 450, "False")
#     gammas[i] = gamma
#
# #for h0 = 0.001 30 iterations for sigma and 1600 for gamma
# print(min(gammas), max(gammas))
#
# plt.plot(hpl, gammas)
# plt.xlabel('h')
# plt.ylabel('gamma of h')
# plt.xlim([0, hpl[N-1]])
# plt.show()
# data = {'h': hpl, 'gamma_dLTI': gammas}
# df = pd.DataFrame(data=data)
# df.to_csv('g_dLTI.csv', index=False)



print('Time of evaluating  is:', time.time() - t)



# h = 2/9
# Ah = np.matrix([[-35, 0, 0, 0, 0],
#                 [0.168, -2.35, -0.00163, 0, 0],
#                 [0, 123, -0.0636, 0, 0],
#                 [0, 0, 0, -2, 0],
#                 [0, 0, 0, 0, -15]])
# Bh = np.matrix([[0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#                 [28.7, 1830, -0.0887, 0, 259],
#                 [-0.222, 3.97, 0.00222, 0, -2],
#                 [-0.416, 0, -0.00347, 0, 0]])
# n = np.shape(Ah)[0]

