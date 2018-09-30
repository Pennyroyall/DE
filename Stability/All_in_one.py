import numpy as np
from scipy.linalg import expm
from scipy.special import lambertw
from scipy.optimize import minimize
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


def to_symmetric_vector(a):
    n = np.shape(a)[0]
    temp = np.zeros([int(n * (n + 1) / 2), 1])
    k = 0
    for i in range(0, n):
        for j in range(i, n):
            temp[k, 0] = a[j, i]
            k += 1
    return temp


def to_symmetric_matrix(a):
    n = int(max(np.roots([1, 1, -2 * np.shape(a)[0]])))
    temp = np.zeros([n, n])
    k = 0
    for i in range(0, n):
        for j in range(i, n):
            temp[i, j] = a[k, 0]
            k += 1
    return temp.T + temp - np.diag(np.diag(temp))


def u_norm(A, B, W0, W1, W2, n, h):
    L = np.block([
        [kronecker_product(np.eye(n), A), kronecker_product(np.eye(n), B)],
        [kronecker_product(-B.T, np.eye(n)), kronecker_product(-A.T, np.eye(n))]
    ])
    M = np.block([
        [kronecker_product(np.eye(n), np.eye(n)), kronecker_product(np.zeros([n, n]), np.zeros([n, n]))],
        [kronecker_product(A.T, np.eye(n)) + kronecker_product(np.eye(n), A), kronecker_product(np.eye(n), B)]
    ])
    N = np.block([
        [kronecker_product(np.zeros([n, n]), np.zeros([n, n])), kronecker_product(-np.eye(n), np.eye(n))],
        [kronecker_product(B.T, np.eye(n)), kronecker_product(np.zeros([n, n]), np.zeros([n, n]))]
    ])
    P = M + N * expm(L * h)
    if np.linalg.det(P) == 0:
        print("LYAPUNOV'S CONDITION IS NOT SATISFIED P")
    if np.linalg.det(L) == 0:
        print("LYAPUNOV'S CONDITION IS NOT SATISFIED L ")
    Wtemp = W0 + W1 + h * W2
    yz0 = np.linalg.inv(P) * np.concatenate((np.zeros([n * n, 1]), to_vector(-Wtemp)), axis=0)
    yztest_1 = expm(L * (-h)) * yz0
    yztest_2 = expm(L * (-1 / 4)) * yz0
    v1 = np.linalg.norm(to_matrix(yz0[0:n * n, :]), 2)
    v2 = np.linalg.norm(to_matrix(yztest_1[0:n * n, :]), 2)
    v3 = np.linalg.norm(to_matrix(yztest_2[0:n * n, :]), 2)
    return max(v1, v2, v3)


def delta_find(bnorm, v, l0, l1, l2):
    delta_vector = np.zeros([1, 3])
    delta_vector[0, 0] = (l0 + bnorm * v) / (2 * v + v * bnorm * h) - \
                         (1 / h) * lambertw(bnorm * h *
                                            np.exp(h * (l0 + bnorm * v) / (2 * v + v * bnorm * h)) /
                                            (2 + bnorm * h))

    delta_vector[0, 1] = 1 / h * np.log(1 + l1 / (v * bnorm + bnorm * bnorm * v * h))

    delta_vector[0, 2] = (l2 + v * bnorm * bnorm) / (v * bnorm) - \
                         (1 / h) * lambertw(bnorm * h * np.exp(h * (l2 + v * bnorm * bnorm) / (v * bnorm)))
    return min(min(delta_vector))


def lmi(S):
    k = int(n * (n + 1) / 2)
    P = to_symmetric_matrix(np.matrix(S[:k]).T)
    Q = to_symmetric_matrix(np.matrix(S[k:]).T)
    beta = delta
    M = np.block([[P * A0 + A0.T * P + Q, P * B0],
                  [np.dot(B0.T, P), -np.exp(-2 * h * beta) * Q]])
    N = np.block([[P, np.zeros_like(P)],
                  [np.zeros_like(P), np.zeros_like(P)]])
    return max(np.linalg.eigvals(M + 2 * beta * N))


def gamma(S):
    if np.isnan(S[0]) is True:
        print('\n'*10)
        r = 100000
    else:
        k = int(n * (n + 1) / 2)
        P = to_symmetric_matrix(np.matrix(S[:k]).T)
        Q = to_symmetric_matrix(np.matrix(S[k:]).T)
        alpha2 = max(np.linalg.eigvals(P)) + h * max(np.linalg.eigvals(Q))
        alpha1 = min(np.linalg.eigvals(P))
        r = np.sqrt(alpha2 / alpha1)
    return r


def check_p(S):
    k = int(n * (n + 1) / 2)
    return min(np.linalg.eigvals(to_symmetric_matrix(np.matrix(S[:k]).T)))


def check_q(S):
    k = int(n * (n + 1) / 2)
    return min(np.linalg.eigvals(to_symmetric_matrix(np.matrix(S[k:]).T)))


def gamma_find(S0):
    k = int(n * (n + 1) / 2)
    cons = ({'type': 'ineq',
             'fun': lambda S: min(np.linalg.eigvals(to_symmetric_matrix(np.matrix(S[:k]).T)))},
            {'type': 'ineq',
             'fun': lambda S: min(np.linalg.eigvals(to_symmetric_matrix(np.matrix(S[k:]).T)))},
            {'type': 'ineq',
             'fun': lambda S: -lmi(S)})
    solution = minimize(gamma, S0, constraints=cons, jac=False, method='SLSQP')
    S0 = solution.x
    return S0, solution.fun


def margin(A0, B0, h, number_of_iterations=200, start=0, GammaFlag='False'):
    """Arbitrary matrices for the purposes of computing the Lyapunov's matrix"""
    W0 = 1 * np.eye(n)
    W1 = 1 * np.eye(n)
    W2 = 1 * np.eye(n)
    l0, u = (np.linalg.eig(W0))
    l1, u = (np.linalg.eig(W1))
    l2, u = (np.linalg.eig(W2))
    l0 = min(l0)
    l1 = min(l1)
    l2 = min(l2)

    """Initial matrix-guess for gamma"""
    P1 = np.eye(n)
    Q1 = np.eye(n)
    S0 = np.vstack([to_symmetric_vector(P1), to_symmetric_vector(Q1)]).ravel()
    """Initialization of algorithm's parameters"""
    global delta
    delta = start

    """Check if the system is delay-free, otherwise perform the algorithm"""
    if h == 0:
        for i in range(1, number_of_iterations + 1):
            A = A0 + B0 + delta * np.eye(n)
            B = np.zeros_like(B0)
            delta = delta + l0 / 2*u_norm(A, B, W0, W1, W2, n, h)
    else:
        """Algorithm itself"""
        for i in range(1, number_of_iterations + 1):
            A = A0 + delta * np.eye(n)
            B = np.exp(delta * h) * B0
            bnorm = np.linalg.norm(B, 2)
            v = u_norm(A, B, W0, W1, W2, n, h)
            delta = delta + delta_find(bnorm, v, l0, l1, l2)

            if (i % (number_of_iterations-1) == 0) & (GammaFlag is 'True'):
                k = int(n * (n + 1) / 2)
                print('Time BEFORE GAMMA of evaluating  is:', time.time() - t)
                [S0, gamma] = gamma_find(S0)
                print('delta is::', delta, 'gamma is::', gamma)

    """Some logging"""
    print('\n')
    print("The margin of stability is:", '%.30f' % delta)
    if GammaFlag is 'True':
        print("The value of overshoot is:", '%.30f' % gamma)
        np.set_printoptions(precision=6, suppress=True)
        # print("P matrix is:\n", to_symmetric_matrix(np.matrix(S0[:k]).T))
        # np.set_printoptions(precision=6, suppress=True)
        # print("Q matrix is:\n", to_symmetric_matrix(np.matrix(S0[k:]).T))
        # np.set_printoptions(precision=6, suppress=True)
        # print(check_q(S0))
        # print(check_p(S0))
    return delta


"""System"""
warnings.simplefilter('ignore')
t = time.time()

"""Given matrices/delay"""
global h, A0, B0, n


# h = 1/2
# A0 = np.matrix([[-4, 1],
#                 [0, -4]])
# B0 = np.matrix([[0.1, 0],
#                 [4, 0.1]])
# n = np.shape(A0)[0]



# h = 1/2
# A0 = np.matrix([[-61/9, -2, -11/9],
#                [2, -16/3, -1/3],
#                [-19/9, -8/9, -53/9]])
# B0 = np.zeros_like(A0)
# n = np.shape(A0)[0]


# h = 2/9
# A0 = np.matrix([[-35, 0, 0, 0],
#                 [3, -2.35, -0, 0],
#                 [6, 0, -6, 0],
#                 [0, 0, 0, -2]])
# B0 = np.matrix([[0, 0, 0, 0],
#                 [0, 0, 0, 0],
#                 [0, 0, -0.0887, 0],
#                 [-0, 0, 0.00222, 0]])
#
# n = np.shape(A0)[0]


# h = 2/9
# A0 = np.matrix([[-35, 0, 0, 0, 5],
#                 [3, -2.35, -0, 0, 0],
#                 [6, 0, -6, 0, 0],
#                 [0, 0, 0, -2, 0],
#                 [7, 0, 0, 0, -15]])
# B0 = np.matrix([[0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, -0.0887, 0, 0],
#                 [-0, 0, 0.00222, 0, -0],
#                 [-0, 0, -0, 0, 0]])
#
# n = np.shape(A0)[0]

h = 2/9
A0 = np.matrix([[-35, 0, 0, 0, 0],
                [0.168, -2.35, -0.00163, 0, 0],
                [0, 123, -0.0636, 0, 0],
                [0, 0, 0, -2, 0],
                [0, 0, 0, 0, -15]])
B0 = np.matrix([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [28.7, 1830, -0.0887, 0, 259],
                [-0.222, 3.97, 0.00222, 0, -2],
                [-0.416, 0, -0.00347, 0, 0]])

n = np.shape(A0)[0]

margin(A0, B0, h, 8*120000, 0.000004876566208622534759342670, GammaFlag='True')


print('Time of evaluating  is:', time.time() - t)

# Time BEFORE GAMMA of evaluating  is: 12154.654866933823
# delta is:: 4.876564162259456e-06 gamma is:: 46.03392515188937
#
#
# The margin of stability is: 0.000004876566208622534759342670
# The value of overshoot is: 46.033925151889370397384482203051
# Time of evaluating  is: 12158.069942474365

# 0.000004876566208622534759342670
# 0.000002823679324137629009783268
# 0.000000770144652051385492086411
