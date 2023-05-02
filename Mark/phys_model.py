import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, eye, kronsum
from scipy.sparse.linalg import spsolve

# Define the problem parameters
N = 4 # number of spectral elements in each direction
p = 4 # polynomial order of the basis functions
x = np.linspace(0, 1, (N + 1) * p + 1)
y = np.linspace(0, 1, (N + 1) * p + 1)
X, Y = np.meshgrid(x, y)

# Define the element-wise mass and stiffness matrices
def element_matrices(p, N):
    x = np.linspace(-1, 1, p + 1)
    h = np.diff(x) / 2
    z = np.zeros_like(x)
    I = np.eye(p + 1)
    V = np.zeros((p + 1, p + 1, 3))
    for i in range(3):
        if i == 0:
            V[:, :, i] = I[:, ::-1]
        elif i == 1:
            V[:, :, i] = I
        elif i == 2:
            V[:, :, i] = np.diag(h)
    M = np.einsum('ijk,lmk->ijlm', V, V)
    K = np.einsum('ijk,lmk->ijlm', V[:, :, :-1], V[:, :, :-1])
    K += np.einsum('ijk,lmk->ijlm', V[:, :, 1:], V[:, :, 1:])
    K -= np.einsum('ijk,lmk->ijlm', V[:, :, 1:], V[:, :, :-1])
    M = M.reshape(p + 1, p + 1, 1, 1)
    M = np.tile(M, (1, 1, N, N))
    K = K.reshape(p + 1, p + 1, 1, 1)
    K = np.tile(K, (1, 1, N, N))
    return M, K

# Construct the global mass and stiffness matrices
M_elem, K_elem = element_matrices(p, N)
M = kronsum(eye(N), M_elem)
K = kronsum(csr_matrix(([1, -1], ([0, 1], [0, 1]))), K_elem)
for i in range(1, N):
    M = kronsum(M, eye(p + 1))
    K = kronsum(K, K_elem)
A = K + M # the global system matrix

# Define the right-hand side function
f = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Apply Dirichlet boundary conditions
b = np.zeros((2 * (p + 1) * N, 1))
idx = np.concatenate([np.arange(0, p + 1), np.arange((N - 1) * (p + 1), N * (p + 1))])
b[idx] = np.sin(np.pi * x[idx])
A = coo_matrix(A)
rows = np.concatenate([idx, np.arange(A.shape[0] - len(idx), A.shape[0])])
cols = np.concatenate([np.zeros_like(idx), np.arange(len(idx), 2 * len(idx))])
data = np.concatenate([np.ones(len(idx)), np.ones(len(idx))])
A = A + co
