import numpy as np
import matplotlib.pyplot as plt

# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1]
    return lambdas[sorted_idx], V[:, sorted_idx]

# SVD: A = U * Sigma * V^T
# V: eigenvector matrix of A^T * A; U: eigenvector matrix of A * A^T 
def mysvd(A):
    lambdas, V = myeig(A.T @ A, symmetric=True)
    lambdas, V = np.real(lambdas), np.real(V)
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V

pts = 50
x = np.linspace(-2, 2, pts)
y = np.zeros(x.shape)

pts2 = pts // 2
y[0:pts2] = -1
y[pts2:] = 1

argidx = np.argsort(x)
x = x[argidx]
y = y[argidx]

T0 = np.max(x) - np.min(x)
f0 = 1.0 / T0
omega0 = 2.0 * np.pi * f0

# Step1:
n_terms = 10
cos_terms = [np.cos(omega0 * (i + 1) * x) for i in range(n_terms)]
sin_terms = [np.sin(omega0 * (i + 1) * x) for i in range(n_terms)]
X = np.column_stack([np.ones(x.shape)] + cos_terms + sin_terms)

# Step2:
U, Sigma, VT = np.linalg.svd(X, full_matrices=False)

# Step3:
Sigma_inv = np.diag(1 / Sigma)  
a = VT.T @ Sigma_inv @ U.T @ y  

y_bar = X @ a

plt.plot(x, y_bar, 'g-', label='predicted values')
plt.plot(x, y, 'b-', label='true values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Square Wave Approximation')
plt.show()

