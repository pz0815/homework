import numpy as np
import numpy.linalg as la

def gram_schmidt(S1: np.ndarray):
    m, n = S1.shape
    S2 = np.zeros((m, n))
    
    for i in range(n):
        vec = S1[:, i]
        for j in range(i):
            vec -= np.dot(S2[:, j], S1[:, i]) * S2[:, j]
        
        S2[:, i] = vec / la.norm(vec)
    
    return S2

S1 = np.array([[ 7,  4,  7, -3, -9],
               [-1, -4, -4,  1, -4],
               [ 8,  0,  5, -6,  0],
               [-4,  1,  1, -1,  4],
               [ 2,  3, -5,  1,  8]], dtype=np.float64)

S2 = gram_schmidt(S1)

np.set_printoptions(precision=2, suppress=True)
print(f'S1 => \n{S1}')
print(f'S2.T @ S2 => \n{S2.T @ S2}')
