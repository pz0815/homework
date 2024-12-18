import numpy as np
import numpy.linalg as la

def scale_to_range(X: np.ndarray, to_range=(0, 1), byrow=False):
    """
    Parameters
    ----------
    X: 
        1D or 2D array
    
    to_range: default to (0,1).
        Desired range of transformed data.
        
    byrow: default to False
        When working with a 2D array, true to perform row mapping; 
        otherwise, column mapping. Ignore if X is 1D. 
    
    ----------
    
    """
    a, b = to_range
    Y = np.zeros(X.shape)
    
    if X.ndim == 1:
        Y = (X - X.min()) / (X.max() - X.min()) * (b - a) + a

    elif X.ndim == 2:
        if byrow:
            min_X = X.min(axis=1).reshape(-1, 1)
            max_X = X.max(axis=1).reshape(-1, 1)
            Y = (X - min_X) / (max_X - min_X) * (b - a) + a
        else:
            min_X = X.min(axis=0).reshape(1, -1)
            max_X = X.max(axis=0).reshape(1, -1)
            Y = (X - min_X) / (max_X - min_X) * (b - a) + a
   
    return Y

print('test case 1:')
A = np.array([1, 2.5, 6, 4, 5])
print(f'A => \n{A}')
print(f'scale_to_range(A) => \n{scale_to_range(A)}\n\n')

print('test case 2:')
A = np.array([[1, 12, 3, 7, 8],
              [5, 14, 1, 5, 5],
              [4, 11, 4, 1, 2],
              [3, 13, 2, 3, 5],
              [2, 15, 6, 3, 2]])
print(f'A => \n{A}')
print(f'scale_to_range(A) => \n{scale_to_range(A)}\n\n')

print('test case 3:')
A = np.array([[1, 2, 3, 4, 5],
              [5, 4, 1, 2, 3],
              [3, 5, 4, 1, 2]])
print(f'A => \n{A}')
print(f'scale_to_range(A, byrow=True) => \n{scale_to_range(A, byrow=True)}\n\n')
