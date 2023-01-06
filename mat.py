import numpy as np
from scipy.linalg import svd
A = np.matrix[[1, 3, 5, 7],
                [11, 13, 17, 19],
                [23, 29, 31, 37],
                [41, 43, 47, 51]]
print(A)
x, b, t = svd(A)

