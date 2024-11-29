import numpy as np

A = np.array([
    0, 0, 1, 0,
    0, 0, 0, 1,
    0, 81.38, -93.49, 0.0038,
    0, -122.03, 89.97, -0.0058
]).reshape((4,4))

C = np.array([
    1, 0, 0, 0,
    0, 0, 0, 0
]).reshape((2, 4))

Obs_1 = C
Obs_2 = C @ A
Obs_3 = C @ (A @ A)
Obs_4 = C @ (A @ (A @ A))

observability_CA = np.concatenate((Obs_1, Obs_2, Obs_3, Obs_4), axis=0)

rank = np.linalg.matrix_rank(observability_CA)

print(rank)
print(observability_CA)
print(observability_CA.shape)