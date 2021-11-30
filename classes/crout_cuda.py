import numpy as np
from numba import njit
from numpy import sum

@njit
def crout(mat):
	# Performs Crout's LU decomposition and stores it in q = L + U - I
	n = mat.shape[0]
	q = np.zeros(mat.shape)
	for j in range(n):
		for k in range(j, n):
			summation = 0
			for i in range(j):
				summation = summation + q[k, i] * q[i, j]
			q[k, j] = mat[k, j] - summation
		for k in range(j+1, n):
			summation = 0
			for i in range(j):
				summation = summation + q[j, i] * q[i, k]
			q[j, k] = 1/q[j, j]*(mat[j, k] - summation)
	return q

@njit
def lu_solve(q, b):
	# Solves the matrix equation, L*U*x = b for x.
	# L and U are stored in the matrix q = L + U - I
	n = b.shape[0]
	y = np.zeros(b.shape)
	x = np.zeros(b.shape)
	for i in range(n):
		summation = 0
		for j in range(i):
			summation = summation + q[i, j]*y[j]
		y[i] = 1/q[i, i]*(b[i] - summation)
	for i in range(n):
		i = n - i - 1  # Backwards
		summation = 0
		for j in range(i, n):
			summation = summation + q[i, j]*x[j]
		x[i] = y[i] - summation
	return x

@njit
def mat_solve(mat, b):
	# Solve the matrix equation Ax=b for x, where A is input argument, mat.
	# returns the vector x
	return lu_solve(crout(mat), b)


if __name__ == "__main__":
	a = np.array([[1, 3, 4, 8], [2, 1, 2, 3], [4, 3, 5, 8], [9, 2, 7, 4]])
	c = np.array([[1], [1], [1], [1]])
	# c = np.array([1, 1, 1, 1])
	print(a)
	print(c)
	print(crout(a))
	print(mat_solve(a, c))

