import numpy as np


def crout(mat):
	# Performs Crout's LU decomposition and stores it in q = L + U - I
	n = mat.shape[0]
	q = np.zeros(mat.shape)
	for j in range(n):
		for k in range(j, n):
			q[k, j] = mat[k, j] - sum([q[k, i] * q[i, j] for i in range(j)])
		for k in range(j+1, n):
			q[j, k] = 1/q[j, j]*(mat[j, k] - sum([q[j, i] * q[i, k] for i in range(j)]))
	return q


def lu_solve(q, b):
	# Solves the matrix equation, L*U*x = b for x.
	# L and U are stored in the matrix q = L + U - I
	n = b.shape[0]
	y = np.zeros(b.shape)
	x = np.zeros(b.shape)
	for i in range(n):
		y[i] = 1/q[i, i]*(b[i] - sum([q[i, j]*y[j] for j in range(i)]))
	for i in range(n):
		i = n - i - 1  # Backwards
		x[i] = y[i] - (sum([q[i, j]*x[j] for j in range(i, n)]))
	return x


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

