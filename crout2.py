import numpy as np


def mat_solve(mat, b):
	# Solve the matrix equation Ax=b for x, where A is input argument, mat.
	# returns the vector x

	# Crout
	# Performs Crout's LU decomposition and stores it in Q = L + U - I
	n = mat.shape[0]
	q = np.zeros(mat.shape)
	# q = mat
	for j in range(n):  # For each column, j:
		for k in range(j, n):  # Fill in jth column of the L matrix starting at diagonal going down.
			q[k, j] = mat[k, j] - sum([q[k, i] * q[i, j] for i in range(j)])
		for k in range(j+1, n):  # Fill in the jth row of the U matrix starting after the diagonal going right.
			q[j, k] = 1/q[j, j]*(mat[j, k] - sum([q[j, i] * q[i, k] for i in range(j)]))
	print(q)
	# Forward-Backwards
	# Solves the matrix equation, L*U*x = b for x.
	# L and U are stored in the matrix q = L + U - I
	n = b.shape[0]
	y = np.zeros(b.shape)
	x = np.zeros(b.shape)
	for i in range(n):  # Forwards
		y[i] = 1/q[i, i]*(b[i] - sum([q[i, j]*y[j] for j in range(i)]))
	for i in range(n):  # Backwards
		i = n - i - 1
		x[i] = y[i] - (sum([q[i, j]*x[j] for j in range(i, n)]))
	return x







if __name__ == "__main__":
	a = np.array([[1, 3, 4, 8], [2, 1, 2, 3], [4, 3, 5, 8], [9, 2, 7, 4]])
	c = np.array([[1], [1], [1], [1]])
	# c = np.array([1, 1, 1, 1])
	print(a)
	print(c)
	print(mat_solve(a, c))

