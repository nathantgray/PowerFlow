import numpy as np
from sparse import Sparse as sp

def sparse_crout(mat, order=None):
	# Solve the matrix equation Ax=b for x, where A is input argument, mat.
	# returns the vector x

	# Crout
	# Performs Crout's LU decomposition and stores it in Q = L + U - I
	n = mat.shape[0]
	if order is None:
		o = np.array(range(n))
	else:
		o = order
	q = sp.empty(mat.shape)
	# q = mat
	for j in range(n):  # For each column, j:
		for k in range(j, n):  # Fill in jth column of the L matrix starting at diagonal going down.
			q[k, j] = mat[o[k], o[j]] - sum([q[k, i] * q[i, j] for i in range(j)])
		for k in range(j+1, n):  # Fill in the jth row of the U matrix starting after the diagonal going right.
			q[j, k] = 1/q[j, j]*(mat[o[j], o[k]] - sum([q[j, i] * q[i, k] for i in range(j)]))
	return q

def lu_solve(q, b):
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


def nz(sparse_mat):
	return len(sparse_mat.values)


def tinny0(sparse_mat):
	# 1. Calculate degree of each node.
	ndegs = node_degrees(sparse_mat)
	# 2. Order nodes from least degree to highest.
	order = np.array([], dtype=int)
	for i in range(len(ndegs)):
		order = np.append(order, np.where(ndegs == i)[0])
	return order



def node_degrees(sparse_mat):
	n = sparse_mat.shape[0]
	n_degs = np.zeros(n)
	for i in range(n):
		nz = 0
		k = sparse_mat.fic[i]
		while k > -1:
			nz += 1
			k = sparse_mat.nic[k]
		n_degs[i] = nz - 1
	return n_degs


def sparse_solve(mat, b):
	# Solve the matrix equation Ax=b for x, where A is input argument, mat.
	# returns the vector x
	return lu_solve(sparse_crout(mat), b)


if __name__ == "__main__":
	# v = np.array([1, -2, 2, 8, 1, 3, -2, -3, 2, 1, 2, -4, 2])
	# r = np.array([1,  1, 2, 2, 2, 3,  3,  4, 4, 5, 5,  5, 3]) - 1
	# c = np.array([1,  3, 1, 2, 4, 3,  5,  2, 3, 1, 2,  5, 4]) - 1
	v = np.ones((44,))
	v = np.array([4, 1, 1, 1, 1, 4, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 6, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 4, 1, 2, 1, 1, 1, 4])
	r = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 9, 9])
	c = np.array([0, 1, 3, 7, 0, 1, 6, 9, 2, 3, 4, 6, 7, 9, 0, 2, 3, 4, 2, 3, 4, 5, 6, 9, 4, 5, 6, 1, 2, 4, 5, 6, 7, 8, 0, 2, 6, 7, 6, 8, 1, 2, 4, 9])
	a = sp(r, c, v)
	# c = np.array([[1], [1], [1], [1]])
	# c = np.array([1, 1, 1, 1])
	a.alpha()
	print(a.full())
	# print(c)
	order = tinny0(a)
	q = sparse_crout(a, order=tinny0(a))
	print(q.full(dtype=bool).astype(int))
	# x = lu_solve(q, c)
	# print("x=\n", x)
	print("alpha=", q.alpha())
	print("beta=", q.beta())
	print("degrees: ", node_degrees(a))
	ndegs = node_degrees(a)
	print("order: ", order)

