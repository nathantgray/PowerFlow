import numpy as np
from classes.sparse import Sparse as sp
from copy import deepcopy


def mat_solve(mat, b, order=None):
	# Solve the matrix equation Ax=b for x, where A is input argument, mat.
	# returns the vector x
	return lu_solve(sparse_crout(mat, order=order), b, order=order)


def sparse_crout(mat, order=None):
	# Crout
	# Performs Crout's LU decomposition and stores it in Q = L + U - I
	n = mat.shape[0]
	if order is None:
		o = np.array(range(n))
	else:
		o = order
	q = sp.zeros(mat.shape)
	# q = mat
	for j in range(n):  # For each column, j:
		for k in range(j, n):  # Fill in jth column of the L matrix starting at diagonal going down.
			q[k, j] = mat[o[k], o[j]] - sum([q[k, i] * q[i, j] for i in range(j)])
		for k in range(j+1, n):  # Fill in the jth row of the U matrix starting after the diagonal going right.
			q[j, k] = 1/q[j, j]*(mat[o[j], o[k]] - sum([q[j, i] * q[i, k] for i in range(j)]))
	return q


def lu_solve(q, b, order=None):
	# len of b must match number of cols in q
	qn = q.shape[1]
	bn = len(b)
	if qn != bn:
		raise Exception("Dimensions do not match.")
	# Forward-Backwards
	# Solves the matrix equation, L*U*x = b for x.
	# L and U are stored in the matrix q = L + U - I
	n = q.shape[0]
	if order is None:
		o = np.array(range(n))
	else:
		o = order
	y = np.zeros(q.shape[0])
	x = np.zeros(q.shape[0])
	for i in range(n):  # Forwards
		y[o[i]] = 1/q[i, i]*(b[o[i]] - sum([q[i, j]*y[o[j]] for j in range(i)]))
	for i in range(n):  # Backwards
		i = n - i - 1
		x[o[i]] = y[o[i]] - (sum([q[i, j]*x[o[j]] for j in range(i, n)]))
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


def tinny1(sparse_mat):
	bmat = deepcopy(sparse_mat)
	bmat.values = bmat.values.astype(bool)  # use binary matrix to track location of connections and fills
	# 1. Calculate degree of each node.
	ndegs = node_degrees(bmat)
	# 2. Order nodes from least degree to highest.
	order = np.array([], dtype=int)
	fills = np.zeros(len(ndegs))
	for n in range(len(ndegs)):
		order_tmp = np.array([], dtype=int)
		for i in range(len(ndegs)):
			order_tmp = np.append(order_tmp, np.where(ndegs == i)[0])
			if len(order_tmp) > 0:
				break

		# remove first from order
		selected = order_tmp[0]
		ndegs[selected] = -1
		order = np.append(order, selected)
		# find connections
		connected = node_connections(bmat, selected)
		if len(connected) == 0:
			# this is the last node
			break
		bmat[selected, selected] = 0
		for node in connected:
			bmat[selected, node] = 0
			bmat[node, selected] = 0
		# reduce number of degrees to those nodes
		ndegs[connected] -= 1
		# add degrees based on fills, also record fills
		for i, node in enumerate(connected):
			for j in range(i+1, len(connected)):
				if bmat.ij_to_k(node, connected[j]) == -1:
					bmat[node, connected[j]] = 1
					bmat[connected[j], node] = 1
					fills[selected] += 2
					ndegs[node] += 1
					ndegs[connected[j]] += 1
	# check that each node that was connected to that node can connect to each other
	# each new connection is a fill
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


def node_connections(sparse_mat, node):
	n = sparse_mat.shape[0]
	connections = np.array([], dtype=int)
	k = sparse_mat.fic[node]
	while k > -1:
		if node != sparse_mat.rows[k]:
			connections = np.append(connections, sparse_mat.rows[k])
		k = sparse_mat.nic[k]
	return connections


if __name__ == "__main__":
	import time
	v = np.ones((44,))
	v = np.array([4, 1, 1, 1, 1, 4, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 6, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 4, 1, 2, 1, 1, 1, 4])
	r = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 9, 9])
	c = np.array([0, 1, 3, 7, 0, 1, 6, 9, 2, 3, 4, 6, 7, 9, 0, 2, 3, 4, 2, 3, 4, 5, 6, 9, 4, 5, 6, 1, 2, 4, 5, 6, 7, 8, 0, 2, 6, 7, 6, 8, 1, 2, 4, 9])
	a = sp(r, c, v)
	b = np.array(range(a.shape[1]))

	# Test 1: no ordering
	print("\nStart test-1 with no ordering---------------------------")
	time_start = time.perf_counter()
	q = sparse_crout(a)
	print("crout time = ", time.perf_counter() - time_start)
	print(q.full(dtype=bool).astype(int))
	time_start = time.perf_counter()
	x = lu_solve(q, b)
	print("solve time = ", time.perf_counter() - time_start)
	print("x=\n", x)
	print("b true  =\n", b)
	print("b check =\n", a.dot(x))
	print("alpha=", q.alpha())
	print("beta=", q.beta())
	print("alpha + beta = ", q.alpha() + q.beta())

	# Test 2: Tinny-0 ordering
	print("\nStart test-2 with Tinny-0 ordering---------------------------")
	order0 = tinny0(a)
	print("order 0: ", order0)
	time_start = time.perf_counter()
	q = sparse_crout(a, order=tinny0(a))
	print("crout time = ", time.perf_counter() - time_start)
	print(q.full(dtype=bool).astype(int))
	time_start = time.perf_counter()
	x = lu_solve(q, b, order=order0)
	print("solve time = ", time.perf_counter() - time_start)
	print("x=\n", x)
	print("b true  =\n", b)
	print("b check =\n", a.dot(x))
	print("alpha=", q.alpha())
	print("beta=", q.beta())
	print("alpha + beta = ", q.alpha() + q.beta())
	print("degrees: ", node_degrees(a))
	ndegs = node_degrees(a)
	print("order: ", order0)

	# Test 3: Tinny-1 ordering
	print("\nStart test-3 with Tinny-1 ordering---------------------------")
	order1 = tinny1(a)
	print("order 1: ", order1)
	time_start = time.perf_counter()
	q = sparse_crout(a, order=tinny1(a))
	print("crout time = ", time.perf_counter() - time_start)
	print(q.full(dtype=bool).astype(int))
	time_start = time.perf_counter()
	x = lu_solve(q, b, order=order1)
	print("solve time = ", time.perf_counter() - time_start)
	print("x=\n", x)
	print("b true  =\n", b)
	print("b check =\n", a.dot(x))
	print("alpha=", q.alpha())
	print("beta=", q.beta())
	print("alpha + beta = ", q.alpha() + q.beta())
	print("degrees: ", node_degrees(a))
	ndegs = node_degrees(a)
	print("order: ", order1)
