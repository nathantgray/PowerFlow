import numpy as np


def pf_jacobian(v, d, y, pq):
	# This function was written by Nathan Gray using formulas from chapter 9 of
	# "Power Systems Analysis" J. Grainger et al.
	# Calculates the Jacobian Matrix for use in the Newton-Raphson Method.
	# Arguments:
	# PQ: List of PQ buses
	# Y: Ybus matrix
	# V: Voltage magnitudes
	# d: Voltage phase angles
	n = y.shape[0]
	s = (v * np.exp(1j * d)) * np.conj(np.dot(y, v * np.exp(1j * d)))
	p = s.real
	q = s.imag
	# J11
	row, col = np.where(y)
	j11 = np.zeros((n-1, n-1))
	for a in range(row.shape[0]):
		i = row[a]
		j = col[a]
		if i != 0 and j != 0:
			if i == j:
				j11[i-1, j-1] = q[i] - v[i]**2*y[i, i].imag
			else:
				j11[i-1, j-1] = -abs(v[i]*v[j]*y[i, j])*np.sin(np.angle(y[i, j]) + d[j] - d[i])

	# J21
	row, col = np.where(y)
	index = np.transpose(np.array([row, col]))
	index = index[np.in1d(index[:, 0], pq)]
	j21 = np.zeros((pq.size, n - 1))
	for a in range(index.shape[0]):
		i = index[a, 0]
		j = index[a, 1]
		k = np.where(pq == i)
		if i == j:
			j21[k, j-1] = p[i] - abs(v[i])**2*y[i, j].real
		else:
			j21[k, j-1] = -abs(v[i]*v[j]*y[i, j])*np.cos(np.angle(y[i, j]) + d[j] - d[i])

	# J12
	row, col = np.where(y)
	index = np.transpose(np.array([row, col]))
	index = index[np.in1d(index[:, 1], pq)]
	j12 = np.zeros((n - 1, pq.size))
	for a in range(index.shape[0]):
		i = index[a, 0]
		j = index[a, 1]
		l: int = np.where(pq == j)
		if i == j:
			j12[i - 1, l] = p[i] + abs(v[i]**2*y[i, j].real)
		else:
			j12[i - 1, l] = abs(v[j]*v[i]*y[i, j])*np.cos(np.angle(y[i, j]) + d[j] - d[i])
	# J22
	row, col = np.where(y)
	index = np.transpose(np.array([row, col]))
	index = index[np.in1d(index[:, 0], pq)]
	index = index[np.in1d(index[:, 1], pq)]
	j22 = np.zeros((pq.size, pq.size))
	for a in range(index.shape[0]):
		i = index[a, 0]
		j = index[a, 1]
		k: int = np.where(pq == i)
		l: int = np.where(pq == j)
		if i == j:
			j22[k, l] = -j11[i-1, j-1] - 2*abs(v[i])**2*y[i, j].imag
		else:
			j22[k, l] = j11[i-1, j-1]
	# Assemble jacobian
	jtop = np.concatenate((j11, j12), axis=1)
	jbottom = np.concatenate((j21, j22), axis=1)
	jacobian = np.concatenate((jtop, jbottom), axis=0)
	return jacobian, j11, j21, j12, j22
