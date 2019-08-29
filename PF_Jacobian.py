import numpy as np


def pf_jacobian(v, d, y, pq):
	# This function was written by Nathan Gray using formulas from chapter 9 of
	# "Power Systems Analysis" J. Grainger et al.
	# Calculates the Jacobian Matrix for use in the Newton-Raphson Method.
	# Arguments:
	# v: Voltage magnitudes
	# d: Voltage phase angles
	# y: Ybus matrix
	# pq: List of PQ buses
	n = y.shape[0]
	# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)
	s = (v*np.exp(1j*d))*np.conj(y.dot(v*np.exp(1j*d)))
	p = s.real
	q = s.imag

	# Find indices of non-zero ybus entries
	row, col = np.where(y)

	# J11
	j11 = np.zeros((n-1, n-1))
	j12 = np.zeros((n - 1, pq.size))
	j21 = np.zeros((pq.size, n - 1))
	j22 = np.zeros((pq.size, pq.size))
	for a in range(row.shape[0]):
		i = row[a]
		j = col[a]

		# J11
		if i != 0 and j != 0:
			if i == j:  # Diagonals of J11
				j11[i-1, j-1] = - q[i] - v[i]**2*y[i, i].imag
			else:  # Off-diagonals of J11
				j11[i-1, j-1] = -abs(v[i]*v[j]*y[i, j])*np.sin(np.angle(y[i, j]) + d[j] - d[i])
		# J21
			if i in pq:
				k: int = np.where(pq == i)  # map bus index to jacobian index
				if i == j:  # Diagonals of J21
					j21[k, j-1] = p[i] - abs(v[i])**2*y[i, j].real
				else:  # Off-diagonals of J21
					j21[k, j-1] = -abs(v[i]*v[j]*y[i, j])*np.cos(np.angle(y[i, j]) + d[j] - d[i])
		# J12
			if j in pq:
				l: int = np.where(pq == j)  # map bus index to jacobian index
				if i == j:  # Diagonals of J12
					j12[i - 1, l] = p[i] + abs(v[i]**2*y[i, j].real)
				else:  # Off-diagonals of J12
					j12[i - 1, l] = abs(v[j]*v[i]*y[i, j])*np.cos(np.angle(y[i, j]) + d[j] - d[i])
		# J22
			if i in pq and j in pq:
				k: int = np.where(pq == i)  # map bus index to jacobian index
				l: int = np.where(pq == j)  # map bus index to jacobian index
				if i == j:  # Diagonal of J22
					j22[k, l] = -j11[i-1, j-1] - 2*abs(v[i])**2*y[i, j].imag
				else:  # Off-diagonals of J22
					j22[k, l] = j11[i-1, j-1]

	# Assemble jacobian
	jtop = np.concatenate((j11, j12), axis=1)
	jbottom = np.concatenate((j21, j22), axis=1)
	jacobian = np.concatenate((jtop, jbottom), axis=0)
	return jacobian
