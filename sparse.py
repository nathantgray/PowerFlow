import numpy as np


class Sparse:
	def __init__(self, i, j, v):
		self.rows = i
		self.cols = j
		self.values = v
		self.n = len(self.rows)
		self.shape = (max(self.rows) + 1, max(self.cols) + 1)
		self.fir = np.ones(self.shape[0]).astype(int) * -1
		self.fic = np.ones(self.shape[1]).astype(int) * -1
		self.nir = np.ones(self.n).astype(int) * -1
		self.nic = np.ones(self.n).astype(int) * -1
		self.make_fir()
		self.make_fic()
		self.make_nir()
		self.make_nic()

	def __setitem__(self, ij, v):
		i = ij[0]
		j = ij[1]
		self.rows = np.r_[self.rows, i]
		self.cols = np.r_[self.cols, j]
		self.values = np.r_[self.values, v]
		if not hasattr(i, '__iter__'):
			i = [i]
		if not hasattr(j, '__iter__'):
			j = [j]
		if not hasattr(v, '__iter__'):
			v = [v]
		for m, val in enumerate(v):
			# update fir and nir
			if i[m] < self.shape[0]:
				# check if first in row
				if j[m] < self.cols[self.fir[i[m]]]:
					# adjust nir and fir
					self.nir = np.r_[self.nir, self.fir[i[m]]]
					self.fir[i[m]] = self.n + m
				else:
					# adjust nir
					k_next = self.fir[i[m]]
					while j[m] > self.cols[k_next] and self.nir[k_next] > -1:
						k_next = self.nir[k_next]
					self.nir = np.r_[self.nir, self.nir[k_next]]
					self.nir[k_next] = self.n + m
			else:
				self.fir = np.r_[self.fir, self.n + m]
				self.nir = np.r_[self.nir, -1]

			# update fic
			if max(j) < self.shape[1]:
				# check if first in col
				if i[m] < self.rows[self.fic[j[m]]]:
					self.nic = np.r_[self.nic, self.fic[j[m]]]
					self.fic[j[m]] = self.n + m
				else:
					k_next = self.fic[j[m]]
					while i[m] > self.rows[k_next] and self.nic[k_next] > -1:
						k_next = self.nic[k_next]
					self.nic = np.r_[self.nic, self.nic[k_next]]
					self.nic[k_next] = self.n + m
			else:
				self.fic = np.r_[self.fic, self.n + m]
				self.nic = np.r_[self.nic, -1]

		self.n = len(self.rows)
		self.shape = (max(self.rows) + 1, max(self.cols) + 1)

	def __getitem__(self, ij):
		# locate value by index
		i = ij[0]
		j = ij[1]
		try:
			m = self.fic[j]
		except:
			print("column index out of bounds")
			return 0
		else:
			try:
				k = self.fir[i]
			except:
				print("row index out of bounds")
				return 0
			else:
				while self.cols[k] != j:
					k = self.nir[k]
					if k < 0:
						return 0
				if self.rows[k] == i and self.cols[k] == j:
					return self.values[k]

	def __add__(self, other):
		try:
			return self.values + other.values
		except:
			try:
				return self.values + other
			except:
				print("NotImplemented")


	def make_fic(self):
		for j in range(self.shape[1]):
			for k, col in enumerate(self.cols):
				if col == j:
					if self.rows[self.fic[j]] > self.rows[k] or self.fic[j] < 0:
						self.fic[j] = k

	# start with first column
	# find smallest row number with that column number
	# add the index of that number to the list

	def make_fir(self):
		for i in range(self.shape[0]):
			for k, row in enumerate(self.rows):
				if row == i:
					if self.cols[self.fir[i]] > self.cols[k] or self.fir[i] < 0:
						self.fir[i] = k

	def make_nir(self):
		for i, k_first in enumerate(self.fir):
			k_prev = k_first
			while True:
				for k, col in enumerate(self.cols):
					if self.rows[k] == i and k != k_prev:
						if col > self.cols[k_prev]:
							if self.nir[k_prev] < 0:
								self.nir[k_prev] = k
							elif col < self.cols[self.nir[k_prev]]:
								self.nir[k_prev] = k
				k_prev = self.nir[k_prev]
				if k_prev < 0:
					break

	def make_nic(self):
		for j, k_first in enumerate(self.fic):
			k_prev = k_first
			while True:
				for k, row in enumerate(self.rows):
					if self.cols[k] == j and k != k_prev:
						if row > self.rows[k_prev]:
							if self.nic[k_prev] < 0:
								self.nic[k_prev] = k
							elif row < self.rows[self.nic[k_prev]]:
								self.nic[k_prev] = k
				k_prev = self.nic[k_prev]
				if k_prev < 0:
					break

	def full(self):
		# convert to full matrix
		full_array = np.zeros(self.shape)
		for k, value in enumerate(self.values):
			full_array[self.rows[k], self.cols[k]] = value
		return full_array

	def dot(self, vector):
		if len(vector) != self.shape[1]:
			print("Vector has size ", len(vector), " but matrix has ", self.shape[1], " columns!")
			return None
		else:
			result = np.array(np.zeros(len(vector))).astype(type(self.values[0]))
			for i in range(len(vector)):
				k = self.fir[i]
				while k >= 0:
					result[i] += self.values[k]*vector[self.cols[k]]
					k = self.nir[k]
			return result


if __name__ == "__main__":
	# TEST CODE
	i_vec = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5]) - 1
	j_vec = np.array([1, 3, 1, 2, 4, 3, 5, 2, 3, 1, 2, 5]) - 1
	val = np.array([1, -2, 2, 8, 1, 3, -2, -3, 2, 1, 2, -4])
	array = np.array([[1, 0, -2, 0, 0], [2, 8, 0, 1, 0],[0, 0, 3, 0, -2], [0, -3, 2, 0, 0], [1, 2, 0, 0, -4]])
	a = Sparse(i_vec, j_vec, val)
	print(a.full())
	x = np.array([1, 2, 3, 4, 5])
	print(array.dot(x))
	print(a.dot(x))
	row, col = np.where(array)
	r = a.rows
	c = a.cols

	print(a[0, 3])
	a[(0, 3, 5, 3, 6), (3, 0, 3, 5, 6)] = (103, 130, 60, 61, 66)
	print(a[0, 3])
	print(a[3, 0])
	print(a[5, 3])
	print(a[3, 5])
	print(a[6, 6])
	print('end')

