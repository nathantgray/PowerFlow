import numpy as np


class Sparse:
	def __init__(self, i, j, v):
		# TODO add optional arguments for setting fir, fic, nir, and nic to make calls faster
		self.rows = i
		self.cols = j
		self.values = v
		self.n = len(self.rows)
		if len(self.rows) == 0:
			i_size = int(0)
		else:
			i_size =int(max(self.rows) + 1)
		if len(self.cols) == 0:
			j_size = int(0)
		else:
			j_size = int(max(self.cols) + 1)
		self.shape = (i_size, j_size)
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
		if not hasattr(i, '__iter__'):
			i = [int(i)]
		if not hasattr(j, '__iter__'):
			j = [int(j)]
		if not hasattr(v, '__iter__'):
			v = [v]
		for m, val in enumerate(v):
			existing_value, existing_k = self.__getitem__((i[m], j[m]), return_k=True)
			if existing_value != 0:
				self.values[existing_k] = val
			else:
				self.rows = np.r_[self.rows, i].astype(int)
				self.cols = np.r_[self.cols, j].astype(int)
				self.values = np.r_[self.values, v]
				# update fir and nir
				if i[m] < len(self.fir): # True if adding new row
					if self.fir[i[m]] < 0:
						self.fir[i[m]] = self.n + m
						self.nir = np.r_[self.nir, - 1]
					else:
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
				else: # not adding a new row
					while len(self.fir) < i[m]:
						self.fir = np.r_[self.fir, -1]
					self.fir = np.r_[self.fir, self.n + m]
					self.nir = np.r_[self.nir, -1]


				# update fic and nic
				if j[m] < len(self.fic):  # True if adding new col
					if self.fic[j[m]] < 0:  # True if fic value is empty (-1)
						self.fic[j[m]] = self.n + m
						self.nic = np.r_[self.nic, - 1]
					else:
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
					while len(self.fic) < j[m]:
						self.fic = np.r_[self.fic, -1]
					self.fic = np.r_[self.fic, self.n + m]
					self.nic = np.r_[self.nic, - 1]

		self.n = len(self.rows)
		self.shape = (int(max(self.rows)) + 1, int(max(self.cols) + 1))

	def __getitem__(self, ij, return_k=False):
		# locate value by index
		i = ij[0]
		j = ij[1]

		if isinstance(i, (int, np.int32)) and isinstance(j, (int, np.int32)):
			if len(self.fic) - 1 < j:  # Is the index in bounds?
				print("column index, ", j, " out of bounds.")
				if return_k:
					return 0, -1
				else:
					return 0
			elif len(self.fir) - 1 < i:  # Is the index in bounds?
				print("Row index, ", i, " out of bounds.")
				if return_k:
					return 0, -1
				else:
					return 0
			else:  # The index is within bounds.
				k = self.fir[i]  # Start with first value in row.
				while self.cols[k] != j:  # Loops until column matches j index.
					k = self.nir[k]
					if k < 0: # end of row and value not found, return 0
						if return_k:
							return 0, -1
						else:
							return 0
				if self.rows[k] == i and self.cols[k] == j:
					if return_k:
						return self.values[k], k
					else:
						return self.values[k]
		# The remaining code runs if multiple indexes or slices are given. Returns a new sparse array.
		if isinstance(i, slice):
			i_start = i.start
			i_stop = i.stop
			i_step = i.step
			if i_start == None:
				i_start = 0
			if i_stop == None:
				i_stop = self.shape[0]
			if i_step == None:
				i_step = 1
			i = [index for index in range(i_start, i_stop, i_step)]
		if isinstance(j, slice):
			j_start = j.start
			j_stop = j.stop
			j_step = j.step
			if j_start is None:
				j_start = 0
			if j_stop is None:
				j_stop = self.shape[0]
			if j_step is None:
				j_step = 1
			j = [index for index in range(j_start, j_stop, j_step)]
		if not hasattr(i, '__iter__'):
			i_start = i
			i = [int(i)]
		if not hasattr(j, '__iter__'):
			j_stop = j
			j = [int(j)]
		_v = np.array([], dtype=int)
		_r = np.array([], dtype=int)
		_c = np.array([], dtype=int)
		for n_row, row in enumerate(i):
			n_col = 0
			k = self.fir[row]  # Start with first value in row.
			#while n_col < len(j):
			while k > -1 and n_col < len(j):
				while self.cols[k] != j[n_col] and k > -1:  # Loops until column matches j index or till end of row.
					if self.cols[k] > j[n_col]:
						# If next in row is farther right than next column asked for, a zero must exist in that column
						n_col = n_col + 1
					else:  # otherwise go to next in row
						k = self.nir[k]
				if self.rows[k] == row and self.cols[k] in j:  # If value found, add to new list
					_v = np.r_[_v, self.values[k]]
					_r = np.r_[_r, n_row]
					_c = np.r_[_c, n_col]
					k = self.nir[k]  # Go to next value in row.
					n_col = n_col + 1
				if k > -1 and n_col < len(j): # If not end of row and not end of j list
					if self.cols[k] > j[n_col]:
						# If next in row is farther right than next column asked for, a zero must exist in that column
						n_col = n_col + 1
		return self.return_new_object(_r, _c, _v)


	def __add__(self, other):
		try:
			return self.values + other.values
		except:
			try:
				return self.values + other
			except:
				print("NotImplemented")

	@classmethod
	def return_new_object(cls, i, j, v):
		return cls(i, j, v)

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
		full_array = np.zeros(self.shape, dtype=type(self.values[0]))
		for k, value in enumerate(self.values):
			full_array[self.rows[k], self.cols[k]] = value
		return full_array

	def dot(self, vector):
		if len(vector) != self.shape[1]:
			print("Vector has size ", len(vector), " but matrix has ", self.shape[1], " columns!")
			return None
		else:
			result = np.array(np.zeros(len(vector)), dtype=type(self.values[0]))
			for i in range(len(vector)):
				k = self.fir[i]
				while k >= 0:
					result[i] += self.values[k]*vector[self.cols[k]]
					k = self.nir[k]
			return result

	def __neg__(self):
		return self.return_new_object(self.rows, self.cols, -self.values)

	def imag(self):
		return self.return_new_object(self.rows, self.cols, np.imag(self.values))
	imag = property(imag)

	def real(self):
		return self.return_new_object(self.rows, self.cols, np.real(self.values))
	real = property(real)

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
	sliced = slice(1, 3, None)
	print(a[0, 3])
	a[(0, 3, 5, 3, 6), (3, 0, 3, 5, 6)] = (103, 130, 60, 61, 66)
	# a[:,:] = 1
	print(a[3, 0])
	print(a[5, 3])
	print(a[3, 5])
	print(a[6, 6])
	b = Sparse(np.array([]), np.array([]), np.array([]))
	b[0, 1] = 1
	b[1, 0] = 2
	print(b.full())
	new_a = a[[0, 1, 3, 4], 2]
	print(a.full())
	print(new_a.full())
	print('end')

