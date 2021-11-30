import numpy as np
from copy import deepcopy


class Sparse:
	def __init__(self, i, j, v, shape=None, dtype=float):
		# TODO add optional arguments for setting fir, fic, nir, and nic to make calls faster
		self.rows = i
		self.cols = j
		self.values = v
		self.length = len(self.rows)
		if shape is None:
			if len(self.rows) == 0:
				i_size = int(0)
			else:
				i_size = int(max(self.rows) + 1)
			if len(self.cols) == 0:
				j_size = int(0)
			else:
				j_size = int(max(self.cols) + 1)
			self.shape = (i_size, j_size)
		else:
			self.shape = shape
		if len(self.values) > 0:
			self.dtype = type(self.values[0])
		else:
			self.dtype = dtype
		self.fir = np.ones(self.shape[0]).astype(int) * -1
		self.fic = np.ones(self.shape[1]).astype(int) * -1
		self.make_fir()
		self.make_fic()
		self.make_nir()
		self.make_nic()

	def __setitem__(self, ij, v):
		self.length = len(self.rows)
		i = ij[0]
		j = ij[1]
		if isinstance(i, tuple):
			i = i[0]
		if isinstance(j, tuple):
			j = j[0]
		if not hasattr(i, '__iter__'):
			i = [int(i)]
		if not hasattr(j, '__iter__'):
			j = [int(j)]
		if not hasattr(v, '__iter__'):
			v = [v]
		for m, val in enumerate(v):
			# existing_value, existing_k = self.__getitem__((i[m], j[m]), return_k=True)

			existing_k = self.ij_to_k(i[m], j[m])
			if existing_k != -1:
				if val == 0:
					self.del_item(existing_k)
				else:
					self.values[existing_k] = val
			else:
				if val == 0:
					break
				self.rows = np.r_[self.rows, i].astype(int)
				self.cols = np.r_[self.cols, j].astype(int)
				self.values = np.r_[self.values, v]
				# update fir and nir
				if i[m] < len(self.fir): # True if adding new row TODO possible error here
					if self.fir[i[m]] < 0:  # True if no existing values in row.
						self.fir[i[m]] = self.length + m  # set fir to k index of new value
						self.nir = np.r_[self.nir, - 1]  # set nir to -1 for new value
					else:  # Other values already in this row
						# check if first in row
						if j[m] < self.cols[self.fir[i[m]]]:  # If new value is first in the row
							# adjust nir and fir
							self.nir = np.r_[self.nir, self.fir[i[m]]]
							self.fir[i[m]] = self.length + m
						else:  # If new value is not first in the row
							# adjust nir
							# cycle through values in row starting with the first in row
							k_next = self.fir[i[m]]
							while j[m] > self.cols[k_next] and self.nir[k_next] > -1:
								# while indexed value is in a prior column and not last in row
								k_next = self.nir[k_next]  # point to next value in row
							self.nir = np.r_[self.nir, self.nir[k_next]]
							self.nir[k_next] = self.length + m  # TODO self.length is the wrong value. 49 but should be 146
				else:  # not adding a new row
					while len(self.fir) < i[m]:
						self.fir = np.r_[self.fir, -1]
					self.fir = np.r_[self.fir, self.length + m]
					self.nir = np.r_[self.nir, -1]


				# update fic and nic
				if j[m] < len(self.fic):  # True if adding new col
					if self.fic[j[m]] < 0:  # True if fic value is empty (-1)
						self.fic[j[m]] = self.length + m
						self.nic = np.r_[self.nic, - 1]
					else:
						# check if first in col
						if i[m] < self.rows[self.fic[j[m]]]:
							self.nic = np.r_[self.nic, self.fic[j[m]]]
							self.fic[j[m]] = self.length + m
						else:
							k_next = self.fic[j[m]]
							while i[m] > self.rows[k_next] and self.nic[k_next] > -1:
								k_next = self.nic[k_next]
							self.nic = np.r_[self.nic, self.nic[k_next]]
							self.nic[k_next] = self.length + m
				else:
					while len(self.fic) < j[m]:
						self.fic = np.r_[self.fic, -1]
					self.fic = np.r_[self.fic, self.length + m]
					self.nic = np.r_[self.nic, - 1]

		self.length = len(self.rows)
		if int(max(self.rows)) + 1 > self.shape[0]:
			shape0 = int(max(self.rows)) + 1
		else:
			shape0 = self.shape[0]
		if int(max(self.cols)) + 1 > self.shape[1]:
			shape1 = int(max(self.cols)) + 1
		else:
			shape1 = self.shape[1]
		self.shape = (shape0, shape1)

	def __getitem__(self, ij, return_k=False):
		# locate value by index
		i = ij[0]
		j = ij[1]
		if isinstance(i, tuple):
			i = i[0]
		if isinstance(j, tuple):
			j = j[0]
		try:
			if len(i) == 1:
				i = i[0]
		except:
			pass
		try:
			if len(j) == 1:
				j = j[0]
		except:
			pass


		if isinstance(i, (int, np.int32, np.int64)) and isinstance(j, (int, np.int32, np.int64)):
			k = self.ij_to_k(i, j)
			if k == -1:
				if return_k:
					return 0, k
				else:
					return 0
			else:
				if return_k:
					return self.values[k], k
				else:
					return self.values[k]

		# The remaining code runs if multiple indexes or slices are given. Returns a new sparse array.
		if isinstance(i, slice):
			i_start = i.start
			i_stop = i.stop
			i_step = i.step
			if i_start is None:
				i_start = 0
			if i_stop is None:
				i_stop = self.shape[0]
			if i_step is None:
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

	def del_item(self, k):
		if self.pir(k) == - 1:  # item is first in row...
			self.fir[self.rows[k]] = self.nir[k]  # the pointer in fir changes to point to next in row
		else:
			self.nir[self.pir(k)] = self.nir[k]  # The nir pointer for the previous changes to this nir
		if self.pic(k) == - 1:  # item is first in row...
			self.fic[self.cols[k]] = self.nic[k]  # the pointer in fic changes to point to next in col
		else:
			self.nic[self.pic(k)] = self.nic[k]  # The nic pointer for the previous changes to this nic

		# reduce indexes by one if they are larger than k
		for i, r in enumerate(self.fir):
			if r >= k:
				self.fir[i] -= 1
		for i, r in enumerate(self.fic):
			if r >= k:
				self.fic[i] -= 1
		for i, r in enumerate(self.nir):
			if r >= k:
				self.nir[i] -= 1
		for i, r in enumerate(self.nic):
			if r >= k:
				self.nic[i] -= 1

		# now delete the kth row from each vector
		self.values = np.delete(self.values, k)
		self.rows = np.delete(self.rows, k)
		self.cols = np.delete(self.cols, k)
		self.nir = np.delete(self.nir, k)
		self.nic = np.delete(self.nic, k)
		self.length = len(self.rows)

	def pir(self, k):  # Previous in row
		p = self.fir[self.rows[k]]
		if p == k:
			# first row
			return -1
		while self.nir[p] != k:  # loop through row
			p = self.nir[p]
		if self.nir[p] == k:
			return p

	def pic(self, k):  # Previous in col
		p = self.fic[self.cols[k]]
		if p == k:
			# first row
			return -1
		while self.nic[p] != k:  # loop through col
			p = self.nic[p]
		if self.nic[p] == k:
			return p

	def ij_to_k(self, i, j):
		if i > self.shape[0] - 1 or j > self.shape[1] - 1:  # check if i and j are in bounds
			return -1
		k = self.fir[i]  # Start with first value in requested row.
		if k == -1:
			return k
		else:
			while self.cols[k] != j:  # Loops until column matches j index.
				k = self.nir[k]
				if k == -1:  # end of row and value not found
					return k
				if self.rows[k] != i:
					print("Seaching for ", (i, j))
					print("------------Row should be ", i, " but is ", self.rows[k], ".")
			if self.rows[k] == i and self.cols[k] == j:
				return k

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
		# start with first column
		# find smallest row number with that column number
		# add the index of that number to the list
		for j in range(self.shape[1]):
			for k, col in enumerate(self.cols):
				if col == j:
					if self.rows[self.fic[j]] > self.rows[k] or self.fic[j] < 0:
						self.fic[j] = k

	def make_fir(self):
		for i in range(self.shape[0]):
			for k, row in enumerate(self.rows):
				if row == i:
					if self.cols[self.fir[i]] > self.cols[k] or self.fir[i] < 0:
						self.fir[i] = k

	def make_nir(self):
		# noinspection PyAttributeOutsideInit
		self.nir = np.ones(len(self.rows)).astype(int) * -1
		for i, k_first in enumerate(self.fir):
			k_prev = k_first
			while k_first > -1:
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
		# noinspection PyAttributeOutsideInit
		self.nic = np.ones(len(self.rows)).astype(int) * -1
		for j, k_first in enumerate(self.fic):
			k_prev = k_first
			while k_first > -1:
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

	def full(self, dtype=None):
		# convert to full matrix
		if dtype is not None:
			type = dtype
		else:
			type = self.dtype
		full_array = np.zeros(self.shape, dtype=type)
		for k, value in enumerate(self.values):
			full_array[int(self.rows[k]), int(self.cols[k])] = value
		return full_array
	full = property(full)

	def dot(self, vector):
		if len(vector) != self.shape[1]:
			print("Vector has size ", len(vector), " but matrix has ", self.shape[1], " columns!")
			return None
		else:
			if isinstance(sum(self.values)+sum(vector), (complex, np.complex128)):
				result = np.array(np.zeros(len(vector)), dtype=complex)
			else:
				result = np.array(np.zeros(len(vector)))
			for i in range(len(vector)):
				k = self.fir[i]
				while k >= 0:
					result[i] += self.values[k]*vector[self.cols[k]]
					k = self.nir[k]
			return result

	@staticmethod
	def where(sparse_mat):
		return (sparse_mat.rows, sparse_mat.cols)


	def __neg__(self):
		return self.return_new_object(self.rows, self.cols, -self.values)

	def imag(self):
		return self.return_new_object(self.rows, self.cols, np.imag(self.values))
	imag = property(imag)

	def real(self):
		return self.return_new_object(self.rows, self.cols, np.real(self.values))
	real = property(real)

	def alpha(self):
		# matrix must be square
		n = self.shape[0]
		m = self.shape[1]
		if n != m:
			raise Exception("Matrix must be square.")

		# Sum(#NZ in col[i] below qii + 1)(#NZ in row[i] right of qii)
		alpha = 0
		for i in range(n):

			# count nz in col[i] below qii
			nz_col = 0
			k = self.fic[i]
			while k > -1:
				if self.rows[k] > i:
					nz_col += 1
				k = self.nic[k]

			# count nz in row[i] right of qii
			nz_row = 0
			k = self.fir[i]
			while k > -1:
				if self.cols[k] > i:
					nz_row += 1
				k = self.nir[k]
			alpha += (nz_col + 1)*nz_row
		return alpha

	def beta(self):
		return len(self.values)


	@staticmethod
	def concatenate(mat_tuple, axis=0):
		mat1 = deepcopy(mat_tuple[0])
		mat2 = deepcopy(mat_tuple[1])
		length = len(mat1.rows)
		mat1.values = np.r_[mat1.values, mat2.values]
		if axis == 0:  # stack vertically
			mat1.rows = np.r_[mat1.rows, mat2.rows + mat1.shape[0]]
			mat1.cols = np.r_[mat1.cols, mat2.cols]
			# FIC
			for i, k in enumerate(mat1.fic):
				if k == -1:
					mat1.fic[i] = mat2.fic[i]
			# FIR
			fir_append = mat2.fir  # TODO does this need to be a deepcopy?
			for i, k in enumerate(mat2.fir):
				if k > -1:
					fir_append[i] = k + length
			mat1.fir = np.r_[mat1.fir, fir_append]
			# NIR
			nir_append = mat2.nir  # TODO does this need to be a deepcopy?
			for i, k in enumerate(mat2.nir):
				if k > -1:
					nir_append[i] = k + length
			mat1.nir = np.r_[mat1.nir, nir_append]
			# NIC
			mat1.shape = (mat1.shape[0] + mat2.shape[0], mat1.shape[1])
			mat1.make_nic()

		if axis == 1:  # stack horizontally
			mat1.rows = np.r_[mat1.rows, mat2.rows]
			mat1.cols = np.r_[mat1.cols, mat2.cols + mat1.shape[1]]
			# FIR
			for i, k in enumerate(mat1.fir):
				if k == -1:
					mat1.fir[i] = mat2.fir[i]
			# FIC
			fic_append = mat2.fic  # TODO does this need to be a deepcopy?
			for i, k in enumerate(mat2.fic):
				if k > -1:
					fic_append[i] = k + length
			mat1.fic = np.r_[mat1.fic, fic_append]
			# NIC
			nic_append = mat2.nic  # TODO does this need to be a deepcopy?
			for i, k in enumerate(mat2.nic):
				if k > -1:
					nic_append[i] = k + length
			mat1.nic = np.r_[mat1.nic, nic_append]
			# NIR
			mat1.shape = (mat1.shape[0], mat1.shape[1] + mat2.shape[1])
			mat1.make_nir()
		mat1.length = len(mat1.rows)
		return mat1

	@classmethod
	def zeros(cls, shape, dtype=float):
		return cls(np.array([]), np.array([]), np.array([]), shape=shape, dtype=dtype)


if __name__ == "__main__":
	# TEST CODE
	i_vec = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5]) - 1
	j_vec = np.array([1, 3, 1, 2, 4, 3, 5, 2, 3, 1, 2, 5]) - 1
	val = np.array([1, -2, 2, 8, 1, 3, -2, -3, 2, 1, 2, -4])
	array = np.array([[1, 0, -2, 0, 0], [2, 8, 0, 1, 0], [0, 0, 3, 0, -2], [0, -3, 2, 0, 0], [1, 2, 0, 0, -4]])
	a = Sparse(i_vec, j_vec, val)
	print(a.full())
	x = np.array([1, 2, 3, 4, 5])
	print("array dot x: ", array.dot(x))
	print("a dot x: ", a.dot(x))
	row, col = np.where(array)
	r = a.rows
	c = a.cols
	sliced = slice(1, 3, None)
	print(a[0, 3])
	a[[0, 3, 5, 3, 6], [3, 0, 3, 5, 6]] = (103, 130, 60, 61, 66)
	# a[:,:] = 1
	print(a[3, 0])
	print(a[5, 3])
	print(a[3, 5])
	print(a[6, 6])
	b = Sparse(np.array([]), np.array([]), np.array([]))
	b[0, 1] = 1
	b[1, 0] = 2
	print(b.full())
	#new_a = a[[0, 1, 2], 1]
	print(a.full())
	#print(new_a.full())
	mat1 = Sparse.zeros((3, 2))
	mat1[0, 0] = 1
	mat1[1, 1] = 2
	mat1[2, 0] = 3
	mat2 = Sparse.zeros((3, 2))
	mat2[0, 0] = 4
	mat2[1, 0] = 6
	mat2[1, 1] = 7
	print(mat1.full())
	print(mat2.full())
	newmat = Sparse.concatenate((mat1, mat2), axis=1)
	print(newmat.full())
	newmat = Sparse.concatenate((mat1, mat2), axis=0)
	print(newmat.full())
	k = a.ij_to_k(4, 0)
	print("k: ", k)
	p = a.pic(k)
	print("p: ", p)
	print('end')

