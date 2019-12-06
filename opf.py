import numpy as np
from copy import deepcopy
from power_system import PowerSystem
from numpy.linalg import inv


if __name__ == "__main__":
	# 1. Solve ED without PF for P_total = 259 MW
	# 2. From base case PF, used P_total = P1 + P2
	# 3. Solve OPF with u = P2
	# 4 Solve OPF with P12 <= 5MW using penalty function
	# (use p.u. with base of 100MW)
	a1 = [8, 6.4]  # F1(P1) = a2[1]*P1**2 + a1[1]*P1,   F2(P2) = a2[2]*P2**2 + a1[2]*P2
	a2 = [0.004, 0.0048]



	# 1.
