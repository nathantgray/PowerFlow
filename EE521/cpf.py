#  This file is for presentation of the pf_continuation method of the PowerSystem class.
from classes.power_system import PowerSystem
import numpy as np
from copy import deepcopy
from classes.crout_reorder import mat_solve


class CPF_demonstration(PowerSystem):
	def pf_continuation(self, watch_bus):
		print("\n~~~~~~~~~~ Start Voltage Stability Analysis ~~~~~~~~~~\n")
		σ = 0.1
		λ = 1
		psched = deepcopy(self.psched)
		qsched = deepcopy(self.qsched)
		kpq = np.r_[psched, qsched]
		y = self.y_bus
		n = np.shape(y)[0]
		pvpq = self.pvpq
		pq = deepcopy(self.pq)
		# ~~~~~~~ Run Conventional Power Flow on Base Case ~~~~~~~~~~
		v, d = self.flat_start()
		d = self.pf_dc(d, y, pvpq, psched, lam=λ)
		v, d, it = self.pf_newtonraphson(v, d, prec=3, maxit=10, qlim=False, lam=λ)
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		# ~~~~~ Set watched bus and associated indexes ~~~~~
		# watch_bus = 4
		watch_index = watch_bus - 1
		watch_pq_index = watch_index  # initialize
		for i, bus_type in enumerate(self.bus_data[:, self.busType]):
			if watch_index <= i:
				break
			if bus_type > 0 and watch_index > i:
				watch_pq_index -= 1
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		results = [[σ, v[watch_index], d[watch_index], λ, λ*self.psched[watch_index - 1], λ*self.qsched[watch_pq_index ]]]
		phase = 1  # phase 1 -> increasing load, phase 2 -> decreasing V, phase 3 -> decreasing load

		# Continuation Power Flow or Voltage Stability Analysis
		while True:

			# kpq_jon = np.zeros(kpq.shape)
			# kpq_jon[watch_index-1] = -1
			# Calculate Jacobian
			if phase == 1:
				kt = len(pvpq) + len(pq)
				tk = 1
				jac = self.cpf_jacobian(v, d, pq, kpq, kt, tk)
			if phase == 2:
				kt = len(pvpq) + watch_pq_index
				tk = -1
				jac = self.cpf_jacobian(v, d, pq, kpq, kt, tk)
			if phase == 3:
				kt = len(pvpq) + len(pq)
				tk = -1
				jac = self.cpf_jacobian(v, d, pq, kpq, kt, tk)

			# Calculate update values
			# ~~~~~~~~~~ Calculated Tangent Vector ~~~~~~~~~~
			t = mat_solve(jac, np.r_[np.zeros(jac.shape[0] - 1), 1])
			# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			# Update angles: d_(n+1) = d_n + dd
			d_pred = deepcopy(d)
			d_pred[pvpq] = d[pvpq] + σ * t[:n - 1]
			# Update Voltages: V_(n+1) = V_n(1+dV/V_n)
			v_pred = deepcopy(v)
			v_pred[pq] = v[pq] + σ * t[n - 1:-1]
			# Update Lambda
			λ_pred = λ + σ * t[-1]
			# ~~~~~~~~~~ Corrector ~~~~~~~~~~

			d_cor = deepcopy(d_pred)
			v_cor = deepcopy(v_pred)
			λ_cor = deepcopy(λ_pred)
			it = 0
			maxit = 7
			while it < maxit:
				mis, p_calc, q_calc = self.mismatch(v_cor, d_cor, y, pq, pvpq, λ_cor*psched, λ_cor*qsched)
				if phase == 1 or phase == 3:
					mis = np.r_[mis, λ_pred - λ_cor]
				if phase == 2:
					mis = np.r_[mis, v_pred[watch_index] - v_cor[watch_index]]
				# Check error
				if max(abs(mis)) < 10 ** -3:
					break  # return v, d, it
				jac = self.cpf_jacobian(v_cor, d_cor, pq, kpq, kt, tk)
				# Calculate update values
				dx = mat_solve(jac, mis)
				# Update angles: d_(n+1) = d_n + dd
				d_cor[pvpq] = d_cor[pvpq] + dx[:n - 1]
				# Update Voltages: V_(n+1) = V_n(1+dV/V_n)
				v_cor[pq] = v_cor[pq] + dx[n - 1:n + pq.size - 1]
				# Update Lambda
				λ_cor = λ_cor + dx[-1]
				it += 1

			if phase == 1:
				if it >= maxit:
					phase = 2
					σ = 0.025
					print('phase 2')
				else:
					v = deepcopy(v_cor)
					d = deepcopy(d_cor)
					λ = deepcopy(λ_cor)
					print(round(λ, 8), v[watch_index])
					results = np.r_[results, [[σ, v[watch_index], d[watch_index], λ, λ*self.psched[watch_index - 1], λ*self.qsched[watch_pq_index ]]]]

			elif phase == 2:
				if it >= maxit:
					print("phase 2 not converged")
					#break
					phase = 3
					σ = 0.1
					print('phase 3')
				elif results[-2, 3] - results[-1, 3] > 0.2:
					phase = 3
					σ = 0.1
					print('phase 3')
				else:
					v = deepcopy(v_cor)
					d = deepcopy(d_cor)
					λ = deepcopy(λ_cor)
					print(round(λ, 8), v[watch_index])
					results = np.r_[results, [[σ, v[watch_index], d[watch_index], λ, λ*self.psched[watch_index - 1], λ*self.qsched[watch_pq_index ]]]]

			if phase == 3:
				# break
				if λ < 1:
					break

				v = deepcopy(v_cor)
				d = deepcopy(d_cor)
				λ = deepcopy(λ_cor)
				print(round(λ, 8), v[watch_index])
				results = np.r_[results, [[σ, v[watch_index], d[watch_index], λ, λ*self.psched[watch_index - 1], λ*self.qsched[watch_pq_index ]]]]

		return results


if __name__ == "__main__":
	import matplotlib.pyplot as plt

	# case_name = "../CaseFiles/IEEE14BUS.txt"
	case_name = "../CaseFiles/IEEE14BUS_handout.txt"
	# case_name = "../CaseFiles/2BUS.txt"
	ps = CPF_demonstration(case_name, sparse=True)
	watch_bus = 14
	results = ps.pf_continuation(watch_bus)
	nose_point_index = np.argmax(results[:, 3])
	nose_point = results[nose_point_index, :]
	print(nose_point)
	plt.plot(results[:, 3], results[:, 1], '-o')
	plt.title('PV Curve for Modified IEEE 14-Bus System at Bus {}'.format(watch_bus))
	plt.xlabel('Lambda (schedule multiplication factor)')
	plt.ylabel('Bus Voltage (p.u.)')
	plt.show()