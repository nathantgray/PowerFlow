import pandas as pd
from classes.power_system import PowerSystem
import numpy as np

"""
Consider Kundur two area system described in Example 12.6 of the book.

a) Model Bus 1 as a slack bus. Model Buses 2, 3 and 4 as PV buses and solve the power-flow using Newton-Raphson and Fast
	Decoupled algorithms. Verify your solutions with the power-flow results given in Example 12.6.

b) Assume there are three lines connecting buses 7 and 8 and two lines connecting buses 8 and 9. Assume the capacities 
	of the shunt capacitors have been increased to be 400 MVar each at buses 7 and 9. Resolve part a) using 
	Newton-Raphson and Fast decoupled algorithms.

c) For part b), model Buses 2, 3 and 4 as PQ buses assuming the respective injections to match the solutions of part b).
	Resolve the power-flow using Newton-Raphson and Fast Decoupled algorithms.
"""

cases = [
	("KundurEx12-6.txt", "EE523HW1_result_a.csv"),
	("KundurEx12-6_hw1b.txt", "EE523HW1_result_b.csv"),
	("KundurEx12-6_hw1c.txt", "EE523HW1_result_c.csv"),
	("Kundur_modified.txt", "EE523_result.csv")]
for case_name, results_file in cases:
	ps = PowerSystem(case_name, sparse=False)
	v0, d0 = ps.flat_start()
	# d_dc = ps.pf_dc(d0, ps.y_bus, ps.pvpq, ps.psched)
	d0 = np.array([0, -0.1631, -0.3319, -0.5076, -0.1099, -0.2775, -0.4128, -0.5526, -0.7645, -0.6233, -0.4467])
	v_nr, d_nr, it_nr = ps.pf_newtonraphson(v0, d0, prec=7, maxit=8, qlim=False, debug_file="NRdebug"+results_file)
	v_fd, d_fd, it_fd = ps.pf_fast_decoupled(v0, d0, prec=3, maxit=30, qlim=False, debug_file="FDdebug"+results_file)
	s_nr = (v_nr * np.exp(1j * d_nr)) * np.conj(ps.y_bus.dot(v_nr * np.exp(1j * d_nr)))
	s_fd = (v_fd * np.exp(1j * d_fd)) * np.conj(ps.y_bus.dot(v_fd * np.exp(1j * d_fd)))

	nrd = {
		'Bus': ps.bus_data[:, 0],
		"Type": ps.bus_data[:, 4],
		"V Result": v_nr,
		"Angle Result": d_nr*180/np.pi,
		"MW Injected": s_nr.real * ps.p_base,
		"MVAR Injected": s_nr.imag * ps.p_base,
		"Iterations": it_nr}
	fdd = {
		'Bus': ps.bus_data[:, 0],
		"Type": ps.bus_data[:, 4],
		"V Result": v_fd,
		"Angle Result": d_fd*180/np.pi,
		"MW Injected": s_fd.real * ps.p_base,
		"MVAR Injected": s_fd.imag * ps.p_base,
		"Iterations": it_fd}

	df_space = pd.DataFrame(data={"": [""]})
	df_nr = pd.DataFrame(data=nrd)
	df_fd = pd.DataFrame(data=fdd)
	df = pd.concat([df_nr, df_space, df_fd], axis=1)
	# with pd.ExcelWriter(results_file) as writer:
	# 	df_nr.to_excel(writer, sheet_name="Newton Raphson")
	# 	df_fd.to_excel(writer, sheet_name="Fast Decoupled")
	df.to_csv(results_file)
	print(df_nr)
	print(df_fd)