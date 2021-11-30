import pandas as pd
from classes.power_system import PowerSystem
import numpy as np
results_file = "./results/Results.xlsx"
case_name = "./CaseFiles/IEEE14BUS.txt"
# case_name = "IEEE14BUS_handout.txt"
ps = PowerSystem(case_name, sparse=False)
v0, d0 = ps.flat_start()
v_nr, d_nr, it_nr = ps.pf_newtonraphson(v0, d0, prec=2, maxit=10, qlim=False, qlim_prec=2, debug_file="14bus_NRdebug.csv")
v_fd, d_fd, it_fd = ps.pf_fast_decoupled(v0, d0, prec=2, maxit=100, qlim=False, qlim_prec=2, debug_file="14_bus_FDdebug.csv")
s_nr = (v_nr * np.exp(1j * d_nr)) * np.conj(ps.y_bus.dot(v_nr * np.exp(1j * d_nr)))
s_fd = (v_fd * np.exp(1j * d_fd)) * np.conj(ps.y_bus.dot(v_fd * np.exp(1j * d_fd)))
nrd = {'Bus': ps.bus_data[:, 0],
	  "Type": ps.bus_data[:, 4],
	  "V Result": v_nr,
	  "Angle Result": d_nr*180/np.pi,
	  "MW Injected": s_nr.real * ps.p_base,
	  "MVAR Injected": s_nr.imag * ps.p_base}
fdd = {'Bus': ps.bus_data[:, 0],
	  "Type": ps.bus_data[:, 4],
	  "V Result": v_fd,
	  "Angle Result": d_fd*180/np.pi,
	  "MW Injected": s_fd.real * ps.p_base,
	  "MVAR Injected": s_fd.imag * ps.p_base}

df_nr = pd.DataFrame(data=nrd)
df_fd = pd.DataFrame(data=fdd)
with pd.ExcelWriter(results_file) as writer:
	df_nr.to_excel(writer, sheet_name="Newton Raphson")
	df_fd.to_excel(writer, sheet_name="Fast Decoupled")
print(df_nr)
print(df_fd)
