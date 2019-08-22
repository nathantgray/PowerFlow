import pandas as pd
filename = 'IEEE14BUS.txt'
df = pd.read_fwf(filename)
for line in df:
	print(line)