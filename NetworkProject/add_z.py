import mpu
import numpy as np
import pandas as pd

node_df = pd.read_csv('ScenarioFiles/node-file-case1.csv', delimiter=',')
edge_df = pd.read_csv('ScenarioFiles/edge-file-case1.csv', delimiter=',')
d = np.array([])
for k in range(len(edge_df['from_node'])):
	from_node = edge_df['from_node'].iloc[k]
	to_node = edge_df['to_node'].iloc[k]
	f_lat = node_df['lat'].loc[node_df['name'] == from_node].values[0]
	f_long = node_df['long'].loc[node_df['name'] == from_node].values[0]
	t_lat = node_df['lat'].loc[node_df['name'] == to_node].values[0]
	t_long = node_df['long'].loc[node_df['name'] == to_node].values[0]
	d = np.r_[d, mpu.haversine_distance((f_lat, f_long), (t_lat, t_long))]  # distance in km
edge_df = edge_df.assign(distance=d)
r_per_mile = 0.693*1/0.693
x_per_mile = 0.462*1/0.693
mile_per_km = 0.6213712
r_per_km = r_per_mile*mile_per_km
x_per_km = x_per_mile*mile_per_km
new_r = d*r_per_km
new_x = d*x_per_km
z_base = 518.4
r_pu = new_r/z_base
r_pu[r_pu<0.000001] = 1e-6
x_pu = new_x/z_base
x_pu[x_pu<0.000001] = 1e-6
edge_df = edge_df.assign(r=r_pu, x=x_pu)
edge_df.to_csv('ScenarioFiles/edge-file-case1.csv', sep=',', index=False)