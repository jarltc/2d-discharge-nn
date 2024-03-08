# code to plot a single variable, made this to cram a figure for the mext research proposal

import torch
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from image_data_helpers import get_data
from plot import draw_apparatus

root = Path.cwd()
nc_data1 = root/'data'/'interpolation_datasets'/'full_interpolation.nc'
ds = xr.open_dataset(nc_data1)

image = ds['Nm (#/m^-3)'].sel(V=300, P=60).values
print(image.shape)

#### mesh figure ###
avg_data= root/'data'/'avg_data.feather'
mesh_df = pd.read_feather(avg_data)

dataset = mesh_df.loc[(mesh_df['Vpp [V]'] == 300) & (mesh_df['P [Pa]'] == 60)]

nm_max = dataset['Nm (#/m^-3)'].max()
nm_min = dataset['Nm (#/m^-3)'].min()

fig, ax = plt.subplots(dpi=300, figsize=(12, 6))
sc = ax.scatter(dataset['X']*100, dataset['Y']*100, c=dataset['Nm (#/m^-3)'], cmap='viridis', s=0.2)
ax.set_aspect('equal')
ax.set_title('$n_m$ [$m^{-3}$]')
ax.set_ylabel('z [cm]')
ax.set_xlabel('r [cm]')
ax.set_ylim(0, 70.9)
ax.set_xlim(0, 20)
draw_apparatus(ax)
plt.colorbar(sc, aspect=30, pad=0.02)
fig.savefig('nm_mesh.png', bbox_inches='tight')
