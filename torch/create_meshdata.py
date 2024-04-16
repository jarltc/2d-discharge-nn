import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm
from itertools import product

if __name__ == "__main__":
    root = Path.cwd()
    source = root/'data'/'avg_data.feather'
    if not source.exists():
        raise ValueError(f'{source} not found!')
    
    out_dir = root/'data'/'mesh_datasets'
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    voltages = [200, 300, 400, 500] # V
    pressures = [  5,  10,  30,  45, 60, 80, 100, 120] # Pa

    vps = list(product(voltages, pressures))

    # load feather file
    feather = pd.read_feather(source)

    for vp in vps:
        v, p = vp
        filestem = f"{v}_{p}"
        subdf = feather.loc[(feather['Vpp [V]'] == v) & (feather['P [Pa]'] == p)]
        subdf.to_feather(out_dir/f'{filestem}.feather')
        
    # write metadata
    info = {'voltages': voltages,
            'pressures': pressures,
            'description': "Feather datasets for each pair of V and P. File names follow the format V_P.feather"}
    
    with open(out_dir/'input.yml', 'w') as yml_file:
        yaml.dump(info, yml_file)