# Helper functions for data preprocessing.
# TODO: add type hinting

import os
import re
import time
import pickle
import xarray as xr
import torch
from torchvision.transforms.functional import crop
import posixpath
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# original data functions
def read_all_data(fldr_path, voltages, pressures):
    file_count = 0
    data_table = None
    for voltage in voltages:
        for pressure in pressures:
            file_name = '{0:d}Vpp_{1:03d}Pa_node.dat'.format(voltage,pressure)
            file_path = posixpath.join(fldr_path, file_name)
            if os.path.exists(file_path):
                data = read_file(file_path)
                file_count += 1
            else:
                continue
            
            data = attach_VP_columns(data, voltage, pressure)
            data_table = data if file_count==1 else pd.concat([data_table,data], ignore_index=True)
    return data_table


def read_file(file_path):
    with open(file_path, 'r') as f:
        data = []
        for n,line in enumerate(f,1):
            if n==1:
                line = line.strip()
                line = re.findall(r'"[^"]*"', line) # get ['"var_name1"', '"var_name2"', ...]
                column_labels = [var_name.replace('"','') for var_name in line]
            elif n==2:
                continue
            else:
                data_line = [eval(data) for data in line.split()]
                if len(data_line)==4:
                    break
                data.append(data_line)
    return pd.DataFrame(data, columns=column_labels)


def attach_VP_columns(data, voltage, pressure):
    num_data_points = len(data)
    vp_columns = [[voltage, pressure] for n in range(num_data_points)]
    vp_columns = pd.DataFrame(vp_columns, columns=['Vpp [V]', 'P [Pa]'])
    return vp_columns.join(data)


def create_output_dir(root):
    rslt_dir = root / 'created_models'
    if not os.path.exists(rslt_dir):
        os.mkdir(rslt_dir)
    
    date_str = datetime.today().strftime('%Y-%m-%d_%H%M')
    out_dir = rslt_dir / date_str
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print('directory', out_dir, 'has been created.\n')
    
    return out_dir


# more stuff from elsewhere
def data_preproc(data_table, scale_exp, lin=True):
    trgt_params = ('potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)')

    def get_param_exp(col_vals):
        ''' get exponent of the parameter's mean value for scaling. '''
        mean_exp = round(np.log10(col_vals.mean()), 0) - 1.0
        # return 0 if mean_exp is less than zero to avoid blowing up small values
        if mean_exp >=  0.0:
            return mean_exp
        else:
            return 0.0


    for col_n,(col_name,col_vals) in enumerate(data_table.iteritems(), start=1):
        if col_name in trgt_params:
            if lin:
                # get exponent for scaling
                exponent = get_param_exp(col_vals)
                scale_exp.append(exponent)            
                tmp_col = col_vals.values.reshape(-1,1)/(10**exponent)  # scale by dividing
            else:
                tmp_col = np.log10(col_vals.values.reshape(-1,1))
        else:
            tmp_col = col_vals.values.reshape(-1,1)
        proced_table = tmp_col if col_n==1 else np.hstack([proced_table,tmp_col])
    
    proced_table = pd.DataFrame(proced_table, columns=data_table.columns)
    proced_table = proced_table.replace([np.inf,-np.inf], np.nan)
    proced_table = proced_table.dropna(how='any')
    
    return proced_table


def scale_all(data_table, x_or_y, out_dir=None):
    data_table = data_table.copy()
    for n,column in enumerate(data_table.columns, start=1):
        scaler = MinMaxScaler()
        data_col = data_table[column].values.reshape(-1,1)
        scaler.fit(data_col)
        scaled_data = scaler.transform(data_col)
        scaled_data_table = scaled_data if n==1 else np.hstack([scaled_data_table,scaled_data])
        
        if out_dir is not None:
            pickle_file = out_dir / f'{x_or_y}scaler_{n:02d}.pkl'
            with open(pickle_file, mode='wb') as pf:
                pickle.dump(scaler, pf, protocol=4)
    
    scaled_data_table = pd.DataFrame(scaled_data_table, columns=data_table.columns)
    
    return scaled_data_table

def read_aug_data(file):
    """Read data file and return a DataFrame.

    Args:
        file (PosixPath): Path to .feather file.

    Returns:
        interp_df: DataFrame of interpolated data.
    """
    interp_df = pd.read_feather(file).drop(columns=['Ex (V/m)', 'Ey (V/m)'])
    return interp_df


##### data processing ######
def get_augmentation_data(data_used, root, xy: bool, vp: bool):
    """Get augmentation data for training.

    Args:
        xy (bool): Include xy grid augmented data
        vp (bool): Inclde vp augmentation data
    """
    if xy:  # xy augmentation
        xyfile = Path(root/'data'/'interpolation_datasets'/'rec-interpolation2.nc')
        xydf = xr.open_dataset(xyfile).to_dataframe().reset_index().dropna()
    else: xydf = None

    if vp:  # vp augmentation
        vpfolder = Path(root/'data'/'interpolation_feather'/'20221209')
        # read all files and combine into a single df
        vpdf = [read_aug_data(file) for file in vpfolder.glob('*.feather')]
        vpdf = pd.concat(vpdf).rename(columns={'Vpp [V]' : 'V', 
                                               'P [Pa]'  : 'P',
                                               'X'       : 'x',
                                               'Y'       : 'y'}, inplace=True)
    else: vpdf = None

    # make sure that the data follows the correct format before returning
    return pd.concat([data_used, xydf, vpdf], ignore_index=True)
    

def get_data(root, voltages, pressures, excluded, xy=False, vp=False):
    """Get dataset

    Assumes the DataFrame has previously been saved as a .feather file. If not,
    a new .feather file is created in the root/data folder.

    Args:
        xy (bool, optional): Include xy augmentation. Defaults to False.
        vp (bool, optional): Include vp augmentation. Defaults to False.

    Raises:
        Exception: Raises an error if no data is available in avg_data.

    Returns:
        data_used: DataFrame of data to be used, with specified augmentation data.
        data_excluded: DataFrame of excluded data.
    """
    avg_data_file = root/'data'/'avg_data.feather'
    data_fldr_path = root/'data'
    voltage_excluded, pressure_excluded = excluded

    # check if feather file exists and load avg_data
    if avg_data_file.is_file():
        print('reading data feather file...')
        avg_data = pd.read_feather(avg_data_file)
    
    else:
        print('data feather file not found, building file...')
        start_time = time.time()
        avg_data = read_all_data(data_fldr_path, voltages, pressures).drop(columns=['Ex (V/m)', 'Ey (V/m)'])
        elapsed_time = time.time() - start_time
        print(f' done ({elapsed_time:0.1f} sec).\n')

        if len(avg_data)==0:
            raise Exception('No data available.')

        avg_data.to_feather(avg_data_file)

    # separate data to be excluded (to later check the model)
    data_used     = avg_data[~((avg_data['Vpp [V]']==voltage_excluded) & (avg_data['P [Pa]']==pressure_excluded))].copy()
    data_excluded = avg_data[  (avg_data['Vpp [V]']==voltage_excluded) & (avg_data['P [Pa]']==pressure_excluded) ].copy()

    # rename columns
    data_used.rename(columns={'Vpp [V]' : 'V',
                              'P [Pa]'  : 'P',
                              'X'       : 'x', 
                              'Y'       : 'y'}, inplace=True)

    if (xy or vp):
        data_used = get_augmentation_data(data_used, root, xy, vp)

    # create new column of x^2 and y^2
    # data_used['x**2'] = data_used['x']**2
    # data_used['y**2'] = data_used['y']**2
    
    return data_used, data_excluded

##### misc #####
def yn(str):
    if str.lower() in ['y', 'yes', 'yea', 'ok', 'okay', 'k',  
                       'sure', 'hai', 'aye', 'ayt', 'fosho']:
        return True
    elif str.lower() in ['n', 'no', 'nope', 'nah', 'hold this l']:
        return False
    else:
        raise Exception(str + 'not recognized: use y - yes, n - no')
    

##### image datasets (autoencoder, gan, etc) #####
class ImageDataset:
    def __init__(self, data_dir: Path, square=False):
        self.data_dir = data_dir
        self.square = square
        self.v_excluded = 300  # ideally inferred from the nc's metadata
        self.p_excluded = 60
        self.scaler_dict = {}  # FIXME: this gets overwritten when creating both train and test
        self._train = None
        self._test = None

    @property
    def train(self):
        if self._train is not None:
            return self._train
        else:
            train_data = self.data_dir/'train_set.pt'

            if train_data.exists():  # consider using a try-except
                train = torch.load(train_data)
            else:
                train_ds = self.data_dir/'rec-interpolation2.nc'
                train = self._nc_to_np(train_ds, 'train')
            
            if self.square:
                train = crop(train, 350, 0, 200, 200)

            self._train = train
            return self._train


    @property
    def test(self):
        if self._test is not None:
            return self._test
        else:
            test_data = self.data_dir/'test_set.pt'
            
            if test_data.exists():
                test = torch.load(test_data)
            else:
                test_ds = self.data_dir/'test_set.nc'
                test = self._nc_to_np(test_ds, 'test')
            
            if self.square:
                test = crop(test, 350, 0, 200, 200)

            self._test = test
            return self._test


    def _scale_np(self, array: np.ndarray, var: str, scaler_dict: dict):
        max = np.nanmax(array)
        min = np.nanmin(array)

        scaler_dict[var] = (min, max)

        return (array - min) / (max - min)


    def _nc_to_np(self, ds: xr.Dataset, which='train'):
        """Create NumPy arrays from NetCDF dataset

        Creates arrays from the .nc files if the .pt files don't yet exist.

        Args:
            ds (xr.Dataset): Dataset of
            which (str, optional): _description_. Defaults to 'train'.

        Returns:
            _type_: _description_
        """
        variables = list(self.ds.data_vars)

        if which == 'test':
            
            for v in ds.V.values:
                for p in ds.P.values:
                    # extract the values from the dataset for all 5 variables
                    vp_data = np.nan_to_num(np.stack(
                        [self._scale_np(ds[var].sel(V=v, P=p).values, var, self.scaler_dict) for var in variables]))
                    features = np.array([v, p])  # not yet used

            # consider saving as .pt file after conversion
            labels = np.expand_dims(np.float32(vp_data), axis=0)
            assert labels.shape == (1, 5, 707, 200)  # samples, channels, height, width

            torch.save(labels, self.data_dir/'test_labels.pt')
            torch.save(features, self.data_dir/'test_features.pt')

            return [features, labels]
        
        elif which == 'train':
            data_list = []
            feature_list = []
            for v in ds.V.values:
                for p in ds.P.values:
                    # extract the values from the dataset for all 5 variables
                    vp_data = np.nan_to_num(np.stack(
                        [self._scale_np(ds[var].sel(V=v, P=p).values, var, self.scaler_dict) for var in variables]))
                    feature = np.array([v, p])
                    if (v == self.v_excluded) & (p == self.p_excluded):
                        pass  # this is a hole in the data set that contains only nans
                    else:
                        data_list.append(vp_data)
                        feature_list.append(feature)

            labels = np.float32(np.stack(data_list))
            features = np.float32(np.stack(data_list))  # not yet used
            # samples, channels, height, width
            assert labels.shape == (31, 5, 707, 200)

            torch.save(labels, self.data_dir/'train_labels.pt')
            torch.save(features, self.data_dir/'train_features.pt')

            return [features, labels]
        