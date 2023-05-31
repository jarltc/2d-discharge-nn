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
    def __init__(self, data_dir: Path, is_square=False):
        self.data_dir = data_dir
        self.is_square = is_square
        self.v_excluded = None
        self.p_excluded = None
        self._train = None  # list of [features, labels]
        self._test = None  # list of [features, labels]
        self.v_used = None
        self.p_used = None

        if (data_dir/'scaler_dict.pkl').exists():
            with open(data_dir/'scaler_dict.pkl', 'rb') as f:
                self.scaler_dict = pickle.load(f)
        else:
            self.scaler_dict = {}

    @property
    def train(self) -> list[np.ndarray]:
        """Return train dataset (features, labels).

        Loads the dataset as the self._test property if not yet set.

        Returns:
            list[np.ndarray]: List containing features, i.e. 2d profiles and labels, i.e. (V, P) 
        """
        if self._train is not None:
            return self._train
        else:
            train_features = self.data_dir/'train_features.pt'
            train_labels = self.data_dir/'train_labels.pt'

            if train_features.exists() & train_labels.exists():  # consider using a try-except
                train = [torch.load(train_features), torch.load(train_labels)]
            else:
                train_ds = xr.open_dataset(self.data_dir/'rec-interpolation2.nc')
                train = self._nc_to_np(train_ds, 'train')
                # TODO: does this create train_features and train_labels?

            self.v_used = {pair[0] for pair in train_labels} 
            self.p_used = {pair[1] for pair in train_labels}
            
            if self.is_square:
                train[0] = crop(torch.tensor(train[0]), 350, 0, 200, 200).numpy()  # TODO: use opencv for cropping

            self._train = train
            return self._train


    @property
    def test(self) -> list[np.ndarray]:
        """Return test dataset (features, labels).

        Loads the dataset as the self._test property if not yet set.
        
        Returns:
            list[np.ndarray]: List containing features, i.e. 2d profiles and labels, i.e. (V, P)
        """
        if self._test is not None:
            return self._test
        else:
            test_features = self.data_dir/'test_features.pt'
            test_labels = self.data_dir/'test_labels.pt'
            
            if test_features.exists() & test_labels.exists():
                test = [torch.load(test_features), torch.load(test_labels)]
            else:
                test_ds = xr.open_dataset(self.data_dir/'test_set.nc')
                test = self._nc_to_np(test_ds, 'test')
            
            self.v_excluded = {pair[0] for pair in test_labels} 
            self.p_excluded = {pair[1] for pair in test_labels}

            if self.is_square:
                test[0] = crop(torch.tensor(test[0]), 350, 0, 200, 200).numpy()  # crop features only

            self._test = test
            return self._test


    def _scale_np(self, array: np.ndarray, var: str, scaler_dict: dict):
        """Apply scaling on np arrays

        Saves the scaler dict containing (min, max) for each variable.
        Args:
            array (np.ndarray): NumPy array containing a variable's data.
            var (str): String of the variable's name. Ex: "potential (V)" 
                I don't remember if this includes the units.
            scaler_dict (dict): Dict containing a tuple of (min, max) for each variable.

        Returns:
            np.ndarray: Array of minmax-scaled values for the variable.
        """
        if scaler_dict == {}:
            max = np.nanmax(array)
            min = np.nanmin(array)
            scaler_dict[var] = (min, max)
        else:
            try:  # hmm
               min, max = scaler_dict[var]
            except:
               max = np.nanmax(array)
               min = np.nanmin(array)
               scaler_dict[var] = (min, max)
           
        return (array - min) / (max - min)


    def _nc_to_np(self, ds: xr.Dataset, which='train') -> list[np.ndarray]:
        """Create NumPy arrays from NetCDF dataset

        Creates arrays from the .nc files if the .pt files don't yet exist, and 
        applies minmax scaling to return a pair of features and labels.

        Args:
            ds (xr.Dataset): NetCDF dataset containing images.
            which (str, optional): _description_. Defaults to 'train'.

        Returns:
            list[np.ndarray]: List containing features, i.e. 2d profiles and labels, i.e. (V, P)
        """
        variables = list(ds.data_vars)

        if which == 'test':
            
            for v in ds.V.values:
                for p in ds.P.values:
                    # extract the values from the dataset for all 5 variables
                    vp_data = np.nan_to_num(np.stack(
                        [self._scale_np(ds[var].sel(V=v, P=p).values, var, self.scaler_dict) for var in variables]))
                    labels = np.array([v, p])  # not yet used

            # consider saving as .pt file after conversion
            features = np.expand_dims(np.float32(vp_data), axis=0)
            assert features.shape == (1, 5, 707, 200)  # samples, channels, height, width

            with open(self.data_dir/'scaler_dict.pkl', 'wb') as f:
                pickle.dump(self.scaler_dict, f)

            torch.save(labels, self.data_dir/'test_labels.pt')
            torch.save(features, self.data_dir/'test_features.pt')

            return [features, labels]
        
        elif which == 'train':
            data_list = []
            label_list = []
            for v in ds.V.values:
                for p in ds.P.values:
                    # extract the values from the dataset for all 5 variables
                    vp_data = np.nan_to_num(np.stack(
                        [self._scale_np(ds[var].sel(V=v, P=p).values, var, self.scaler_dict) for var in variables]))
                    label = np.array([v, p])
                    if (v == self.v_excluded) & (p == self.p_excluded):
                        pass  # this is a hole in the data set that contains only nans
                    else:
                        data_list.append(vp_data)
                        label_list.append(label)

            features = np.float32(np.stack(data_list))
            labels = np.float32(np.stack(label_list))  # not yet used
            # samples, channels, height, width
            assert features.shape == (31, 5, 707, 200)

            with open(self.data_dir/'scaler_dict.pkl', 'wb') as f:
                pickle.dump(self.scaler_dict, f)

            torch.save(labels, self.data_dir/'train_labels.pt')
            torch.save(features, self.data_dir/'train_features.pt')

            return [features, labels]
        