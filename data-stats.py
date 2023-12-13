""" Code to analyze the interpolation dataset.

This dataset is obtained by linear interpolation from an 
irregular mesh to a rectangular mesh."""

import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, LogLocator
import numpy as np

cat_rainbow = ['#d20f39', '#df8e1d', '#40a02b', '#04a5e5', '#8839ef']

def print_statistics(variable, file=None):
    """Create a dictionary of important stats for a single var.

    Args:
        variable (str): Variable to analyze. 
        file (Path, optional): Filepath to save statistics. Defaults to None.
    """
    stats_dict = dict(mean = ds[variable].mean().item(),
                      minimum = ds[variable].min().item(),
                      maximum = ds[variable].max().item(),
                      q1 = ds[variable].quantile(0.25).item(),
                      q2 = ds[variable].median().item(),
                      q3 = ds[variable].quantile(0.75).item(),
                      P99 = ds[variable].quantile(0.99).item()
    )
    print(f"***** {variable} *****", file=file)
    for key in stats_dict:
        print(f"{key}  = {stats_dict[key]:.2e}", file=file)
    print("\n", file=file)


def stats2file(file_path:Path):
    """Perform print_statistics for all variables and save.

    Args:
        file_path (Path): Path to output file.
    """
    with open(file_path/"stats.txt", 'w') as file:
        for variable in list(ds.keys()):
            print_statistics(variable, file)
    
    print(f"statistics saved to {file_path}.")


def data_histplot(ds:xr.Dataset):
    """Create a histogram for each variable in the dataset.

    Args:
        ds (xr.Dataset): Reference dataset containing training data.

    Returns:
        fig: Figure containing 5 histograms for each variable.
    """
    variables = list(ds.keys())
    columns_math = ['$\phi$', '$n_e$', '$n_i$', '$n_m$', '$T_e$']
    units = {'potential (V)'    :'($\mathrm{10 V}$)', 
            'Ne (#/m^-3)'      :'[$\mathrm{m^{-3}}$]',
            'Ar+ (#/m^-3)'     :'[$\mathrm{m^{-3}}$]', 
            'Nm (#/m^-3)'      :'[$\mathrm{m^{-3}}$]',
            'Te (eV)'          :'[eV]'}  # list of keys can be extracted using units.keys()

    fig, ax = plt.subplots(figsize=(4, 12), ncols=1, nrows=5, dpi=200)
    plt.subplots_adjust(hspace=0.4)

    for i, variable in enumerate(variables):
        data = np.nan_to_num(ds[variable].values.flatten())
        
        ax[i].set_axisbelow(True)
        ax[i].grid(linestyle='dashed')
        ax[i].hist(data, bins=100, color=cat_rainbow[i])  # range=[np.quantile(data, 0.1), np.quantile(data, 0.9)]

        # tick stuff
        ax[i].ticklabel_format(style='sci', scilimits=(0, 3), axis='both', useMathText=True)
        ax[i].set_yscale('log')
        locmajy = LogLocator(base=10) 
        locminy = LogLocator(base=10,subs=np.arange(2, 10) * .1,numticks=100)

        ax[i].yaxis.set_major_locator(locmajy)
        ax[i].yaxis.set_minor_locator(locminy)

        ax[i].set_xlim(left=0)
        ax[i].set_xlabel(f"{columns_math[i]} {units[variable]}")

        # annotate 99th percentile
        p99 = np.quantile(data, 0.99)
        ax[i].axvline(x=p99, linewidth=1.2, color='r')
        ax[i].text(p99, 0.7, '$P_{99}$', transform=ax[i].get_xaxis_text1_transform(0)[0], 
                   fontsize=10, ha='left', va='center', color='r')
    
    file_path = ds_path.parent/"hist.png"
    fig.savefig(file_path, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    root = Path.cwd()
    ds_path = root/"data"/"interpolation_datasets"/"rec-interpolation2.nc"

    ds = xr.open_dataset(ds_path)
    # stats2file(ds_path.parent)  # save stats in the same folder as ds
    data_histplot(ds)

