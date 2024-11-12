#!/usr/bin/env python

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import datetime
import json
import pandas as pd
import xarray as xr

from cyclopts import App, Parameter, validators
from pathlib import Path

from zarr.convenience import consolidate_metadata
from zarr.util import NumberEncoder

from rich.console import Console
from rich import pretty

from typing import Annotated, Dict, List, Optional, Union

pretty.install()
con = Console(record=True)

app = App(default_parameter=Parameter(negative=()))


@app.default()
def extend_time(dst_zarr: Annotated[Path, Parameter(validator=validators.Path(exists=True))],
                end_date: Union[datetime.datetime, datetime.date, str],
                freq: Optional[str] = '1h'):
    """Extend the time dimension in an existing zarr dataset

    :param dst_zarr: zarr store to extend time in
    :param end_date: new ending date for zarr
    :param freq: frequency to use for timesteps
    """

    dst_filename = f'{dst_zarr}/.zmetadata'

    con.print(f'Zarr store: {dst_zarr}')
    con.print(f'New end date: {end_date}')
    con.print('-'*40)

    # Read the consolidated metadata
    with open(dst_filename, 'r') as in_hdl:
        data = json.load(in_hdl)

    # Open the target zarr dataset
    con.print('  reading zarr store')
    ds = xr.open_dataset(dst_zarr, engine='zarr',
                         backend_kwargs=dict(consolidated=True), chunks={})

    # Define the new time range
    # Date range should always start from the original starting date in the zarr dataset
    dates = pd.date_range(start=ds.time[0].values, end=end_date, freq=freq, unit=ds.time.encoding['units'])

    con.print('  reading metadata')
    # Get the index for time dimension of each variable from the consolidated metadata
    time_index = {}

    for kk, vv in data['metadata'].items():
        if kk in ['.zattrs', '.zgroup']:
            continue

        varname, metatype = kk.split('/')

        if metatype == '.zattrs':
            try:
                time_index[varname] = vv['_ARRAY_DIMENSIONS'].index('time')
            except ValueError:
                # Time dimension not used for this variable
                pass

            # con.print(f'{kk} {vv["_ARRAY_DIMENSIONS"]}')

    # Index for the time dimension for each variable
    # con.print(time_index)

    # Change the size of the time dimension in the unconsolidated metadata for each variable
    # This will overwrite the original .zarray file for each variable
    con.print('  updating metadata')
    for kk, vv in time_index.items():
        cfilename = f'{dst_zarr}/{kk}/.zarray'

        with open(cfilename, 'r') as in_hdl:
            uncol_meta = json.load(in_hdl)

        # Update the shape of the variable
        uncol_meta['shape'][vv] = len(dates)

        # con.print('-'*10, kk, '-'*10)
        # con.print(uncol_meta)

        # Write the updated metadata file
        with open(cfilename, 'w') as out_hdl:
            json.dump(uncol_meta, out_hdl, indent=4, sort_keys=True, ensure_ascii=True,
                      separators=(',', ': '), cls=NumberEncoder)

    ds.close()

    con.print('  consolidating metadata')
    # Re-open the zarr datastore using the unconsolidated metadata
    ds = xr.open_dataset(dst_zarr, engine='zarr',
                         backend_kwargs=dict(consolidated=False), chunks={},
                         decode_times=False)

    # Write a new consolidated metadata file
    consolidate_metadata(store=dst_zarr, metadata_key='.zmetadata')

    # Write the new time values
    con.print('  updating time variable in zarr store')
    save_enc = ds['time'].encoding

    ds.coords['time'] = dates
    ds['time'].encoding.update(save_enc)

    ds[['time']].to_zarr(dst_zarr, mode='a')


def main():
    app()


if __name__ == '__main__':
    main()
