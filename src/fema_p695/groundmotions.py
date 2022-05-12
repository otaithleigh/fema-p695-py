"""Ground motions in HDF5 format, with BLOSC compression.

File structure:

    records/
        <tabular data>
    groundmotions/
        <tabular data>
    _timeseries/
        <recordID>/<component>/Time
        <recordID>/<component>/RecordedAcceleration
        <recordID>/<component>/NormalizedAcceleration
"""
import importlib.resources
import typing as t
from functools import lru_cache

import h5py
import hdf5plugin  # Not used directly, but need to import to load libraries
import pandas as pd
import xarray as xr
from xarrayhdf import hdf_to_dataframe, hdf_to_dataset

__all__ = [
    'GroundMotionData',
    'load_ground_motions',
    'get_ground_motion',
    'get_metadata',
    'get_response_spectrum',
]


class GroundMotionData(t.NamedTuple):
    records: pd.DataFrame
    ground_motions: pd.DataFrame
    response_spectra: xr.Dataset


@lru_cache
def load_ground_motions(record_set):
    """Load all the data for a given record set into memory.

    Parameters
    ----------
    record_set : {'farfield', 'nearfield', 'nearfield_pulse', 'nearfield_no_pulse'}
        The ground motion set to load. Note that 'nearfield_pulse' and
        'nearfield_no_pulse' are just subsets of the 'nearfield' record set.
    """
    if record_set == 'nearfield':
        pulse = load_ground_motions('nearfield_pulse')
        nopulse = load_ground_motions('nearfield_no_pulse')
        return GroundMotionData(pd.concat([pulse[0], nopulse[0]]),
                                pd.concat([pulse[1], nopulse[1]]),
                                xr.merge([pulse[2], nopulse[2]]))

    with importlib.resources.path(__package__, record_set + '.hdf5') as p:
        records, ground_motions, spectra = _from_hdf5(p)

    return GroundMotionData(records, ground_motions, spectra)


def get_metadata(record_set, record: int):
    """Return the metadata for a specific record.

    Parameters
    ----------
    record_set : {'farfield', 'nearfield', 'nearfield_pulse', 'nearfield_no_pulse'}
        The ground motion set to load. Note that 'nearfield_pulse' and
        'nearfield_no_pulse' are just subsets of the 'nearfield' record set.
    record : int
        The record ID number. (Not the record sequence number.)
    """
    gmdata = load_ground_motions(record_set)
    meta = gmdata.records.loc[record]
    return meta


def get_ground_motion(record_set, record: int, component: str):
    """Return the timeseries data for a specific ground motion.

    Parameters
    ----------
    record_set : {'farfield', 'nearfield', 'nearfield_pulse', 'nearfield_no_pulse'}
        The ground motion set to load. Note that 'nearfield_pulse' and
        'nearfield_no_pulse' are just subsets of the 'nearfield' record set.
    record : int
        The record ID number. (Not the record sequence number.)
    component : {'a', 'b'}
        Which component of the record.
    """
    gmdata = load_ground_motions(record_set)
    gm = gmdata.ground_motions.loc[record, component]
    return gm


def get_response_spectrum(record_set, record: int, component: str):
    """Return the response spectrum data for a specific ground motion.

    Parameters
    ----------
    record_set : {'farfield', 'nearfield', 'nearfield_pulse', 'nearfield_no_pulse'}
        The ground motion set to load. Note that 'nearfield_pulse' and
        'nearfield_no_pulse' are just subsets of the 'nearfield' record set.
    record : int
        The record ID number. (Not the record sequence number.)
    component : {'a', 'b'}
        Which component of the record.
    """
    gmdata = load_ground_motions(record_set)
    rs = gmdata.response_spectra.sel(record=record, component=component)
    return rs


def _from_hdf5(filename) -> t.Tuple[pd.DataFrame, pd.DataFrame, xr.Dataset]:
    records = hdf_to_dataframe(filename, 'records')
    groundmotions = hdf_to_dataframe(filename, 'groundmotions')
    spectra = hdf_to_dataset(filename, 'spectra')
    spectra['component'] = spectra.component.astype('U')

    timeseries = pd.DataFrame(
        columns=['Time', 'RecordedAcceleration', 'NormalizedAcceleration'],
        index=groundmotions.index,
    )
    with h5py.File(filename, 'r') as h5:
        for recID, compID in groundmotions.index:
            group = h5['_timeseries'][str(recID)][compID]
            timeseries.loc[(recID, compID), 'Time'] = group['Time'][()]
            timeseries.loc[
                (recID, compID),
                'RecordedAcceleration'] = group['RecordedAcceleration'][()]
            timeseries.loc[
                (recID, compID),
                'NormalizedAcceleration'] = group['NormalizedAcceleration'][()]

    groundmotions = pd.concat([groundmotions, timeseries], axis='columns')

    return records, groundmotions, spectra
