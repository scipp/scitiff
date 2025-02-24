# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import json
import pathlib

import scipp as sc
import tifffile as tf
from scipp.compat.dict import from_dict


def _ensure_dimension_order(sizes: dict) -> dict:
    # Order of the dimensions is according to the HyperStacks tiff format.
    order = ['c', 't', 'z', 'y', 'x']
    return {key: sizes[key] for key in order if key in sizes}


def _to_dict(var: sc.Variable) -> dict:
    return {
        'dims': list(var.dims),
        'dtype': str(var.dtype),
        'shape': list(var.shape),
        'unit': str(var.unit),
        'values': var.values.tolist(),
    }


def _extract_metadata_from_dataarray(da: sc.DataArray) -> dict:
    default_sizes = {'x': 1, 'y': 1, 'z': 1, 't': 1, 'c': 1}
    final_sizes = _ensure_dimension_order({**default_sizes, **da.sizes})
    return {
        'masks': {key: _to_dict(mask) for key, mask in da.masks.items()},
        'coords': {key: _to_dict(coord) for key, coord in da.coords.items()},
        'data': {
            'dims': list(final_sizes.keys()),
            'shape': list(final_sizes.values()),
            'unit': str(da.unit),
            'dtype': str(da.dtype),
        },
    }


def _extract_metadata_from_datagroup(dg: sc.DataGroup) -> dict:
    raise NotImplementedError(
        "Extracting metadata from DataGroup to SCITIFF is not yet implemented."
    )


def extract_metadata(dg: sc.DataGroup | sc.DataArray) -> dict:
    if isinstance(dg, sc.DataArray):
        _metadata = {"image": _extract_metadata_from_dataarray(dg)}
    else:
        _metadata = _extract_metadata_from_datagroup(dg)

    return {'scitiffmeta': _metadata}


def _export_data_array(da: sc.DataArray, file_path: str | pathlib.Path) -> None:
    metadata = _extract_metadata_from_dataarray(da)
    # Make sure the data is consistent with the metadata
    # It is because ``z`` dimension and ``c`` dimension are often not present
    # but it is require by the HyperStacks and scitiffmeta schema.
    # Also, HyperStacks require specific order of dimensions.
    dims = metadata['data']['dims']
    shape = metadata['data']['shape']
    sizes = dict(zip(dims, shape, strict=True))
    final_image = sc.broadcast(da, sizes=sizes)
    tf.imwrite(
        file_path,
        final_image.values,
        imagej=True,
        metadata=json.dumps(metadata),
        dtype=str(final_image.dtype),
    )


def _export_data_group(dg: sc.DataGroup, file_path: str | pathlib.Path) -> None:
    raise NotImplementedError('Exporting DataGroup to SCITIFF is not yet implemented.')


def export_scitiff(
    dg: sc.DataGroup | sc.DataArray, file_path: str | pathlib.Path
) -> None:
    if isinstance(dg, sc.DataArray):
        _export_data_array(dg, file_path)
    else:
        _export_data_group(dg, file_path)


def load_scitiff(file_path: str | pathlib.Path) -> sc.DataArray | sc.DataGroup:
    with tf.TiffFile(file_path) as tif:
        metadata = json.loads(tif.imagej_metadata['scitiffmeta'])

    img = tf.imread(file_path, squeeze=True)
    # imread squeezes the dimensions, so we need to keep the original sizes
    # if not squeezed, the sizes have 1 extra dimension that I do not understand...
    sizes = dict(zip(metadata['data']['dims'], metadata['data']['shape'], strict=True))
    # Drop 1 size dimensions
    sizes = {key: value for key, value in sizes.items() if value > 1}
    # Update the metadata with the correct sizes
    metadata['data']['dims'] = tuple(sizes.keys())
    metadata['data']['shape'] = tuple(sizes.values())
    metadata['data']['values'] = img
    return from_dict(metadata)
