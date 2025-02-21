import json
import pathlib

import scipp as sc
import tifffile as tf
from scipp.compat.dict import from_dict, to_dict


def _ensure_dimension_order(sizes: dict) -> dict:
    # Order of the dimensions is according to the HyperStacks tiff format.
    order = ['c', 't', 'z', 'y', 'x']
    return {key: sizes[key] for key in order if key in sizes}


def _to_dict(var: sc.Variable) -> dict:
    dict_var = to_dict(var)
    dict_var['dtype'] = str(var.dtype)
    dict_var['unit'] = str(var.unit)
    dict_var['values'] = var.values.tolist()
    return dict_var


def _extract_metadata_from_dataarray(da: sc.DataArray) -> dict:
    default_sizes = {'x': 1, 'y': 1, 'z': 1, 't': 1, 'c': 1}
    final_sizes = _ensure_dimension_order({**default_sizes, **da.sizes})
    return {
        'masks': {},
        'coords': {key: _to_dict(da.coords[key]) for key in da.coords},
        'data': {
            'dims': tuple(final_sizes.keys()),
            'shape': tuple(final_sizes.values()),
            'unit': str(da.unit),
            'dtype': str(da.dtype),
        },
    }


def _extract_metadata_from_datagroup(dg: sc.DataGroup) -> dict:
    raise NotImplementedError(
        "Extracting metadata from DataGroup to ESSTIFF is not yet implemented."
    )


def extract_metadata(dg: sc.DataGroup | sc.DataArray) -> dict:
    if isinstance(dg, sc.DataArray):
        return _extract_metadata_from_dataarray(dg)
    else:
        return _extract_metadata_from_datagroup(dg)


def _export_data_array(da: sc.DataArray, file_path: str | pathlib.Path) -> None:
    metadata = _extract_metadata_from_dataarray(da)
    # Make sure the data is consistent with the metadata
    # It is because ``z`` dimension and ``c`` dimension are often not present
    # but it is require by the HyperStacks and esstiffmeta schema.
    # Also, HyperStacks require specific order of dimensions.
    dims = metadata['data']['dims']
    shape = metadata['data']['shape']
    sizes = dict(zip(dims, shape, strict=True))
    final_image = sc.broadcast(da, sizes=sizes)
    tf.imwrite(
        file_path,
        final_image.values,
        imagej=True,
        metadata={'esstiffmeta': json.dumps(metadata)},
        dtype=str(final_image.dtype),
    )


def _export_data_group(dg: sc.DataGroup, file_path: str | pathlib.Path) -> None:
    raise NotImplementedError('Exporting DataGroup to ESSTIFF is not yet implemented.')


def export(dg: sc.DataGroup | sc.DataArray, file_path: str | pathlib.Path) -> None:
    if isinstance(dg, sc.DataArray):
        _export_data_array(dg, file_path)
    else:
        _export_data_group(dg, file_path)


def load(file_path: str | pathlib.Path) -> sc.DataArray | sc.DataGroup:
    with tf.TiffFile(file_path) as tif:
        metadata = json.loads(tif.imagej_metadata['esstiffmeta'])
        # metadata = tif.imagej_metadata['esstiffmeta']

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
