import pathlib

import scipp as sc
import tifffile as tf


def _ensure_dimension_order(sizes: dict) -> dict:
    # Order of the dimensions is according to the HyperStacks tiff format.
    order = ["c", "t", "z", "y", "x"]
    return {key: sizes[key] for key in order if key in sizes}


def _extract_metadata_from_dataarray(da: sc.DataArray) -> dict:
    default_sizes = {"x": 1, "y": 1, "z": 1, "t": 1, "c": 1}
    final_sizes = {**default_sizes, **da.sizes}
    return {
        "sizes": _ensure_dimension_order(final_sizes),
        "unit": da.unit,
        "coords": {key: da.coords[key] for key in da.coords},
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
    # but it is require by the HyperStacks and ESSTIFFMETA schema.
    # Also, HyperStacks require specific order of dimensions.
    final_image = sc.broadcast(da, sizes=metadata["sizes"])
    tf.imwrite(
        file_path,
        final_image.values,
        imagej=True,
        metadata={"ESSTIFFMETA": metadata},
        dtype=str(final_image.dtype),
    )


def _export_data_group(dg: sc.DataGroup, file_path: str | pathlib.Path) -> None:
    raise NotImplementedError("Exporting DataGroup to ESSTIFF is not yet implemented.")


def export(dg: sc.DataGroup | sc.DataArray, file_path: str | pathlib.Path) -> None:
    if isinstance(dg, sc.DataArray):
        _export_data_array(dg, file_path)
    else:
        _export_data_group(dg, file_path)
