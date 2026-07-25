"""Read-only metadata-setter guards (ARC-6) and closed-dataset guards (ARC-43)."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._errors import ReadOnlyError
from pyramids.dataset import Dataset


@pytest.fixture
def ro_dataset(tmp_path):
    """A single-band GeoTIFF reopened read-only."""
    path = tmp_path / "ro.tif"
    Dataset.create_from_array(
        np.ones((3, 3), dtype="float32"),
        top_left_corner=(0.0, 3.0),
        cell_size=1.0,
        epsg=4326,
        path=str(path),
    )
    return Dataset.read_file(str(path), read_only=True), path


_METADATA_SETTERS = [
    pytest.param(lambda ds: ds.set_crs(epsg=3857), id="set_crs"),
    pytest.param(lambda ds: setattr(ds, "epsg", 3857), id="epsg"),
    pytest.param(lambda ds: setattr(ds, "crs", ds.crs), id="crs"),
    pytest.param(lambda ds: setattr(ds, "meta_data", {"k": "v"}), id="meta_data"),
    pytest.param(lambda ds: setattr(ds, "scale", [2.0]), id="scale"),
    pytest.param(lambda ds: setattr(ds, "offset", [1.0]), id="offset"),
    pytest.param(lambda ds: setattr(ds, "band_units", ["m"]), id="band_units"),
    pytest.param(lambda ds: setattr(ds, "band_names", ["renamed"]), id="band_names"),
    pytest.param(lambda ds: setattr(ds, "no_data_value", [-9999.0]), id="no_data_value"),
]


class TestReadOnlyMetadataSetters:
    """ARC-6: metadata setters reject a read-only dataset instead of PAM-spilling."""

    @pytest.mark.parametrize("mutate", _METADATA_SETTERS)
    def test_setter_raises_read_only(self, ro_dataset, mutate):
        """Each metadata setter raises ReadOnlyError on a read-only dataset."""
        ds, _ = ro_dataset
        with pytest.raises(ReadOnlyError, match="read-only"):
            mutate(ds)

    @pytest.mark.parametrize("mutate", _METADATA_SETTERS)
    def test_setter_writes_no_pam_sidecar(self, ro_dataset, mutate):
        """A rejected setter leaves no .aux.xml PAM sidecar on disk."""
        ds, path = ro_dataset
        sidecar = path.with_suffix(path.suffix + ".aux.xml")
        with pytest.raises(ReadOnlyError):
            mutate(ds)
        ds.close()
        assert not sidecar.exists(), f"a PAM sidecar was spilled: {sidecar}"

    def test_in_memory_copy_setter_allowed(self, ro_dataset):
        """A writable MEM copy (access=='read_only', empty path) accepts a setter."""
        ds, _ = ro_dataset
        copy = ds.copy()
        copy.epsg = 3857
        assert copy.epsg == 3857, "in-memory copy should accept the epsg setter"

    def test_vsimem_setter_allowed(self):
        """A /vsimem raster (in-memory, access=='read_only') is not blocked."""
        src = Dataset.create_from_array(
            np.ones((3, 3), dtype="float32"),
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
        )
        ds = Dataset.from_bytes(src.to_bytes())
        assert ds.file_name.startswith("/vsimem/"), "expected a /vsimem-backed raster"
        ds.epsg = 3857
        assert ds.epsg == 3857, "/vsimem raster should accept the epsg setter"


_CLOSED_READS = [
    pytest.param(lambda ds: ds.meta_data, id="meta_data"),
    pytest.param(lambda ds: ds.driver_type, id="driver_type"),
    pytest.param(lambda ds: repr(ds), id="repr"),
    pytest.param(lambda ds: str(ds), id="str"),
    pytest.param(lambda ds: ds._iloc(0), id="_iloc"),
]


class TestClosedDatasetGuards:
    """ARC-43: state reads on a closed dataset raise a consistent error."""

    @pytest.mark.parametrize("read", _CLOSED_READS)
    def test_state_read_raises_after_close(self, ro_dataset, read):
        """Each state read raises RuntimeError('closed') after close()."""
        ds, _ = ro_dataset
        ds.close()
        with pytest.raises(RuntimeError, match="closed dataset"):
            read(ds)

    def test_repr_does_not_return_none_string(self, ro_dataset):
        """repr() raises rather than silently returning the string 'None'."""
        ds, _ = ro_dataset
        ds.close()
        with pytest.raises(RuntimeError):
            repr(ds)
