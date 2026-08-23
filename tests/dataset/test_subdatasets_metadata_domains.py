"""Subdatasets (#1030) and dataset-level metadata domains (#1028).

Covers the `SubDataset` value object, `RasterBase.subdatasets` enumeration,
`Dataset.open_subdataset`, and the domain-aware `get_meta_data` / `set_meta_data`
/ `metadata_domains` accessors — including that a `NetCDF` container inherits
`subdatasets` without losing its own variable surface.
"""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import ReadOnlyError
from pyramids.dataset import Dataset
from pyramids.dataset._subdataset import SubDataset

pytestmark = pytest.mark.core

# A classic-mode NetCDF whose GDAL handle exposes 4 subdatasets (Band1..Band4).
NETCDF_CONTAINER = (
    Path(__file__).parents[1] / "data" / "netcdf" / "cf__6v__1d2-2d4__geog__y-asc.nc"
)


@pytest.fixture
def plain() -> Dataset:
    """A 2x3 in-memory raster with no subdatasets."""
    return Dataset.create_from_array(
        np.zeros((2, 3), dtype="float32"),
        top_left_corner=(0, 0),
        cell_size=1.0,
        epsg=4326,
    )


@pytest.fixture
def container():
    """A NetCDF container opened classic-mode, exposing 4 subdatasets."""
    from pyramids.netcdf import NetCDF

    return NetCDF.read_file(str(NETCDF_CONTAINER), open_as_multi_dimensional=False)


class TestSubDataset:
    """The `SubDataset` frozen value object."""

    def test_fields_and_equality(self):
        """It carries name/description/index and compares by value."""
        sd = SubDataset("NAME", "desc", 0)
        assert (sd.name, sd.description, sd.index) == ("NAME", "desc", 0)
        assert sd == SubDataset("NAME", "desc", 0), "value equality expected"
        assert sd != SubDataset("OTHER", "desc", 0), "differing name must differ"

    def test_is_frozen(self):
        """It is immutable."""
        sd = SubDataset("N", "d", 0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            sd.name = "X"  # type: ignore[misc]

    def test_pickle_round_trip(self):
        """It survives a pickle round-trip (it is plain data)."""
        sd = SubDataset("N", "d", 2)
        assert pickle.loads(pickle.dumps(sd)) == sd, "pickle must round-trip"

    def test_exported_from_package(self):
        """SubDataset is importable from the public pyramids.dataset namespace."""
        from pyramids.dataset import SubDataset as Exported

        assert Exported is SubDataset, "SubDataset must be re-exported from the package"


class TestSubdatasetsProperty:
    """`RasterBase.subdatasets` enumeration."""

    def test_plain_raster_has_none(self, plain):
        """A normal raster reports no subdatasets."""
        assert plain.subdatasets == [], "a plain raster has no subdatasets"

    def test_container_enumerates_in_order(self, container):
        """A container reports one entry per nested raster, in order."""
        subs = container.subdatasets
        assert len(subs) == 4, f"expected 4 subdatasets, got {len(subs)}"
        assert [s.index for s in subs] == [0, 1, 2, 3], "indices must be 0..n in order"
        assert all(isinstance(s, SubDataset) for s in subs), "entries are SubDataset"
        assert subs[0].name.endswith(":Band1"), f"first name unexpected: {subs[0].name}"


class TestOpenSubdataset:
    """`Dataset.open_subdataset` — opens a nested raster as a base Dataset."""

    def test_open_by_index(self, container):
        """Opening by 0-based index yields a base Dataset with the band."""
        sub = container.open_subdataset(0)
        assert type(sub) is Dataset, f"expected a base Dataset, got {type(sub)}"
        assert sub.band_count == 1, "the Band1 subdataset has one band"

    def test_open_by_name(self, container):
        """Opening by connection-string name yields the same raster."""
        name = container.subdatasets[0].name
        sub = container.open_subdataset(name)
        assert type(sub) is Dataset, f"expected a base Dataset, got {type(sub)}"
        assert sub.band_count == 1, "the Band1 subdataset has one band"

    def test_unknown_name_raises(self, container):
        """An unknown subdataset name raises ValueError."""
        with pytest.raises(ValueError, match="not a subdataset"):
            container.open_subdataset('NETCDF:"nope.nc":var')

    def test_out_of_range_index_raises(self, container):
        """An out-of-range index raises IndexError."""
        with pytest.raises(IndexError):
            container.open_subdataset(999)

    def test_opened_subdataset_pickles(self, container):
        """An opened subdataset round-trips through pickle (its name reopens)."""
        sub = container.open_subdataset(0)
        reopened = pickle.loads(pickle.dumps(sub))
        assert reopened.band_count == 1, "reopened subdataset must have its band"

    def test_value_object_open_returns_base_dataset(self, container):
        """`SubDataset.open()` opens the nested raster as a base Dataset."""
        sub = container.subdatasets[0].open()
        assert type(sub) is Dataset, f"expected a base Dataset, got {type(sub)}"
        assert sub.band_count == 1, "the Band1 subdataset has one band"

    def test_bool_index_raises(self, container):
        """A bool key is rejected, not silently treated as index 0/1."""
        with pytest.raises(TypeError, match="not bool"):
            container.open_subdataset(True)

    def test_negative_index_opens_from_end(self, container):
        """A negative index counts from the end, per Python list semantics."""
        last = container.open_subdataset(-1)
        assert type(last) is Dataset, f"expected a base Dataset, got {type(last)}"
        assert last.band_count == 1, "the last subdataset has one band"

    def test_non_int_non_str_key_raises_type_error(self, container):
        """A key that is neither an int index nor a str name raises TypeError."""
        with pytest.raises(TypeError, match="int index or a str name"):
            container.open_subdataset(1.5)  # type: ignore[arg-type]


class TestNetCDFRegression:
    """A NetCDF container inherits `subdatasets` without losing its own surface."""

    def test_inherits_subdatasets_and_keeps_variable_surface(self, container):
        """`subdatasets` works on a NetCDF and `variable_names` still works."""
        assert len(container.subdatasets) == 4, "NetCDF must inherit subdatasets"
        assert container.variable_names[:3] == ["Band1", "Band2", "Band3"], (
            "the NetCDF variable surface must be unaffected"
        )


class TestMetadataDomains:
    """Dataset-level metadata domains (#1028)."""

    def test_default_domain_matches_meta_data(self, plain):
        """`get_meta_data()` equals `meta_data` on a plain Dataset."""
        assert plain.get_meta_data() == plain.meta_data, "default domain must match"

    def test_read_named_domain(self, plain):
        """A named domain reads as a dict (empty if absent)."""
        result = plain.get_meta_data("IMAGE_STRUCTURE")
        assert isinstance(result, dict), f"expected a dict, got {type(result)}"

    def test_metadata_domains_is_list_of_str(self, plain):
        """`metadata_domains` is a list of domain names (None normalised away)."""
        domains = plain.metadata_domains
        assert isinstance(domains, list), "metadata_domains must be a list"
        assert all(isinstance(d, str) for d in domains), "entries must be str"

    def test_set_meta_data_round_trips_a_custom_domain(self, plain):
        """A custom domain can be written and read back, and appears in the list."""
        plain.set_meta_data({"A": "1", "B": "2"}, domain="MYDOMAIN")
        assert plain.get_meta_data("MYDOMAIN") == {"A": "1", "B": "2"}, "round-trip"
        assert "MYDOMAIN" in plain.metadata_domains, "written domain must be listed"

    def test_set_meta_data_replaces_the_domain(self, plain):
        """The setter replaces a domain's metadata rather than merging."""
        plain.set_meta_data({"A": "1", "B": "2"}, domain="D")
        plain.set_meta_data({"C": "3"}, domain="D")
        assert plain.get_meta_data("D") == {"C": "3"}, "set_meta_data must replace"

    def test_set_meta_data_rejects_default_domain(self, plain):
        """set_meta_data refuses the default domain (it would drop GDAL-managed keys).

        The default domain holds georeferencing keys (AREA_OR_POINT, CF axis
        metadata); a whole-domain replace would silently wipe them, so the setter
        points callers at the merging meta_data setter instead.
        """
        with pytest.raises(ValueError, match="named domains only"):
            plain.set_meta_data({"X": "1"})

    def test_get_meta_data_xml_domain_returns_list(self, plain):
        """An xml:* domain returns a list of XML strings, mirroring GDAL."""
        plain._raster.SetMetadata(["<root>hi</root>"], "xml:TEST")
        result = plain.get_meta_data("xml:TEST")
        assert result == ["<root>hi</root>"], f"xml domain must be a list, got {result!r}"

    def test_metadata_domains_normalizes_none(self, plain, mocker):
        """A GDAL handle returning None for the domain list normalises to []."""
        mocker.patch.object(
            plain._raster, "GetMetadataDomainList", return_value=None
        )
        assert plain.metadata_domains == [], "None must normalise to an empty list"

    def test_meta_data_setter_still_merges(self, plain):
        """The existing `meta_data` setter is unchanged (per-key merge)."""
        plain.meta_data = {"A": "1"}
        plain.meta_data = {"B": "2"}
        assert plain.get_meta_data() == {"A": "1", "B": "2"}, "meta_data still merges"

    def test_set_meta_data_read_only_raises(self, tmp_path):
        """Writing metadata on a read-only on-disk dataset raises ReadOnlyError."""
        path = tmp_path / "m.tif"
        Dataset.create_from_array(
            np.zeros((2, 2), dtype="float32"),
            top_left_corner=(0, 0),
            cell_size=1.0,
            epsg=4326,
        ).to_file(str(path))
        ds = Dataset.read_file(str(path), read_only=True)
        with pytest.raises(ReadOnlyError, match="read-only"):
            ds.set_meta_data({"A": "1"}, domain="D")
