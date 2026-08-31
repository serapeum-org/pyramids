"""Subdatasets (#1030) and dataset-level metadata domains (#1028).

Covers the `SubDataset` value object, `RasterBase.subdatasets` enumeration,
`Dataset.open_subdataset`, and the domain-aware `get_meta_data` / `set_meta_data`
/ `meta_data_domains` accessors — including that a `NetCDF` container inherits
`subdatasets` without losing its own variable surface.
"""

from __future__ import annotations

import dataclasses
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import ContainerRasterWarning, ReadOnlyError
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset._subdataset import SubDataset
from pyramids.dataset.abstract_dataset import _reconstruct_dataset
from pyramids.netcdf import NetCDF
from pyramids.netcdf.netcdf import _reconstruct_netcdf

pytestmark = pytest.mark.core

# A classic-mode NetCDF whose GDAL handle exposes 4 subdatasets (Band1..Band4).
NETCDF_CONTAINER = (
    Path(__file__).parents[1] / "data" / "netcdf" / "cf__6v__1d2-2d4__geog__y-asc.nc"
)
# A plain single-band GeoTIFF with no subdatasets.
GEOTIFF_PLAIN = (
    Path(__file__).parents[1] / "data" / "geotiff" / "coello-without-color-table.tif"
)


@pytest.fixture
def plain() -> Dataset:
    """A 2x3 in-memory raster with no subdatasets."""
    return Dataset.from_array(
        np.zeros((2, 3), dtype="float32"),
        geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
    )


@pytest.fixture
def container():
    """A NetCDF container opened classic-mode, exposing 4 subdatasets."""
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

    def test_property_delegates_to_shared_builder(self, container):
        """The `subdatasets` property matches a list built straight from GetSubDatasets."""
        expected = [
            SubDataset(name, description, i)
            for i, (name, description) in enumerate(container.raster.GetSubDatasets())
        ]
        assert container.subdatasets == expected


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

    def test_open_subdataset_carries_access_mode(self, container):
        """The child is opened in the parent container's access mode."""
        child = container.open_subdataset(0)
        assert child.access == container.access, (
            f"child access {child.access!r} must match parent {container.access!r}"
        )

    def test_open_subdataset_forwards_open_context(self, container, mocker):
        """open_subdataset forwards access mode, gdal_env, and open options to read_file."""
        container._open_options = ("HONOUR_VALID_RANGE=NO",)
        sentinel = object()
        patched = mocker.patch.object(Dataset, "read_file", return_value=sentinel)
        result = container.open_subdataset(0)
        assert result is sentinel, "open_subdataset must return read_file's result"
        _, kwargs = patched.call_args
        assert kwargs["read_only"] == (container.access == "read_only"), (
            "parent access mode must be forwarded"
        )
        assert kwargs["open_options"] == ["HONOUR_VALID_RANGE=NO"], (
            "parent open options must be forwarded to the child open"
        )
        assert kwargs["warn_on_container"] is False, (
            "a deliberate drill-in must suppress the container warning on the child open"
        )
        assert "gdal_env" in kwargs, "gdal_env must be forwarded to read_file"


class TestNetCDFRegression:
    """A NetCDF container inherits `subdatasets` without losing its own surface."""

    def test_inherits_subdatasets_and_keeps_variable_surface(self, container):
        """`subdatasets` works on a NetCDF and `variable_names` still works."""
        assert len(container.subdatasets) == 4, "NetCDF must inherit subdatasets"
        assert container.variable_names[:3] == ["Band1", "Band2", "Band3"], (
            "the NetCDF variable surface must be unaffected"
        )


class TestContainerOpenWarning:
    """Opening a container via base `Dataset.read_file` is no longer silent (#1030)."""

    def test_container_open_warns_and_names_subdatasets(self):
        """A base read_file on a container warns and the message names its subdatasets."""
        with pytest.warns(ContainerRasterWarning) as record:
            ds = Dataset.read_file(str(NETCDF_CONTAINER))
        assert ds.band_count == 0, "the container has no bands of its own"
        container_warning = next(
            w for w in record if issubclass(w.category, ContainerRasterWarning)
        )
        message = str(container_warning.message)
        assert "subdataset" in message, "the warning explains it is a container"
        assert "Band1" in message, "the warning names the available subdatasets"

    def test_warning_caps_a_many_subdataset_container(self, mocker):
        """The message lists at most ten subdataset names, then a `… and N more` tail."""
        many = [SubDataset(f'NETCDF:"f.nc":v{i}', f"v{i}", i) for i in range(15)]
        mocker.patch(
            "pyramids.dataset.abstract_dataset.RasterBase.subdatasets",
            new_callable=mocker.PropertyMock,
            return_value=many,
        )
        with pytest.warns(ContainerRasterWarning) as record:
            Dataset.read_file(str(NETCDF_CONTAINER))
        message = str(
            next(
                w for w in record if issubclass(w.category, ContainerRasterWarning)
            ).message
        )
        assert "15 subdataset(s)" in message, "the count reflects every subdataset"
        assert "… and 5 more" in message, (
            "the list caps at ten names with a summary tail"
        )

    def test_warn_on_container_false_is_silent(self):
        """`warn_on_container=False` opens the container without warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", ContainerRasterWarning)
            ds = Dataset.read_file(str(NETCDF_CONTAINER), warn_on_container=False)
        assert len(ds.subdatasets) == 4, "the container still opens, just quietly"

    def test_plain_raster_open_does_not_warn(self):
        """A plain single-band raster opens with no container warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", ContainerRasterWarning)
            ds = Dataset.read_file(str(GEOTIFF_PLAIN))
        assert ds.band_count >= 1, "a plain raster has bands of its own"
        assert ds.subdatasets == [], "a plain raster has no subdatasets"

    def test_netcdf_read_file_does_not_emit_base_container_warning(self):
        """`NetCDF.read_file` opens a container on purpose via its own path and stays quiet.

        Guard test: `NetCDF.read_file` builds a `Container` directly and never reaches the
        base warning branch, so this pins that exemption — it would catch a future refactor
        that routed NetCDF opens back through `Dataset.read_file`.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", ContainerRasterWarning)
            nc = NetCDF.read_file(
                str(NETCDF_CONTAINER), open_as_multi_dimensional=False
            )
        assert nc.subdatasets, "NetCDF opened a real container without the base warning"

    def test_zero_band_raster_with_no_subdatasets_is_silent(self, mocker):
        """A 0-band raster that lists no subdatasets does not warn (the guarded branch).

        Test scenario:
            The container opens 0-band, but its `subdatasets` come back empty (patched),
            so `read_file` must take the inner `if subdatasets:` false path and stay silent.
        """
        mocker.patch(
            "pyramids.dataset.abstract_dataset.RasterBase.subdatasets",
            new_callable=mocker.PropertyMock,
            return_value=[],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ContainerRasterWarning)
            ds = Dataset.read_file(str(NETCDF_CONTAINER))
        assert ds.band_count == 0, "still a 0-band container, just with nothing to list"

    def test_unpickling_a_container_does_not_re_warn(self):
        """Unpickling a base-Dataset container reopens quietly (the `_reconstruct` path).

        Test scenario:
            A 0-band container opened quietly is pickled, then unpickled; the reopen goes
            through `_reconstruct_dataset`, which must pass `warn_on_container=False` so
            deserialization does not re-emit the warning the caller already saw.
        """
        ds = Dataset.read_file(str(NETCDF_CONTAINER), warn_on_container=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ContainerRasterWarning)
            restored = pickle.loads(pickle.dumps(ds))
        assert restored.band_count == 0, "the reopened container is unchanged"
        assert len(restored.subdatasets) == 4, "and still lists its four subdatasets"

    def test_value_object_open_suppresses_the_container_warning(
        self, container, mocker
    ):
        """`SubDataset.open()` reopens the child with `warn_on_container=False`."""
        patched = mocker.patch.object(Dataset, "read_file", return_value=object())
        container.subdatasets[0].open()
        _, kwargs = patched.call_args
        assert kwargs["warn_on_container"] is False, (
            "opening a subdataset value object is a deliberate drill-in, so it must be quiet"
        )

    def test_only_base_dataset_reaches_the_reconstruct_that_suppresses(self, container):
        """Pin the invariant the unpickle suppression relies on.

        `_reconstruct_dataset` passes `warn_on_container=False`, a kwarg only the base
        `Dataset.read_file` accepts. This asserts a base `Dataset` unpickles through
        `_reconstruct_dataset` while `NetCDF` routes to its own `_reconstruct_netcdf`, so
        the suppression can never reach a `read_file` override that lacks the keyword.
        """
        plain_ds = Dataset.read_file(str(GEOTIFF_PLAIN))
        assert plain_ds.__reduce__()[0] is _reconstruct_dataset, (
            "a base Dataset must unpickle through _reconstruct_dataset"
        )
        assert container.__reduce__()[0] is _reconstruct_netcdf, (
            "NetCDF must route to its own reconstruct, never the base one"
        )

    def test_warning_is_importable_from_public_errors(self):
        """The warning is re-exported from the public `pyramids.errors`, not just `_errors`."""
        from pyramids.errors import ContainerRasterWarning as Exported

        assert Exported is ContainerRasterWarning, (
            "the documented way to filter the warning must import from pyramids.errors"
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

    def test_meta_data_domains_is_list_of_str(self, plain):
        """`meta_data_domains` is a list of domain names (None normalised away)."""
        domains = plain.meta_data_domains
        assert isinstance(domains, list), "meta_data_domains must be a list"
        assert all(isinstance(d, str) for d in domains), "entries must be str"

    def test_public_name_is_meta_data_domains_not_metadata_domains(self, plain):
        """Pin the shipped name so code and docs cannot drift again (#1052).

        The listing shipped as `meta_data_domains` (consistent with the `meta_data` /
        `get_meta_data` siblings). PR #1040's body called it `metadata_domains`, which
        never existed; this asserts the real name is present and the other is absent.
        """
        assert hasattr(plain, "meta_data_domains"), (
            "meta_data_domains is the public name"
        )
        assert not hasattr(plain, "metadata_domains"), (
            "metadata_domains must not exist; one canonical name only"
        )

    def test_set_meta_data_round_trips_a_custom_domain(self, plain):
        """A custom domain can be written and read back, and appears in the list."""
        plain.set_meta_data({"A": "1", "B": "2"}, domain="MYDOMAIN")
        assert plain.get_meta_data("MYDOMAIN") == {"A": "1", "B": "2"}, "round-trip"
        assert "MYDOMAIN" in plain.meta_data_domains, "written domain must be listed"

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
        assert result == ["<root>hi</root>"], (
            f"xml domain must be a list, got {result!r}"
        )

    def test_set_meta_data_writes_xml_domain_as_list(self, plain):
        """An xml:* domain round-trips through set_meta_data as a single-element list."""
        plain.set_meta_data(["<root>v</root>"], domain="xml:FOO")
        result = plain.get_meta_data("xml:FOO")
        assert result == ["<root>v</root>"], (
            f"xml domain must round-trip, got {result!r}"
        )

    def test_set_meta_data_empty_dict_empties_keys_but_keeps_domain(self, plain):
        """Assigning {} empties a domain's keys while the domain name stays listed."""
        plain.set_meta_data({"A": "1"}, domain="D")
        plain.set_meta_data({}, domain="D")
        assert plain.get_meta_data("D") == {}, (
            "an empty dict must empty the domain keys"
        )
        assert "D" in plain.meta_data_domains, "the emptied domain name is still listed"

    def test_meta_data_domains_normalizes_none(self, plain, mocker):
        """A GDAL handle returning None for the domain list normalises to []."""
        mocker.patch.object(plain._raster, "GetMetadataDomainList", return_value=None)
        assert plain.meta_data_domains == [], "None must normalise to an empty list"

    def test_meta_data_setter_still_merges(self, plain):
        """The existing `meta_data` setter is unchanged (per-key merge)."""
        plain.meta_data = {"A": "1"}
        plain.meta_data = {"B": "2"}
        assert plain.get_meta_data() == {"A": "1", "B": "2"}, "meta_data still merges"

    def test_set_meta_data_read_only_raises(self, tmp_path):
        """Writing metadata on a read-only on-disk dataset raises ReadOnlyError."""
        path = tmp_path / "m.tif"
        Dataset.from_array(
            np.zeros((2, 2), dtype="float32"),
            geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
        ).to_file(str(path))
        ds = Dataset.read_file(str(path), read_only=True)
        with pytest.raises(ReadOnlyError, match="read-only"):
            ds.set_meta_data({"A": "1"}, domain="D")
