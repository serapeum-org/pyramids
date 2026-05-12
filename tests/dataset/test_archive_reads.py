"""Tests for PY-2 — opening rasters from inside archives.

Covers the ``_io`` archive helpers (``_infer_archive_kind`` / ``_archive_dir_vsi``
/ ``_archive_members``), ``Dataset.read_file(..., vsi=...)``,
``Dataset.from_archive`` and ``DatasetCollection.from_archive``.

The remote (``/vsizip//vsicurl/…``) path is covered here only by asserting
``_archive_dir_vsi`` builds the right VSI string — actually fetching over HTTP is
left out because GDAL's ``/vsizip/`` keys off the file-name extension, so a live
test would just re-exercise GDAL, and ``/vsicurl/`` against a loopback
``http.server`` is known to occasionally hang on Windows.
"""

from __future__ import annotations

import os
import pickle
import zipfile

import numpy as np
import pytest

from pyramids import _io
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.dataset import Dataset, DatasetCollection

pytestmark = pytest.mark.core


def _make_band(
    directory,
    name,
    value,
    *,
    dtype="int16",
    shape=(4, 5),
    cell_size=1.0,
    top_left=(0.0, 0.0),
    epsg=4326,
    no_data_value=-9999,
):
    """Write a constant single-band GeoTIFF and return its path."""
    path = os.path.join(str(directory), name)
    Dataset.create_from_array(
        np.full(shape, value, dtype=dtype),
        top_left_corner=top_left,
        cell_size=cell_size,
        epsg=epsg,
        no_data_value=no_data_value,
        path=path,
    ).close()
    return path


@pytest.fixture()
def band_zip(tmp_path):
    """A ``.zip`` of three single-band tifs (``asset.B2/B3/B4.tif``, =2/3/4) + a JSON sidecar.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        str: Path to the created ``.zip``.
    """
    src = tmp_path / "src"
    src.mkdir()
    members = [
        _make_band(src, "asset.B2.tif", 2),
        _make_band(src, "asset.B3.tif", 3),
        _make_band(src, "asset.B4.tif", 4),
    ]
    sidecar = src / "metadata.json"
    sidecar.write_text("{}")
    zip_path = tmp_path / "download.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for m in members:
            zf.write(m, arcname=os.path.basename(m))
        zf.write(sidecar, arcname="metadata.json")
    return str(zip_path)


class TestInferArchiveKind:
    """Tests for :func:`pyramids._io._infer_archive_kind`."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("x.zip", "zip"),
            ("a/b/c.zip", "zip"),
            ("x.tar", "tar"),
            ("x.tar.gz", "tar"),
            ("x.gz", "gzip"),
            ("x.tif", None),
            ("https://host/getPixels", None),
        ],
    )
    def test_kind_from_extension(self, name, expected):
        """The archive kind is inferred from the path extension (or ``None``).

        Args:
            name: Path/URL to sniff.
            expected: Expected kind, or ``None`` when not an archive name.

        Test scenario:
            ``_infer_archive_kind(name)`` — expected: ``expected``.
        """
        assert _io._infer_archive_kind(name) == expected, f"bad kind for {name!r}"


class TestArchiveDirVsi:
    """Tests for :func:`pyramids._io._archive_dir_vsi`."""

    def test_local_zip_auto(self, band_zip):
        """A local ``.zip`` (kind inferred) becomes ``/vsizip/<path>``.

        Args:
            band_zip: Path to a local ``.zip``.

        Test scenario:
            ``_archive_dir_vsi(band_zip)`` — expected: starts with ``/vsizip/``
            and ends with the zip path.
        """
        out = _io._archive_dir_vsi(band_zip)
        assert out.startswith("/vsizip/"), f"unexpected: {out!r}"
        assert out.endswith(os.path.basename(band_zip)), f"unexpected: {out!r}"

    @pytest.mark.parametrize(
        "url, kind, expected",
        [
            ("https://host/a/x.zip", "zip", "/vsizip//vsicurl/https://host/a/x.zip"),
            (
                "https://earthengine.googleapis.com/v1/projects/p/thumbnails/id:getPixels",
                "zip",
                "/vsizip//vsicurl/https://earthengine.googleapis.com/v1/projects/p/thumbnails/id:getPixels",
            ),
            ("s3://bucket/key/x.zip", "zip", "/vsizip//vsis3/bucket/key/x.zip"),
            ("https://host/x.tar.gz", "tar", "/vsitar//vsicurl/https://host/x.tar.gz"),
        ],
    )
    def test_url_gets_chained_vsi(self, url, kind, expected):
        """A URL with an explicit kind yields the chained ``/vsi<archive>//vsi<scheme>/…`` path.

        Args:
            url: Input URL.
            kind: Explicit archive kind.
            expected: Expected VSI directory path.

        Test scenario:
            ``_archive_dir_vsi(url, kind)`` — expected: ``expected`` (so
            extension-less download URLs at least get the right string).
        """
        assert _io._archive_dir_vsi(url, kind) == expected, f"bad VSI for {url!r}"

    def test_already_vsi_path_unchanged(self):
        """An already-``/vsizip/``-prefixed path is returned untouched (idempotent).

        Test scenario:
            ``_archive_dir_vsi("/vsizip//data/x.zip", "zip")`` — expected:
            unchanged.
        """
        assert (
            _io._archive_dir_vsi("/vsizip//data/x.zip", "zip") == "/vsizip//data/x.zip"
        )

    def test_auto_on_extensionless_raises(self):
        """``kind="auto"`` with no recognisable extension raises a clear error.

        Test scenario:
            ``_archive_dir_vsi("https://host/getPixels")`` — expected:
            ``FileFormatNotSupportedError`` telling the caller to pass ``kind=``.
        """
        with pytest.raises(FileFormatNotSupportedError, match="pass kind="):
            _io._archive_dir_vsi("https://host/getPixels")

    def test_unknown_kind_raises_value_error(self, band_zip):
        """An unrecognised ``kind`` raises ``ValueError``.

        Args:
            band_zip: A real archive path (the failure is on ``kind``, not the path).

        Test scenario:
            ``_archive_dir_vsi(band_zip, kind="rar")`` — expected: ``ValueError``.
        """
        with pytest.raises(ValueError, match="unknown archive kind"):
            _io._archive_dir_vsi(band_zip, kind="rar")


class TestArchiveMembers:
    """Tests for :func:`pyramids._io._archive_members`."""

    def test_lists_sorted_members(self, band_zip):
        """All top-level members are returned, sorted.

        Args:
            band_zip: A ``.zip`` of three tifs + a JSON sidecar.

        Test scenario:
            ``_archive_members(dir)`` — expected: the four names, sorted.
        """
        members = _io._archive_members(_io._archive_dir_vsi(band_zip))
        assert members == [
            "asset.B2.tif",
            "asset.B3.tif",
            "asset.B4.tif",
            "metadata.json",
        ], f"unexpected members: {members}"

    def test_member_glob_filters(self, band_zip):
        """``member_glob`` keeps only matching members.

        Args:
            band_zip: A ``.zip`` of three tifs + a JSON sidecar.

        Test scenario:
            ``_archive_members(dir, "*.tif")`` — expected: only the three tifs.
        """
        members = _io._archive_members(_io._archive_dir_vsi(band_zip), "*.tif")
        assert members == ["asset.B2.tif", "asset.B3.tif", "asset.B4.tif"], members

    def test_no_match_raises_file_not_found(self, band_zip):
        """A glob matching nothing raises ``FileNotFoundError``.

        Args:
            band_zip: A ``.zip`` with no ``.jp2`` members.

        Test scenario:
            ``_archive_members(dir, "*.jp2")`` — expected: ``FileNotFoundError``.
        """
        with pytest.raises(FileNotFoundError, match="no members matching"):
            _io._archive_members(_io._archive_dir_vsi(band_zip), "*.jp2")

    def test_not_an_archive_raises(self, tmp_path):
        """A file that is not a valid archive raises ``FileFormatNotSupportedError``.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``_archive_members("/vsizip/<garbage.zip>")`` — expected:
            ``FileFormatNotSupportedError`` (GDAL ``ReadDir`` returns ``None``).
        """
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"definitely not a zip")
        with pytest.raises(FileFormatNotSupportedError, match="could not list archive"):
            _io._archive_members(_io._archive_dir_vsi(str(bad)))


class TestReadFileVsi:
    """Tests for ``Dataset.read_file(..., vsi=...)``."""

    def test_vsi_zip_opens_first_member(self, band_zip):
        """``vsi="zip"`` opens member 0 of the archive.

        Args:
            band_zip: A ``.zip`` whose first (sorted) member is ``asset.B2.tif`` (=2).

        Test scenario:
            ``Dataset.read_file(band_zip, vsi="zip")`` — expected: a dataset
            whose first pixel is ``2``.
        """
        ds = Dataset.read_file(band_zip, vsi="zip")
        assert (
            int(ds.read_array().flat[0]) == 2
        ), "expected member 0 (asset.B2.tif, value 2)"

    def test_vsi_zip_file_index(self, band_zip):
        """``file_i`` selects which member to open.

        Args:
            band_zip: A ``.zip`` whose member 2 (sorted) is ``asset.B4.tif`` (=4).

        Test scenario:
            ``Dataset.read_file(band_zip, vsi="zip", file_i=2)`` — expected:
            first pixel ``4``.
        """
        ds = Dataset.read_file(band_zip, vsi="zip", file_i=2)
        assert (
            int(ds.read_array().flat[0]) == 4
        ), "expected member 2 (asset.B4.tif, value 4)"

    def test_vsi_auto_infers_zip(self, band_zip):
        """``vsi="auto"`` infers the kind from the extension and opens member 0.

        Args:
            band_zip: A ``.zip`` file.

        Test scenario:
            ``Dataset.read_file(band_zip, vsi="auto")`` — expected: first pixel ``2``.
        """
        ds = Dataset.read_file(band_zip, vsi="auto")
        assert int(ds.read_array().flat[0]) == 2, "auto-inferred zip, member 0"

    def test_file_index_out_of_range_raises(self, band_zip):
        """An out-of-range ``file_i`` raises ``FileNotFoundError``.

        Args:
            band_zip: A 4-member archive.

        Test scenario:
            ``Dataset.read_file(band_zip, vsi="zip", file_i=99)`` — expected:
            ``FileNotFoundError`` mentioning the member count.
        """
        with pytest.raises(FileNotFoundError, match="out of range"):
            Dataset.read_file(band_zip, vsi="zip", file_i=99)

    def test_extension_sniffing_still_works(self, band_zip):
        """Without ``vsi=``, a ``zip/member`` path still resolves (unchanged behaviour).

        Args:
            band_zip: A ``.zip`` path.

        Test scenario:
            ``Dataset.read_file(f"{band_zip}/asset.B3.tif")`` — expected: first
            pixel ``3``.
        """
        ds = Dataset.read_file(f"{band_zip}/asset.B3.tif")
        assert int(ds.read_array().flat[0]) == 3, "extension-sniffed member open"


class TestDatasetFromArchive:
    """Tests for :meth:`Dataset.from_archive`."""

    def test_merges_members_into_multiband(self, band_zip):
        """Every matching member becomes one band of the result.

        Args:
            band_zip: A ``.zip`` of three tifs (=2/3/4) + a JSON sidecar.

        Test scenario:
            ``Dataset.from_archive(band_zip, member_glob="*.tif")`` — expected:
            a 3-band dataset, band names ``["B2", "B3", "B4"]``, values ``[2, 3, 4]``.
        """
        ds = Dataset.from_archive(band_zip, member_glob="*.tif")
        assert ds.band_count == 3, f"expected 3 bands, got {ds.band_count}"
        assert ds.band_names == ["B2", "B3", "B4"], f"unexpected names: {ds.band_names}"
        assert [int(ds.read_array(band=i).flat[0]) for i in range(3)] == [
            2,
            3,
            4,
        ], "per-member values not preserved as bands"

    def test_writes_to_disk(self, band_zip, tmp_path):
        """``path=`` writes the merged raster to disk.

        Args:
            band_zip: Source archive.
            tmp_path: pytest temp directory.

        Test scenario:
            ``Dataset.from_archive(band_zip, member_glob="*.tif", path=out.tif)``
            then ``read_file(out)`` — expected: a 3-band GeoTIFF on disk.
        """
        out = tmp_path / "merged.tif"
        Dataset.from_archive(band_zip, member_glob="*.tif", path=str(out))
        assert out.exists(), "output not written"
        assert Dataset.read_file(str(out)).band_count == 3, "wrong band count on reload"


class TestDatasetCollectionFromArchive:
    """Tests for :meth:`DatasetCollection.from_archive`."""

    def test_each_member_is_a_timestep(self, band_zip):
        """Each matching member becomes one timestep of the collection.

        Args:
            band_zip: A ``.zip`` of three tifs + a JSON sidecar.

        Test scenario:
            ``DatasetCollection.from_archive(band_zip, member_glob="*.tif")`` —
            expected: ``time_length == 3`` and the template matches a member's
            shape.
        """
        col = DatasetCollection.from_archive(band_zip, member_glob="*.tif")
        assert col.time_length == 3, f"expected 3 timesteps, got {col.time_length}"
        assert col.base.shape == (
            1,
            4,
            5,
        ), f"unexpected template shape: {col.base.shape}"

    def test_default_glob_includes_all_members(self, band_zip):
        """The default ``"*"`` glob counts every top-level member (incl. the sidecar).

        Args:
            band_zip: A ``.zip`` with 3 tifs + 1 JSON = 4 members.

        Test scenario:
            ``DatasetCollection.from_archive(band_zip)`` — expected:
            ``time_length == 4`` (lazy — only the first member is opened, and
            ``asset.B2.tif`` sorts before ``metadata.json``).
        """
        col = DatasetCollection.from_archive(band_zip)
        assert col.time_length == 4, f"expected 4 members, got {col.time_length}"

    def test_lazy_member_access(self, band_zip):
        """Per-timestep arrays are read on demand from inside the archive.

        Args:
            band_zip: A ``.zip`` of three tifs (=2/3/4).

        Test scenario:
            ``col[1]`` on a ``"*.tif"`` collection — expected: an array whose
            first value is ``3`` (``asset.B3.tif``).
        """
        col = DatasetCollection.from_archive(band_zip, member_glob="*.tif")
        arr = col[1]
        assert int(np.asarray(arr).flat[0]) == 3, f"unexpected timestep-1 value: {arr}"

    def test_collection_is_picklable(self, band_zip):
        """An archive-backed collection round-trips through pickle (paths, not handles).

        Args:
            band_zip: Source archive.

        Test scenario:
            ``pickle.loads(pickle.dumps(col))`` — expected: same ``time_length``
            (the members are referenced by their ``/vsizip/…`` paths, which are
            re-openable, so unlike ``/vsimem/`` datasets this is picklable).
        """
        col = DatasetCollection.from_archive(band_zip, member_glob="*.tif")
        restored = pickle.loads(pickle.dumps(col))
        assert (
            restored.time_length == col.time_length
        ), "time_length changed on pickle round-trip"

    def test_bad_kind_raises(self, band_zip):
        """An unknown ``kind`` raises ``ValueError`` before any I/O.

        Args:
            band_zip: A real archive (failure is on ``kind``).

        Test scenario:
            ``DatasetCollection.from_archive(band_zip, kind="7z")`` — expected:
            ``ValueError``.
        """
        with pytest.raises(ValueError, match="unknown archive kind"):
            DatasetCollection.from_archive(band_zip, kind="7z")
