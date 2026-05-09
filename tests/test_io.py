from pathlib import Path
from typing import List

import pytest
from osgeo import gdal

from pyramids._io import _parse_path, extract_from_gz, read_file
from pyramids.base._errors import FileFormatNotSupportedError


class TestZipFiles:
    def test_one_compressed_zip_file(
        self,
        one_compressed_file_zip: str,
        multiple_compressed_file_zip_content: List[str],
    ):
        first_file = multiple_compressed_file_zip_content[0]
        """zip file contzins only one compressed file"""
        res = _parse_path(one_compressed_file_zip)
        assert res == f"/vsizip/{one_compressed_file_zip}/{first_file}"

    def test_multiple_compressed_zip_file(
        self,
        multiple_compressed_file_zip: str,
        multiple_compressed_file_zip_content: List[str],
    ):
        first_file = multiple_compressed_file_zip_content[0]
        second_file = multiple_compressed_file_zip_content[1]
        """zip file contains multiple compressed file"""
        res = _parse_path(multiple_compressed_file_zip, file_i=0)
        assert res == f"/vsizip/{multiple_compressed_file_zip}/{first_file}"
        res = _parse_path(multiple_compressed_file_zip, file_i=1)
        assert res == f"/vsizip/{multiple_compressed_file_zip}/{second_file}"

    def test_give_path_inside_zip_file(
        self,
        multiple_compressed_file_zip: str,
        multiple_compressed_file_zip_content: List[str],
    ):
        first_file = multiple_compressed_file_zip_content[0]
        """zip file contains multiple compressed file"""
        res = _parse_path(f"{multiple_compressed_file_zip}/{first_file}")
        assert res == f"/vsizip/{multiple_compressed_file_zip}/{first_file}"


class TestGzipFiles:
    def test_one_compressed_gzip_file(self, one_compressed_file_gzip: str):
        """zip file contains only one compressed file"""
        res = _parse_path(one_compressed_file_gzip)
        assert res == f"/vsigzip/{one_compressed_file_gzip}"

    def test_multiple_compressed_gzip_file(
        self,
        multiple_compressed_file_gzip: str,
        multiple_compressed_file_gzip_content: List[str],
    ):
        """zip file contains only one compressed file"""
        first_file = multiple_compressed_file_gzip_content[0]
        second_file = multiple_compressed_file_gzip_content[1]

        res = _parse_path(multiple_compressed_file_gzip)
        assert res == f"/vsigzip/{multiple_compressed_file_gzip}/{first_file}"
        res = _parse_path(multiple_compressed_file_gzip, file_i=0)
        assert res == f"/vsigzip/{multiple_compressed_file_gzip}/{first_file}"
        res = _parse_path(multiple_compressed_file_gzip, file_i=1)
        assert res == f"/vsigzip/{multiple_compressed_file_gzip}/{second_file}"

    def test_give_path_inside_gzip_file(
        self,
        multiple_compressed_file_gzip: str,
        multiple_compressed_file_gzip_content: List[str],
    ):
        first_file = multiple_compressed_file_gzip_content[0]
        """zip file contains multiple compressed file"""
        res = _parse_path(f"{multiple_compressed_file_gzip}/{first_file}")
        assert res == f"/vsigzip/{multiple_compressed_file_gzip}/{first_file}"


def test_extract_from_gz(
    one_compressed_file_gzip,
    unzip_gzip_file_name,
):
    unzip_gzip_file_name = Path(unzip_gzip_file_name)
    extract_from_gz(one_compressed_file_gzip, unzip_gzip_file_name, delete=False)
    assert unzip_gzip_file_name.exists()
    # deleting the uncompressed file
    unzip_gzip_file_name.unlink()


class TestReadZip:
    def test_read_single_compressed_zip(self, one_compressed_file_zip: str):
        src = read_file(one_compressed_file_zip)
        assert isinstance(src, gdal.Dataset)
        assert src.GetDescription() == f"/vsizip/{one_compressed_file_zip}/1.asc"

    def test_multiple_compressed_zip_file(self, multiple_compressed_file_zip: str):
        src = read_file(multiple_compressed_file_zip)
        assert isinstance(src, gdal.Dataset)
        assert src.GetDescription() == f"/vsizip/{multiple_compressed_file_zip}/1.asc"


class TestReadGzip:
    def test_read_single_compressed_gzip(self, one_compressed_file_gzip: str):
        src = read_file(one_compressed_file_gzip)
        assert isinstance(src, gdal.Dataset)
        assert src.GetDescription() == f"/vsigzip/{one_compressed_file_gzip}"

    def test_multiple_compressed_gzip_file_error(
        self, multiple_compressed_file_gzip: str
    ):
        with pytest.raises(FileFormatNotSupportedError):
            read_file(multiple_compressed_file_gzip)

    def test_multiple_compressed_gzip_file_with_internal_path(
        self,
        multiple_compressed_file_gzip: str,
        multiple_compressed_file_gzip_content: List[str],
    ):
        first_file = multiple_compressed_file_gzip_content[0]
        try:
            read_file(f"{multiple_compressed_file_gzip}/{first_file}")
        except FileFormatNotSupportedError:
            pass


class TestReadTar:
    def test_read_single_compressed(self, one_compressed_file_tar: str):
        src = read_file(one_compressed_file_tar)
        assert isinstance(src, gdal.Dataset)

    def test_multiple_compressed_tar_file_with_internal_path(
        self,
        multiple_compressed_file_tar: str,
        multiple_compressed_file_gzip_content: List[str],
    ):
        first_file = multiple_compressed_file_gzip_content[0]
        src = read_file(f"{multiple_compressed_file_tar}/{first_file}")
        assert isinstance(src, gdal.Dataset)


class TestHttpsrequest:
    @pytest.mark.vfs
    def test_read_from_aws(self):
        url = (
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/31/U/FU/2020/3"
            "/S2A_31UFU_20200328_0_L2A/B01.tif"
        )
        src = read_file(url)
        assert src.RasterXSize == 1830
        assert src.RasterYSize == 1830
        assert isinstance(src, gdal.Dataset)


import gzip
import shutil
import tempfile

import numpy as np

from pyramids._io import (
    _get_tar_path,
    _is_gzip,
    _is_tar,
    _is_zip,
    extract_from_gz,
    insert_space,
    to_ascii,
)

pytestmark = pytest.mark.core


class TestExtractFromGzDelete:
    """Tests for extract_from_gz with delete=True."""

    def test_extract_and_delete_input(self, tmp_path):
        """Extracting with delete=True should remove the original .gz file."""
        # create a temporary gzip file
        content = b"Hello World compressed content"
        gz_path = tmp_path / "test_delete.gz"
        out_path = tmp_path / "test_delete_output.txt"
        with gzip.open(gz_path, "wb") as f:
            f.write(content)
        assert gz_path.exists(), "gz file should exist before extraction"
        extract_from_gz(gz_path, out_path, delete=True)
        assert out_path.exists(), "output file should exist after extraction"
        assert not gz_path.exists(), "gz file should be deleted when delete=True"
        # verify content
        with open(out_path, "rb") as f:
            assert f.read() == content, "extracted content should match original"


class TestReadFileTypeError:
    """Tests for read_file with non-string path."""

    def test_non_string_path_raises_type_error(self):
        """Passing a non-string/non-Path path to read_file should raise TypeError."""
        with pytest.raises(TypeError, match="string or Path type"):
            read_file(12345)

    def test_list_path_raises_type_error(self):
        """Passing a list as path to read_file should raise TypeError."""
        with pytest.raises(TypeError, match="string or Path type"):
            read_file(["file1.tif", "file2.tif"])


class TestReadFileNotFound:
    """Tests for read_file when the file does not exist."""

    def test_nonexistent_file_raises(self):
        """Reading a file that does not exist should raise an error."""
        with pytest.raises(Exception):
            read_file("/nonexistent/path/to/file_that_does_not_exist.tif")

    def test_nonexistent_compressed_file_raises(self):
        """Reading a non-existent .zip file should raise an error."""
        with pytest.raises(Exception):
            read_file("nonexistent_file.zip")

    def test_unrecognized_format_non_compressed_reraises(self):
        """A file with unrecognized format and non-compressed extension re-raises the original error."""
        with pytest.raises(Exception):
            read_file("/nonexistent/path/to/bad_format.xyz")


class TestReadFileExceptionBranches:
    """Tests covering exception handling branches in read_file using mocks."""

    def test_not_recognized_format_compressed_raises_file_format_not_supported(
        self, monkeypatch
    ):
        """When GDAL raises 'not recognized' for a compressed path, FileFormatNotSupportedError is raised."""
        import pyramids._io as io_mod

        def mock_parse_path(path, file_i=0):
            """Return a path ending in .gz to trigger compressed branch."""
            return "fake_path.gz"

        def mock_open_shared(path, access):
            """Simulate GDAL raising 'not recognized' for a compressed file."""
            raise RuntimeError(f"'{path}' not recognized as a supported file format.")

        monkeypatch.setattr(io_mod, "_parse_path", mock_parse_path)
        monkeypatch.setattr(gdal, "OpenShared", mock_open_shared)
        with pytest.raises(FileFormatNotSupportedError):
            read_file("some_file.tif")

    def test_not_recognized_format_non_compressed_reraises(self, monkeypatch):
        """When GDAL raises 'not recognized' for a non-compressed file, the original error is re-raised."""
        import pyramids._io as io_mod

        def mock_parse_path(path, file_i=0):
            """Return a non-compressed path."""
            return "fake_path.tif"

        def mock_open_shared(path, access):
            """Simulate GDAL raising 'not recognized' for a regular file."""
            raise RuntimeError(f"'{path}' not recognized as a supported file format.")

        monkeypatch.setattr(io_mod, "_parse_path", mock_parse_path)
        monkeypatch.setattr(gdal, "OpenShared", mock_open_shared)
        with pytest.raises(RuntimeError, match="not recognized"):
            read_file("some_file.tif")

    def test_no_such_file_raises_file_not_found(self, monkeypatch):
        """When GDAL raises 'No such file or directory', FileNotFoundError is raised."""
        import pyramids._io as io_mod

        def mock_parse_path(path, file_i=0):
            """Return a non-compressed path."""
            return "missing_file.tif"

        def mock_open_shared(path, access):
            """Simulate GDAL raising 'No such file or directory'."""
            raise RuntimeError(f"'{path}': No such file or directory")

        monkeypatch.setattr(io_mod, "_parse_path", mock_parse_path)
        monkeypatch.setattr(gdal, "OpenShared", mock_open_shared)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            read_file("missing_file.tif")

    def test_other_exception_reraises(self, monkeypatch):
        """When GDAL raises an unrecognized error, the original exception is re-raised."""
        import pyramids._io as io_mod

        def mock_parse_path(path, file_i=0):
            """Return a non-compressed path."""
            return "some_file.tif"

        def mock_open_shared(path, access):
            """Simulate GDAL raising an unknown error."""
            raise RuntimeError("some completely unexpected GDAL error")

        monkeypatch.setattr(io_mod, "_parse_path", mock_parse_path)
        monkeypatch.setattr(gdal, "OpenShared", mock_open_shared)
        with pytest.raises(RuntimeError, match="unexpected"):
            read_file("some_file.tif")


class TestToAscii:
    """Tests for to_ascii function."""

    def test_non_string_path_raises_type_error(self):
        """Passing a non-string/non-Path path to to_ascii should raise TypeError."""
        arr = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(TypeError, match="string or Path type"):
            to_ascii(arr, 1.0, 0.0, 0.0, -9999.0, 12345)

    def test_existing_file_raises_file_exists_error(self, tmp_path):
        """Writing to an existing file should raise FileExistsError."""
        arr = np.ones((3, 3), dtype=np.float32)
        path = str(tmp_path / "existing.asc")
        # create the file first
        with open(path, "w") as f:
            f.write("dummy")
        with pytest.raises(FileExistsError, match="same path"):
            to_ascii(arr, 1.0, 0.0, 0.0, -9999.0, path)


class TestGetTarPath:
    """Tests for _get_tar_path."""

    def test_tar_path_with_internal_path(self):
        """A path containing .tar with an internal path should get /vsitar/ prefix."""
        result = _get_tar_path("archive.tar/internal_file.asc")
        assert (
            result == "/vsitar/archive.tar/internal_file.asc"
        ), f"Expected /vsitar/ prefix with internal path, got {result}"

    def test_tar_path_without_internal_path(self):
        """A .tar path without internal path should still get /vsitar/ prefix."""
        result = _get_tar_path("archive.tar")
        assert (
            result == "/vsitar/archive.tar"
        ), f"Expected /vsitar/ prefix, got {result}"


class TestInsertSpace:
    """Tests for insert_space helper."""

    def test_insert_space_adds_trailing_spaces(self):
        """insert_space should append two trailing spaces."""
        result = insert_space(42)
        assert result == "42  ", f"Expected '42  ', got '{result}'"

    def test_insert_space_with_string(self):
        """insert_space with string input should append two trailing spaces."""
        result = insert_space("hello")
        assert result == "hello  ", f"Expected 'hello  ', got '{result}'"


class TestHelperFunctions:
    """Tests for _is_zip, _is_gzip, _is_tar."""

    def test_is_zip_with_zip_extension(self):
        """_is_zip should return True for .zip files."""
        assert _is_zip("file.zip") is True, "file.zip should be identified as zip"

    def test_is_zip_with_internal_path(self):
        """_is_zip should return True for paths containing .zip."""
        assert (
            _is_zip("file.zip/internal.asc") is True
        ), "path containing .zip should be identified"

    def test_is_zip_non_zip(self):
        """_is_zip should return False for non-zip files."""
        assert _is_zip("file.tif") is False, "file.tif should not be zip"

    def test_is_gzip_with_gz_extension(self):
        """_is_gzip should return True for .gz files."""
        assert _is_gzip("file.gz") is True, "file.gz should be identified as gzip"

    def test_is_gzip_non_gz(self):
        """_is_gzip should return False for non-gz files."""
        assert _is_gzip("file.tif") is False, "file.tif should not be gzip"

    def test_is_tar_with_tar_extension(self):
        """_is_tar should return True for .tar files."""
        assert _is_tar("file.tar") is True, "file.tar should be identified as tar"

    def test_is_tar_with_tar_gz_extension(self):
        """_is_tar should return True for .tar.gz files."""
        assert _is_tar("file.tar.gz") is True, "file.tar.gz should be identified as tar"

    def test_is_tar_non_tar(self):
        """_is_tar should return False for non-tar files."""
        assert _is_tar("file.tif") is False, "file.tif should not be tar"


class TestParsePathRemote:
    """URL-scheme rewriting in _parse_path (Task 5)."""

    def test_s3_becomes_vsis3(self):
        assert _parse_path("s3://bucket/key.tif") == "/vsis3/bucket/key.tif"

    def test_gs_becomes_vsigs(self):
        assert _parse_path("gs://bucket/key.tif") == "/vsigs/bucket/key.tif"

    def test_az_becomes_vsiaz(self):
        assert _parse_path("az://container/blob.tif") == "/vsiaz/container/blob.tif"

    def test_https_becomes_vsicurl(self):
        assert _parse_path("https://foo.com/x.tif") == "/vsicurl/https://foo.com/x.tif"

    def test_http_becomes_vsicurl(self):
        assert _parse_path("http://foo.com/x.tif") == "/vsicurl/http://foo.com/x.tif"

    def test_already_vsi_passthrough(self):
        assert _parse_path("/vsicurl/https://foo/x.tif") == "/vsicurl/https://foo/x.tif"

    def test_vsis3_passthrough(self):
        assert _parse_path("/vsis3/bucket/key.tif") == "/vsis3/bucket/key.tif"

    def test_local_path_unchanged(self):
        assert _parse_path("/home/user/x.tif") == "/home/user/x.tif"

    def test_windows_local_unchanged(self):
        assert _parse_path("C:/data/x.tif") == "C:/data/x.tif"

    def test_presigned_url_query_preserved(self):
        url = "https://foo.com/x.tif?sig=abc&exp=123"
        assert _parse_path(url) == f"/vsicurl/{url}"
