from __future__ import annotations

import fnmatch
import gzip
import itertools
import tarfile
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
from osgeo import gdal

from pyramids.base import remote
from pyramids.base._errors import FileFormatNotSupportedError

COMPRESSED_FILES_EXTENSIONS = [".zip", ".gz", ".tar"]
DOES_NOT_SUPPORT_INTERNAL = [".gz"]

# User-facing archive ``kind`` -> GDAL VSI handler prefix. ``"tar.gz"`` /
# ``"tgz"`` go through ``/vsitar/`` (GDAL's tar handler decompresses gzip
# inline); ``"gz"`` / ``"gzip"`` is for a single gzip-compressed file.
_VSI_ARCHIVE_KINDS: dict[str, str] = {
    "zip": "/vsizip/",
    "tar": "/vsitar/",
    "tar.gz": "/vsitar/",
    "tgz": "/vsitar/",
    "gz": "/vsigzip/",
    "gzip": "/vsigzip/",
}

# Process-wide monotonic counter guaranteeing `/vsimem/` path uniqueness.
# `time.time_ns()` repeats within a clock tick (coarse on Windows), so a
# strictly increasing counter — not entropy — is what makes successive
# paths collision-proof within a process run. `next()` on an
# itertools.count is atomic under the GIL.
_VSIMEM_COUNTER = itertools.count()


def new_vsimem_path(suffix: str = ".tif") -> str:
    """Return a fresh, unique GDAL ``/vsimem/`` path.

    Mirrors :func:`pyramids.feature._ogr._new_vsimem_path` but takes an
    arbitrary extension so the same scheme can back rasters, NetCDFs, or
    anything else. The ``<time_ns>_<counter>`` body is collision-proof
    within a single process run: the strictly increasing counter
    guarantees uniqueness even when ``time.time_ns()`` repeats within a
    clock tick.

    Args:
        suffix: Extension to append (including the leading dot). Used by
            GDAL as a driver hint when the in-memory bytes have no magic
            header. Defaults to ``".tif"``.

    Returns:
        str: A ``/vsimem/<time>_<counter><suffix>`` path.

    Examples:
        - A path with no explicit suffix lives under ``/vsimem/`` and ends in ``.tif``:
            ```python
            >>> from pyramids._io import new_vsimem_path
            >>> p = new_vsimem_path()
            >>> p.startswith("/vsimem/")
            True
            >>> p.endswith(".tif")
            True

            ```
        - A custom extension is appended verbatim (handy as a GDAL driver hint):
            ```python
            >>> from pyramids._io import new_vsimem_path
            >>> new_vsimem_path(".nc").endswith(".nc")
            True

            ```
        - Two calls never collide, so concurrent conversions stay isolated:
            ```python
            >>> from pyramids._io import new_vsimem_path
            >>> new_vsimem_path() != new_vsimem_path()
            True

            ```

    See Also:
        bytes_to_gdal: Uses this to back an in-memory dataset.
        silent_unlink: Removes the path once the dataset is gone.
    """
    return f"/vsimem/{time.time_ns()}_{next(_VSIMEM_COUNTER)}{suffix}"


def silent_unlink(path: str) -> None:
    """:func:`osgeo.gdal.Unlink` that never raises.

    Safe to register with :func:`weakref.finalize` — under
    ``gdal.UseExceptions()`` an ``Unlink`` on a path that is already gone
    raises ``RuntimeError``, which would surface as an "Exception ignored
    in" message during garbage collection. Swallowing it keeps cleanup
    quiet and idempotent.

    Args:
        path: The ``/vsimem/`` (or other VSI) path to remove.

    Examples:
        - An existing ``/vsimem/`` file is removed:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids._io import new_vsimem_path, silent_unlink
            >>> path = new_vsimem_path()
            >>> _ = gdal.FileFromMemBuffer(path, b"hello")
            >>> gdal.VSIStatL(path) is not None
            True
            >>> silent_unlink(path)
            >>> gdal.VSIStatL(path) is None
            True

            ```
        - Unlinking a path that does not exist is a quiet no-op (safe inside
          :func:`weakref.finalize`):
            ```python
            >>> from pyramids._io import silent_unlink
            >>> silent_unlink("/vsimem/this-path-never-existed.tif")

            ```

    See Also:
        new_vsimem_path: Mints the paths this cleans up.
    """
    try:
        gdal.Unlink(path)
    except Exception:  # pragma: no cover - cleanup must never raise
        pass


def read_vsi_bytes(path: str) -> bytes:
    """Read the full contents of a GDAL VSI file as bytes.

    The standard ``VSIFOpenL`` / seek-to-end / ``VSIFReadL`` dance used to pull
    an in-memory (``/vsimem/``) or other VSI file back into Python — shared by
    every ``to_*_bytes`` serializer so the read-back logic lives in one place.

    Args:
        path: The VSI path to read (e.g. a ``/vsimem/...`` path produced by
            :func:`new_vsimem_path`).

    Returns:
        bytes: The complete file contents.

    Raises:
        FileNotFoundError: The path cannot be opened (it does not exist or was
            already unlinked).
        OSError: A short read returned fewer than the file's byte count.

    Examples:
        - Write a buffer to ``/vsimem/`` and read it back:
            ```python
            >>> from osgeo import gdal
            >>> from pyramids._io import new_vsimem_path, read_vsi_bytes, silent_unlink
            >>> path = new_vsimem_path(".bin")
            >>> _ = gdal.FileFromMemBuffer(path, b"payload")
            >>> read_vsi_bytes(path)
            b'payload'
            >>> silent_unlink(path)

            ```
        - A missing path raises ``FileNotFoundError``:
            ```python
            >>> from pyramids._io import read_vsi_bytes
            >>> try:
            ...     read_vsi_bytes("/vsimem/never-written.bin")
            ... except FileNotFoundError as exc:
            ...     print("could not open" in str(exc))
            True

            ```

    See Also:
        new_vsimem_path: Mints unique ``/vsimem/`` paths to write into.
        silent_unlink: Removes the path once the bytes are extracted.
    """
    try:
        handle = gdal.VSIFOpenL(path, "rb")
    except RuntimeError:
        # Under gdal.UseExceptions() a missing path raises instead of
        # returning None; normalise both shapes to FileNotFoundError.
        handle = None
    if handle is None:
        raise FileNotFoundError(f"could not open VSI path {path!r} for reading.")
    try:
        gdal.VSIFSeekL(handle, 0, 2)
        size = gdal.VSIFTellL(handle)
        gdal.VSIFSeekL(handle, 0, 0)
        # VSIFReadL returns None (not b"") for a zero-byte read.
        data = gdal.VSIFReadL(1, size, handle) if size else b""
    finally:
        gdal.VSIFCloseL(handle)
    payload = bytes(data)
    if len(payload) != size:
        # VSIFReadL can return fewer than `size` bytes without raising; fail
        # loudly so a truncated serialization never returns a corrupt buffer.
        raise OSError(
            f"short read on VSI path {path!r}: expected {size} bytes, "
            f"got {len(payload)}."
        )
    return payload


def _is_zip(path: str):
    return path.endswith(".zip") or path.__contains__(".zip")


def _is_gzip(path: str):
    return path.endswith(".gz") or path.__contains__(".gz")


def _is_tar(path: str):
    return path.endswith(".tar.gz") or path.__contains__(".tar")


def _get_zip_path(path: str, file_i: int = 0):
    """Get Zip Path.

    Args:
        path (str): Path to the zip file.
        file_i (int): Index to the file inside the compressed file you want to read.

    Returns:
        str: Path for GDAL to read the zipped file.

    Examples:
        - Internal Zip file path (one/multiple files inside the compressed file): if the path contains a zip but does not end with zip (compressed-file-name.zip/1.asc), so the path contains the internal path inside the zip file, so just add the prefix

          ```python
          >>> rdir = "tests/data/virtual-file-system"
          >>> path = _get_zip_path(f"{rdir}/multiple_compressed_files.zip/1.asc")
          >>> print(path)
          "/vsizip/tests/data/virtual-file-system/multiple_compressed_files.zip/1.asc"

          ```

        - Only the Zip file path (one/multiple files inside the compressed file): If you provide the name of the zip file with multiple files inside it, it will return the path to the first file.

          ```python
          >>> path = _get_zip_path(f"{rdir}/multiple_compressed_files.zip")
          >>> print(path)
          "/vsizip/tests/data/virtual-file-system/multiple_compressed_files.zip/1.asc"

          ```

        - Zip file path and an index (one/multiple files inside the compressed file): if you provide the path to the zip file and an index to the file inside the compressed file you want to read

          ```python
          >>> path = _get_zip_path("compressed-file-name.zip", file_i=1)
          >>> print(path)
          "/vsizip/tests/data/virtual-file-system/multiple_compressed_files.zip/2.asc"

          ```
    """
    # get a list of files inside the compressed file
    if path.__contains__(".zip") and not path.endswith(".zip"):
        vsi_path = f"/vsizip/{path}"
    else:
        file_list = zipfile.ZipFile(path).namelist()
        vsi_path = f"/vsizip/{path}/{file_list[file_i]}"
    return vsi_path


def _get_gzip_path(path: str, file_i: int = 0):
    """Get Zip Path.

    - Check if the given path contains a.gz in it.
    - If the path contains a gz but does not end with gz (xxxx.gz/1.asc), so the path contains the internal path inside the gz file, so just add the prefix.
    - Anything else just add the prefix.

    Args:
        path (str): Path to the zip file.

    Returns:
        str: Path for GDAL to read the zipped file.
    """
    # get list of files inside the compressed file
    warnings.warn(
        "gzip compressed files does not support getting internal file list, if the compressed file contains more than "
        "one file error will be given, you have to provide the internal path (i.e. "
        "path/file-name.gz/internal-file.ext)"
    )
    if path.__contains__(".gz") and not path.endswith(".gz"):
        vsi_path = f"/vsigzip/{path}"
    else:
        try:
            with tarfile.open(path) as tf:
                file_list = tf.getnames()
            vsi_path = f"/vsigzip/{path}/{file_list[file_i]}"
        except tarfile.ReadError:
            # if the tarfile.open() does not give a getnames() method, it means the file contains one file
            # so return the path of the main file
            vsi_path = f"/vsigzip/{path}"
    return vsi_path


def _get_tar_path(path: str):
    """Get Zip Path.

    - Check if the given path contains a.tar in it.
    - If the path contains a.tar but does not end with.tar (xxxx.tar/1.asc), so the path contains the internal path inside the tar file, so just add the prefix.
    - Otherwise, just add the prefix.

    Args:
        path (str): Path to the tar file.

    Returns:
        str: Path for GDAL to read the tar file.
    """
    # get list of files inside the compressed file
    vsi_path = f"/vsitar/{path}"
    return vsi_path


def _parse_path(path: str | Path, file_i: int = 0) -> str:
    """Parse Path.

    Args:
        path (str | Path): Path to the file.
        file_i (int): Index to the file inside the compressed file you want to read. If the compressed file has only one file inside, it will read this file; if multiple files are compressed, it will return the first file.

    Returns:
        str: Path to the file to read.
    """
    # Convert to str because the helpers build GDAL virtual filesystem strings
    # (/vsizip/, /vsigzip/, /vsitar/) which are not real filesystem paths.
    path = str(path)
    # Rewrite URL-scheme paths (s3://, gs://, az://, http(s)://, file://)
    # to GDAL /vsi* form BEFORE zip/tar/gzip detection so a remote /vsicurl/
    # path doesn't accidentally get treated as a compressed archive.
    path = remote._to_vsi(path)
    if remote.is_remote(path):
        new_path = path
    elif _is_zip(path):
        new_path = _get_zip_path(path, file_i=file_i)
    elif _is_tar(path):
        new_path = _get_tar_path(path)
    elif _is_gzip(path):
        new_path = _get_gzip_path(path, file_i=file_i)
    else:
        new_path = path
    return str(new_path)


def _infer_archive_kind(path: str) -> str | None:
    """Best-effort archive kind (``"zip"`` / ``"tar"`` / ``"gzip"``) from a path.

    Returns ``None`` when the extension is not a recognised archive type — e.g.
    an extension-less download URL, in which case the caller must pass an
    explicit ``kind``.

    Args:
        path: Path or URL to sniff.

    Returns:
        str | None: ``"zip"``, ``"tar"``, ``"gzip"``, or ``None``.
    """
    result: str | None
    if _is_zip(path):
        result = "zip"
    elif _is_tar(path):
        result = "tar"
    elif _is_gzip(path):
        result = "gzip"
    else:
        result = None
    return result


def _archive_dir_vsi(path: str | Path, kind: str = "auto") -> str:
    """Return the GDAL ``/vsizip|tar|gzip/`` *directory* path for an archive.

    Unlike :func:`_parse_path` (which, for a bare ``.zip``, resolves to the
    *first member*), this returns the archive directory itself so callers can
    :func:`osgeo.gdal.ReadDir` it. The path is first normalised to ``/vsi*``
    form via :func:`pyramids.base.remote._to_vsi` — so ``s3://`` / ``https://``
    URLs work — and then the archive handler prefix is prepended:

    * ``_archive_dir_vsi("https://h/x.zip", "zip")`` →
      ``/vsizip//vsicurl/https://h/x.zip``
    * ``_archive_dir_vsi("/data/x.zip")`` (kind inferred) → ``/vsizip//data/x.zip``
    * ``_archive_dir_vsi("/vsizip//data/x.zip", "zip")`` → unchanged (idempotent)

    Args:
        path: Path or URL of the archive. May be extension-less (e.g. an Earth
            Engine ``getDownloadURL`` ZIP whose URL ends in ``:getPixels``) — in
            that case ``kind`` must be given explicitly.
        kind: One of ``"zip"``, ``"tar"`` (also ``"tar.gz"`` / ``"tgz"``),
            ``"gzip"`` (also ``"gz"``), or ``"auto"`` (default) to infer from the
            extension.

    Returns:
        str: The ``/vsi*`` directory path.

    Raises:
        FileFormatNotSupportedError: ``kind="auto"`` and the extension is not a
            recognised archive type.
        ValueError: ``kind`` is not a recognised archive kind.
    """
    path = str(path)
    if kind == "auto":
        inferred = _infer_archive_kind(path)
        if inferred is None:
            raise FileFormatNotSupportedError(
                f"could not infer the archive kind from {path!r}; pass kind='zip', "
                "'tar', or 'gzip' explicitly (needed for extension-less URLs)"
            )
        kind = inferred
    prefix = _VSI_ARCHIVE_KINDS.get(kind)
    if prefix is None:
        raise ValueError(
            f"unknown archive kind {kind!r}; expected one of "
            f"{sorted(_VSI_ARCHIVE_KINDS)} or 'auto'"
        )
    vsi_path = remote._to_vsi(path)
    if vsi_path.startswith(("/vsizip/", "/vsitar/", "/vsigzip/")):
        return vsi_path
    return f"{prefix}{vsi_path}"


def _archive_members(dir_vsi: str, member_glob: str = "*") -> list[str]:
    """List a ``/vsi*`` archive directory's members, filtered and sorted.

    Only top-level members are returned — recursive listing of nested
    directories inside the archive is not supported.

    Args:
        dir_vsi: An archive directory path as returned by :func:`_archive_dir_vsi`.
        member_glob: :mod:`fnmatch` pattern; only matching members are returned.
            Default ``"*"`` (all). Pass e.g. ``"*.tif"`` for an archive that also
            contains sidecar files.

    Returns:
        list[str]: Member names (not full paths), sorted.

    Raises:
        FileFormatNotSupportedError: GDAL could not list the archive (unreachable
            path/URL, not an archive of that kind, …).
        FileNotFoundError: No member matched ``member_glob``.
    """
    entries = gdal.ReadDir(dir_vsi)
    if entries is None:
        raise FileFormatNotSupportedError(
            f"could not list archive members at {dir_vsi!r}; GDAL's archive handlers "
            "need a recognised extension (.zip / .tar / .tar.gz / .gz) on the file "
            "name, and nested archives are not supported. See "
            "DatasetCollection.from_archive's docstring for the extension-less-URL "
            "workaround (write bytes to '/vsimem/<name>.zip' first)."
        )
    listed = sorted(e for e in entries if e not in (".", ".."))
    members = [e for e in listed if fnmatch.fnmatch(e, member_glob)]
    if not members:
        preview = listed[:10]
        more = (
            ""
            if len(listed) <= len(preview)
            else f" (showing {len(preview)} of {len(listed)})"
        )
        raise FileNotFoundError(
            f"no members matching {member_glob!r} in {dir_vsi!r}; "
            f"available: {preview}{more}"
        )
    return members


# Public aliases for the archive-listing helpers above, exposed as supported entry
# points without the leading underscore. The underscore names are kept for the internal
# callers (dataset.collection, dataset.dataset).
archive_dir_vsi = _archive_dir_vsi
archive_members = _archive_members


def extract_from_gz(input_file: str | Path, output_file: str | Path, delete=False):
    """Extract data from zip/.gz files and save the data.

    Args:
        input_file (str): Zipped file name.
        output_file (str): Path where the unzipped data must be stored.
        delete (bool): True to delete the zipped file after extracting the data.

    Returns:
        None
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    with gzip.GzipFile(input_file, "rb") as zf:
        content = zf.read()
        with open(output_file, "wb") as save_file_content:
            save_file_content.write(content)

    if delete:
        input_file.unlink()


def read_file(
    path: str | Path,
    read_only: bool = True,
    open_as_multi_dimensional: bool = False,
    file_i: int = 0,
    *,
    vsi: str | None = None,
):
    """Open file (GeoTIFF and ASCII).

    - For GeoTIFF and ASCII files.

    Args:
        path (str): Path of file to open (works for ASCII, GeoTIFF).
        read_only (bool): File mode; set to False to open in "update" mode.
        open_as_multi_dimensional (bool): If True, opens using OF_MULTIDIM_RASTER for multi-dimensional formats. Default is False.
        file_i (int): Index to the file inside the compressed file you want to read (default 0). If the compressed file has only one file, the first file is used.
        vsi (str | None): When given, treat ``path`` as an archive of this kind
            (``"zip"`` / ``"tar"`` / ``"gzip"`` / ``"auto"``) and open member
            ``file_i`` from inside it — even when the path/URL has no archive
            extension (e.g. an Earth Engine download URL). Default ``None``
            (path opened directly / extension-sniffed as today).

    Returns:
        gdal.Dataset: Opened dataset.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(
            f"the path parameter should be of string or Path type, given: {type(path)}"
        )
    if vsi is not None:
        dir_vsi = _archive_dir_vsi(path, vsi)
        members = _archive_members(dir_vsi)
        if not 0 <= file_i < len(members):
            raise FileNotFoundError(
                f"archive {path!r} has {len(members)} member(s); file_i={file_i} "
                "is out of range"
            )
        path = f"{dir_vsi}/{members[file_i]}"
    else:
        path = _parse_path(path, file_i=file_i)
    access = gdal.GA_ReadOnly if read_only else gdal.GA_Update
    try:
        # get the file extension
        # Example criteria for using gdal.OpenEx with OF_MULTIDIM_RASTER for complex multi-dimensional formats
        if (
            open_as_multi_dimensional
        ):  # file_extension in ["hdf", "h5", "nc", "nc4", "grib", "grib2", "jp2"]:
            # Use OpenEx with the OF_MULTIDIM_RASTER flag for formats that often require handling of multi-dimensional
            # data
            src = gdal.OpenEx(path, access | gdal.OF_MULTIDIM_RASTER)
        else:
            # Use OpenShared for potentially frequently accessed raster files
            src = gdal.OpenShared(path, access)
    except Exception as e:
        if str(e).__contains__(" not recognized as a supported file format."):
            if any(path.endswith(i) for i in COMPRESSED_FILES_EXTENSIONS):
                raise FileFormatNotSupportedError(
                    "File format is not supported if you provided a gzip compressed file with multiple internal "
                    "files. Currently, it is not supported to read gzip files with multiple compressed internal "
                    "files"
                )
            else:
                raise e
        elif any(path.__contains__(i) for i in DOES_NOT_SUPPORT_INTERNAL) and not any(
            path.endswith(i) for i in DOES_NOT_SUPPORT_INTERNAL
        ):
            raise FileFormatNotSupportedError(
                "File format is not supported, if you provided a gzip/7z compressed file with multiple internal "
                "files. Currently it is not supported to read gzip/7z files with multiple compressed internal "
                "files"
            )
        elif str(e).__contains__(" No such file or directory"):
            raise FileNotFoundError(f"{path} you entered does not exist")
        else:
            raise e
    # if src is None:
    #     raise ValueError(
    #         f"The raster path: {path} you enter gives a None gdal Object check the read premission, maybe "
    #         f"the raster is being used by other software"
    #     )
    return src


def bytes_to_gdal(
    data: bytes | bytearray | memoryview,
    *,
    suffix: str = ".tif",
    read_only: bool = True,
    open_as_multi_dimensional: bool = False,
) -> tuple[gdal.Dataset, str]:
    """Open an in-memory byte string as a GDAL dataset via ``/vsimem/``.

    Writes ``data`` to a fresh ``/vsimem/`` path with
    :func:`osgeo.gdal.FileFromMemBuffer`, then opens it through
    :func:`read_file`. On any failure the ``/vsimem/`` entry is removed
    before the error propagates, so a bad payload never leaks an
    in-memory file. On success the caller owns the returned path and is
    responsible for calling :func:`silent_unlink` (typically via
    :func:`weakref.finalize`) when the dataset is no longer needed.

    Args:
        data: Raw bytes of a raster (GeoTIFF, NetCDF, ASCII grid, ...).
        suffix: Extension hint for GDAL's driver detection. Needed only
            for headerless formats (ESRI ASCII grid); for anything with
            a magic header GDAL sniffs the format regardless. Defaults
            to ``".tif"``.
        read_only: Open the dataset read-only. ``/vsimem/`` files are
            always writable at the GDAL level; pyramids enforces the
            access flag itself. Defaults to ``True``.
        open_as_multi_dimensional: Pass ``gdal.OF_MULTIDIM_RASTER`` (used
            by :class:`~pyramids.netcdf.NetCDF`). Defaults to ``False``.

    Returns:
        tuple[gdal.Dataset, str]: The opened GDAL dataset and the
        ``/vsimem/`` path backing it.

    Raises:
        TypeError: ``data`` is not a bytes-like object.
        ValueError: GDAL could not open the bytes as a dataset.

    Examples:
        - Open the bytes of a GeoTIFF and inspect the GDAL dataset, then clean up:
            ```python
            >>> from pathlib import Path
            >>> from pyramids._io import bytes_to_gdal, silent_unlink
            >>> data = Path("tests/data/acc4000.tif").read_bytes()
            >>> src, vsi_path = bytes_to_gdal(data)
            >>> src.RasterCount
            1
            >>> (src.RasterXSize, src.RasterYSize)
            (14, 13)
            >>> vsi_path.startswith("/vsimem/")
            True
            >>> src = None
            >>> silent_unlink(vsi_path)

            ```
        - Non bytes-like input is rejected before any ``/vsimem/`` file is written:
            ```python
            >>> from pyramids._io import bytes_to_gdal
            >>> try:
            ...     bytes_to_gdal("a string, not bytes")
            ... except TypeError as exc:
            ...     print("bytes-like" in str(exc))
            True

            ```
        - Bytes GDAL cannot parse raise ``ValueError`` (and leak nothing):
            ```python
            >>> from pyramids._io import bytes_to_gdal
            >>> try:
            ...     bytes_to_gdal(b"definitely not a raster")
            ... except ValueError as exc:
            ...     print("suffix" in str(exc))
            True

            ```

    See Also:
        new_vsimem_path: Mints the backing path.
        silent_unlink: How the caller releases the returned path.
        pyramids.dataset.Dataset.from_bytes: The public wrapper around this helper.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"data should be a bytes-like object (bytes/bytearray/memoryview), "
            f"given: {type(data)}"
        )
    vsi_path = new_vsimem_path(suffix)
    gdal.FileFromMemBuffer(vsi_path, bytes(data))
    try:
        src = read_file(
            vsi_path,
            read_only=read_only,
            open_as_multi_dimensional=open_as_multi_dimensional,
        )
    except Exception as e:
        silent_unlink(vsi_path)
        raise ValueError(
            "could not open the supplied bytes as a raster dataset; if the "
            "format has no magic header (e.g. ESRI ASCII grid) pass an "
            f"explicit `suffix=` hint. Underlying error: {e}"
        ) from e
    if src is None:  # pragma: no cover - gdal.UseExceptions() makes this unreachable
        silent_unlink(vsi_path)
        raise ValueError(
            "could not open the supplied bytes as a raster dataset; if the "
            "format has no magic header (e.g. ESRI ASCII grid) pass an "
            "explicit `suffix=` hint."
        )
    return src, vsi_path


def insert_space(inp):
    """Insert space between the ascii file values."""
    return str(inp) + "  "


def to_ascii(
    arr: np.ndarray, cell_size: float, xmin, ymin, no_data_value, path: str | Path
) -> None:
    """Write raster into ASCII file.

    Writes the raster to disk in ASCII format.

    Args:
        arr (np.ndarray): Array you want to write to disk.
        cell_size (int): Cell size.
        xmin (float): X coordinate of the lower left corner.
        ymin (float): Y coordinate of the lower left corner.
        no_data_value (numeric): No data value.
        path (str): Name of the ASCII file to create; should include the extension ".asc".

    Returns:
        None
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(
            f"path input should be string or Path type, given: {type(path)}"
        )

    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"There is a file with the same path you have provided: {path}"
        )
    rows = arr.shape[0]
    columns = arr.shape[1]
    # y_lower_side = geotransform[3] - rows * cell_size
    # write the the ASCII file details
    with open(path, "w") as file:
        file.write("ncols         " + str(columns) + "\n")
        file.write("nrows         " + str(rows) + "\n")
        file.write("xllcorner     " + str(xmin) + "\n")
        file.write("yllcorner     " + str(ymin) + "\n")
        file.write("cellsize      " + str(cell_size) + "\n")
        file.write("NODATA_value  " + str(no_data_value) + "\n")
        # write the array
        for i in range(rows):
            file.writelines(list(map(insert_space, arr[i, :])))
            file.write("\n")
