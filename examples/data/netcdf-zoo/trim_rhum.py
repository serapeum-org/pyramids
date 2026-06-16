"""Trim the COARDS ``rhum`` example to its first 100 time steps, in place.

The published ``rhum.2003.nc`` holds 365 daily steps (~61 MB). For a fixture, 100 steps
(~17 MB) is plenty: the structural signature is unchanged (four 1-D coordinate axes plus one
4-D packed data variable), only the file is smaller.

Done with GDAL's multidimensional layer (the layer pyramids is built on) rather than
``MultiDimTranslate``, which renames the subset dimension and rewrites ``Conventions``. This
controlled copy preserves every name, dimension type/direction, attribute, and the ``int16``
``scale_factor``/``add_offset``/``_FillValue`` packing of ``rhum``.

Run with the pyramids ``dev`` environment, e.g. ``pixi run -e dev python trim_rhum.py``.
"""

from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()

HERE = Path(__file__).resolve().parent
TARGET = HERE / "coards__5v__1d4-4d1.nc"
N = 100

_INT_TYPES = {gdal.GDT_Byte, gdal.GDT_Int16, gdal.GDT_UInt16, gdal.GDT_Int32,
              gdal.GDT_UInt32, gdal.GDT_Int64, gdal.GDT_UInt64}
if hasattr(gdal, "GDT_Int8"):
    _INT_TYPES.add(gdal.GDT_Int8)


def _copy_attrs(s, d):
    for at in s.GetAttributes():
        name, dt, cnt = at.GetName(), at.GetDataType(), at.GetTotalElementsCount()
        if dt.GetClass() == gdal.GEDTC_STRING:
            na = d.CreateAttribute(name, [] if cnt <= 1 else [cnt], gdal.ExtendedDataType.CreateString())
            na.WriteString(at.ReadAsString()) if cnt <= 1 else na.WriteStringArray(at.ReadAsStringArray())
            continue
        numtype = dt.GetNumericDataType()
        is_int = numtype in _INT_TYPES
        na = d.CreateAttribute(name, [] if cnt <= 1 else [cnt], gdal.ExtendedDataType.Create(numtype))
        if cnt <= 1:
            na.WriteInt(at.ReadAsInt()) if is_int else na.WriteDouble(at.ReadAsDouble())
        else:
            na.WriteIntArray(at.ReadAsIntArray()) if is_int else na.WriteDoubleArray(at.ReadAsDoubleArray())


def main() -> None:
    src = gdal.OpenEx(str(TARGET), gdal.OF_MULTIDIM_RASTER)
    rg = src.GetRootGroup()
    tmp = TARGET.with_name("_trim_tmp.nc")
    if tmp.exists():
        tmp.unlink()
    dst = gdal.GetDriverByName("netCDF").CreateMultiDimensional(str(tmp))
    drg = dst.GetRootGroup()

    _copy_attrs(rg, drg)
    sizes = {d.GetName(): d.GetSize() for d in rg.GetDimensions()}
    ddims = {d.GetName(): drg.CreateDimension(d.GetName(), d.GetType(), d.GetDirection(),
                                              N if d.GetName() == "time" else d.GetSize())
             for d in rg.GetDimensions()}

    for nm in ["lon", "lat", "level", "time"]:
        s = rg.OpenMDArray(nm)
        arr = s.ReadAsArray()[:N] if nm == "time" else s.ReadAsArray()
        a = drg.CreateMDArray(nm, [ddims[nm]], gdal.ExtendedDataType.Create(s.GetDataType().GetNumericDataType()))
        if s.GetUnit():
            a.SetUnit(s.GetUnit())
        _copy_attrs(s, a)
        a.Write(arr)

    s = rg.OpenMDArray("rhum")
    data = s.ReadAsArray(array_start_idx=[0, 0, 0, 0], count=[N, sizes["level"], sizes["lat"], sizes["lon"]])
    a = drg.CreateMDArray("rhum", [ddims["time"], ddims["level"], ddims["lat"], ddims["lon"]],
                          gdal.ExtendedDataType.Create(s.GetDataType().GetNumericDataType()))
    if s.GetScale() is not None:
        a.SetScale(s.GetScale())
    if s.GetOffset() is not None:
        a.SetOffset(s.GetOffset())
    if s.GetUnit():
        a.SetUnit(s.GetUnit())
    ndv = s.GetNoDataValueAsDouble()
    if ndv is not None:
        a.SetNoDataValueDouble(ndv)
    _copy_attrs(s, a)
    a.Write(data)

    dst = src = None
    tmp.replace(TARGET)
    print(f"trimmed {TARGET.name} to {N} time steps")


if __name__ == "__main__":
    main()
