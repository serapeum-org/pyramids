"""I/O helpers: resource format sniffing and dispatch (CKAN/HDX-style)."""

from __future__ import annotations

# `load_resource` / `sniff_format` live in `pyramids.io.sniff`, which imports the
# Dataset, FeatureCollection and NetCDF readers (and, transitively, geopandas).
# Expose them lazily via PEP 562 module `__getattr__` so a bare `import
# pyramids.io` stays light — that heavy stack is only pulled in when these
# symbols are first accessed, mirroring what `pyramids/__init__.py` already does
# for `read_resource` / `sniff_kind`.
_LAZY_SNIFF_EXPORTS = frozenset({"load_resource", "sniff_format", "sniff_magic"})
# `sniff` itself is resolved lazily too: the eager `from .sniff import ...` used
# to register the submodule as an attribute of this package, so `import
# pyramids.io; pyramids.io.sniff` worked. Dropping the eager import would break
# that without this entry.
_LAZY_NAMES = _LAZY_SNIFF_EXPORTS | {"sniff"}


def __getattr__(name: str):
    """Lazily import the sniff submodule and its exports on first access (PEP 562)."""
    if name in _LAZY_NAMES:
        import pyramids.io.sniff as sniff

        globals().update(
            sniff=sniff,
            load_resource=sniff.load_resource,
            sniff_format=sniff.sniff_format,
            sniff_magic=sniff.sniff_magic,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-exported names in ``dir(pyramids.io)``."""
    return sorted(set(globals()) | _LAZY_NAMES)


__all__ = ["load_resource", "sniff_format", "sniff_magic"]
