"""L3: PEP-562 ``__getattr__`` hook on :mod:`pyramids.feature`.

The module exposes ``LazyFeatureCollection`` via a package-level
``__getattr__`` rather than a plain ``None`` sentinel. Pin the two
behaviours this buys us:

1. On a full install (dask-geopandas present), the symbol resolves to
   the real class.
2. On a minimal install, ``from pyramids.feature import
   LazyFeatureCollection`` raises a branded :class:`ImportError` with
   the install hint. Library authors who need to dispatch across both
   minimal and full installs guard with ``try/except ImportError`` or
   read the ``_HAS_DASK_GEOPANDAS`` flag directly (``hasattr`` does
   NOT catch ``ImportError`` in Python 3 — it only catches
   ``AttributeError`` — so the raising ``__getattr__`` surfaces the
   error at the access site, which is the intended UX).

The minimal-install path is simulated by patching
:data:`pyramids.feature._HAS_DASK_GEOPANDAS` so the test runs regardless
of which extras are installed in the dev env.
"""

from __future__ import annotations

import pytest
from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask_geopandas

pytestmark = pytest.mark.parquet_lazy

try:
    import_dask_geopandas("dask-geopandas not installed")
    import dask_geopandas
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK_GP = False
else:
    HAS_DASK_GP = True
class TestGetattrHookFullInstall:
    """When ``dask-geopandas`` is installed, the symbol resolves normally."""

    @pytest.mark.skipif(not HAS_DASK_GP, reason="dask-geopandas not installed")
    def test_from_import_returns_class(self):
        """``from pyramids.feature import LazyFeatureCollection`` works."""
        from pyramids.feature import LazyFeatureCollection

        assert LazyFeatureCollection is not None
        assert LazyFeatureCollection.__name__ == "LazyFeatureCollection"


class TestGetattrHookMinimalInstall:
    """Simulate a minimal install by patching the feat-detection flag."""

    def test_from_import_raises_branded_importerror(self, monkeypatch):
        """L3: minimal install → branded :class:`ImportError` with hint.

        Test scenario:
            With ``_HAS_DASK_GEOPANDAS`` flipped to False, evaluating
            ``pyramids.feature.LazyFeatureCollection`` must raise
            :class:`ImportError` naming the ``[parquet-lazy]`` extra
            so users see an actionable install hint rather than a
            ``TypeError: isinstance() arg 2 must be a type`` the
            old ``None`` sentinel produced downstream.
        """
        import pyramids.feature as pf

        monkeypatch.setattr(pf, "_HAS_DASK_GEOPANDAS", False)
        with pytest.raises(ImportError, match=r"pyramids-gis\[parquet-lazy\]"):
            pf.LazyFeatureCollection  # noqa: B018 - attribute access IS the action

    def test_try_except_is_the_correct_guard(self, monkeypatch):
        """Library-author pattern: ``try/except ImportError`` around the import.

        Test scenario:
            Library authors writing dispatch code across both minimal
            and full installs wrap the import in try/except and fall
            back to ``None``. The test mirrors that pattern to pin it
            as the canonical guard documented in ``lazy-vector.md``.
        """
        import pyramids.feature as pf

        monkeypatch.setattr(pf, "_HAS_DASK_GEOPANDAS", False)
        try:
            lazy_cls = pf.LazyFeatureCollection
        except ImportError:
            lazy_cls = None
        assert lazy_cls is None

    def test_unknown_attribute_still_raises_attributeerror(self, monkeypatch):
        """Only ``LazyFeatureCollection`` is handled; other misses raise AttributeError.

        Test scenario:
            The ``__getattr__`` hook must not swallow every unknown
            attribute lookup — only the one name we own. Anything else
            must still surface a normal ``AttributeError`` so typos
            fail loudly.
        """
        import pyramids.feature as pf

        monkeypatch.setattr(pf, "_HAS_DASK_GEOPANDAS", False)
        with pytest.raises(AttributeError, match="does_not_exist"):
            pf.does_not_exist  # noqa: B018


class TestFeatureCollectionStillWorks:
    """Pin that :class:`FeatureCollection` is unaffected by the hook."""

    def test_feature_collection_import_still_works(self):
        """Baseline: the eager class imports normally on all installs."""
        from pyramids.feature import FeatureCollection

        assert FeatureCollection.__name__ == "FeatureCollection"
