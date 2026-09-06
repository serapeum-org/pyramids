"""Guard the public-surface promises the consolidation is easy to break by accident.

Moving a helper down into ``pyramids.base`` is invisible to callers only while the name it used to
live under keeps resolving *and still names the same object*, and adding an ``__all__`` to a module
that never had one silently *narrows* its star-import and its mkdocstrings page. Neither shows up in
a test that exercises behaviour, because both are about which names a module offers, not what they
do.
"""

import inspect

import pyramids.base._bbox as base_bbox
import pyramids.dataset.dataset as dataset_module
import pyramids.feature as feature_package
import pyramids.feature.bbox as feature_bbox
import pyramids.netcdf.utils as netcdf_utils
from pyramids.base._utils import DTYPE_CONVERSION_DF


def _public_functions_defined_here(module) -> list[str]:
    """Names of the public functions ``module`` defines itself (not the ones it imports)."""
    return sorted(
        name
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and inspect.isfunction(obj)
        and obj.__module__ == module.__name__
    )


def _public_names_defined_here(module) -> list[str]:
    """Every public name ``module`` defines itself, whatever kind of object it is.

    Functions are the obvious half and the easy half. A module's API is also its type
    aliases and its constants, and neither is a function, so a function-only sweep reports
    a complete ``__all__`` while the page it governs is missing them. That is exactly how
    ``AttributeValue`` — which appears in ``NetCDFVariable``'s own signature — and
    ``CF_NODATA_KEYS`` were dropped.

    Anything imported from elsewhere is skipped: re-exporting is a decision, and this guard
    is about names the module owns. Functions and classes carry ``__module__``; a bare
    alias or constant does not, so it is taken as defined here — the safe direction, since
    a false positive is a name to justify and a false negative is a name silently
    withdrawn.

    Args:
        module: The module to inspect.

    Returns:
        list[str]: The public names, sorted.
    """
    names = []
    for name, obj in vars(module).items():
        if name.startswith("_") or inspect.ismodule(obj):
            continue
        owner = getattr(obj, "__module__", None)
        if owner is None or owner == module.__name__:
            names.append(name)
    return sorted(names)


class TestNetCDFUtilsAdvertisesEveryPublicNameItDefines:
    """``pyramids.netcdf.utils.__all__`` must not shrink the module's own API."""

    def test_all_lists_every_public_function_defined_in_the_module(self):
        """A public helper defined here is part of the module's API, so ``__all__`` names it.

        Test scenario:
            The module had no ``__all__`` until this branch added one for the CF time
            helpers. Once the list exists it becomes the module's declared surface: a
            star-import stops offering anything left out, and mkdocstrings' public-member
            detection can drop it from the rendered page. Three helpers that other
            pyramids modules import by name — the full-name resolver, the time-conversion
            factory and the CF attribute reader — were not on it.
        """
        missing = sorted(
            set(_public_functions_defined_here(netcdf_utils))
            - set(netcdf_utils.__all__)
        )

        assert not missing, (
            "public functions defined in pyramids.netcdf.utils but absent from its "
            f"__all__: {missing}"
        )

    def test_all_lists_every_public_name_defined_in_the_module(self):
        """A type alias and a constant are API too, and the first sweep missed both.

        Test scenario:
            The function-only check above passed while the module's three attribute type
            aliases and ``CF_NODATA_KEYS`` were still absent from ``__all__`` — and
            griffe, which mkdocstrings renders this module through, reports ``__all__``
            as the module's exports and calls exactly those members public. So the docs
            page silently lost four names, one of them (``AttributeValue``) part of
            ``NetCDFVariable``'s published signature. Sweeping every public object the
            module defines, not only its functions, is what catches that.
        """
        missing = sorted(
            set(_public_names_defined_here(netcdf_utils)) - set(netcdf_utils.__all__)
        )

        assert not missing, (
            "public names defined in pyramids.netcdf.utils but absent from its "
            f"__all__: {missing}"
        )

    def test_every_advertised_name_resolves(self):
        """Nothing in ``__all__`` is a dangling name.

        Test scenario:
            A star-import of the module, or a static analyser reading ``__all__``, must
            not see a name the module does not actually carry — the failure mode of
            hand-maintaining the list in the other direction.
        """
        dangling = [
            name for name in netcdf_utils.__all__ if not hasattr(netcdf_utils, name)
        ]

        assert not dangling, f"names in __all__ with no attribute: {dangling}"


class TestTheDtypeTableStaysImportableFromItsOldHome:
    """``pyramids.dataset.dataset.DTYPE_CONVERSION_DF`` is a name downstream code can hold."""

    def test_the_table_is_still_reachable_through_the_dataset_module(self):
        """The public module keeps offering the name the table moved away from.

        Test scenario:
            The dtype catalogue moved down to ``pyramids.base._utils`` so ``base`` would
            stop reaching up. ``pyramids.dataset.dataset`` is a public module and the
            table was a public name in it, so ``from pyramids.dataset.dataset import
            DTYPE_CONVERSION_DF`` used to work; dropping it is a break no caller was
            warned about, and re-exporting costs one import line.
        """
        assert hasattr(dataset_module, "DTYPE_CONVERSION_DF"), (
            "pyramids.dataset.dataset must keep re-exporting DTYPE_CONVERSION_DF"
        )

    def test_the_re_export_is_the_same_object_as_the_definition(self):
        """The re-export is an alias, not a second copy of the catalogue.

        Test scenario:
            Two tables that drift apart would be worse than one missing name: a caller
            reading the dataset module's copy and a caller reading ``base._utils``' copy
            would disagree about which dtypes exist.
        """
        assert dataset_module.DTYPE_CONVERSION_DF is DTYPE_CONVERSION_DF, (
            "the re-export must alias pyramids.base._utils.DTYPE_CONVERSION_DF"
        )


class TestFeatureBboxStillOffersWhatMovedDownToBase:
    """``Bbox`` and ``transform`` left ``pyramids.feature.bbox``; the names must not have.

    The reprojection moved to ``pyramids.base._bbox`` so ``pyramids.base`` could stop importing
    ``pyramids.feature`` (and, with it, geopandas). ``pyramids.feature.bbox`` re-exports both names,
    which is the whole reason the move was safe for callers.

    The move also took ``pyproj.Transformer`` and ``pyramids.base.crs.crs_from_user_input`` out of
    that module's namespace. Both were incidental -- names imported to implement ``transform``,
    never listed in ``__all__`` and never documented -- so the removal is intended and is recorded
    in ``docs/migration.md`` rather than reverted. Their homes are ``pyproj`` and
    ``pyramids.base.crs``.
    """

    def test_the_re_exports_are_the_definitions_themselves(self):
        """``feature.bbox`` hands out ``base._bbox``'s objects, not lookalikes.

        Test scenario:
            A re-export that resolved to a *copy* would pass a plain ``hasattr`` check and
            still break callers: a second ``Bbox`` alias fails an ``is`` comparison, and a
            second ``transform`` would drift from the one ``base._coverage`` calls. Identity
            is the property that makes the move invisible.
        """
        assert feature_bbox.Bbox is base_bbox.Bbox, (
            "pyramids.feature.bbox.Bbox must be the alias pyramids.base._bbox defines"
        )
        assert feature_bbox.transform is base_bbox.transform, (
            "pyramids.feature.bbox.transform must be the function pyramids.base._bbox defines"
        )

    def test_every_advertised_name_resolves(self):
        """Nothing in ``feature.bbox.__all__`` is a dangling name.

        Test scenario:
            ``__all__`` gained ``Bbox`` and ``transform`` as re-exports of names the module no
            longer defines, so the list is now the only thing tying the advertised surface to
            the imports at the top of the file. Dropping one of those imports would leave a
            star-import raising ``AttributeError``.
        """
        dangling = [
            name for name in feature_bbox.__all__ if not hasattr(feature_bbox, name)
        ]

        assert not dangling, f"names in __all__ with no attribute: {dangling}"

    def test_all_lists_every_public_function_defined_in_the_module(self):
        """A public helper defined here is part of the module's API, so ``__all__`` names it.

        Test scenario:
            The same failure mode the netcdf ``__all__`` had: a hand-maintained list that
            silently narrows the module when a new public helper is added beside the ones
            already on it.
        """
        missing = sorted(
            set(_public_functions_defined_here(feature_bbox))
            - set(feature_bbox.__all__)
        )

        assert not missing, (
            "public functions defined in pyramids.feature.bbox but absent from its "
            f"__all__: {missing}"
        )


class TestAnImplementationImportDoesNotWidenTheFeaturePackage:
    """``pyramids.feature`` borrows helpers from ``base._utils``; none of them is feature API."""

    def test_no_borrowed_helper_is_advertised_by_the_package(self):
        """Nothing imported from ``pyramids.base._utils`` appears in ``pyramids.feature.__all__``.

        Test scenario:
            Composing the ``LazyFeatureCollection`` install hint through the shared
            ``extra_hint`` helper put that name into ``dir(pyramids.feature)``, next to the
            ``import_dask_geopandas`` that was already there. Neither is feature API. What
            decides whether that matters is ``__all__``, which is what a star-import reads
            and what mkdocstrings treats as the declared surface -- so the check that keeps
            such an import harmless is that it never reaches the list.
        """
        borrowed = {
            name
            for name in dir(feature_package)
            if not name.startswith("_")
            and getattr(getattr(feature_package, name), "__module__", None)
            == "pyramids.base._utils"
        }
        advertised = sorted(borrowed & set(feature_package.__all__))

        assert not advertised, (
            "pyramids.feature.__all__ advertises helpers it only imported from "
            f"pyramids.base._utils: {advertised}"
        )
