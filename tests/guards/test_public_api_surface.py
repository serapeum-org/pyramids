"""Guard two public-surface promises the consolidation is easy to break by accident.

Moving a helper down into ``pyramids.base`` is invisible to callers only while the name it used to
live under keeps resolving, and adding an ``__all__`` to a module that never had one silently
*narrows* its star-import and its mkdocstrings page. Neither shows up in a test that exercises
behaviour, because both are about which names a module offers, not what they do.
"""

import inspect

import pyramids.dataset.dataset as dataset_module
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


class TestNetCDFUtilsAdvertisesEveryPublicFunctionItDefines:
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
