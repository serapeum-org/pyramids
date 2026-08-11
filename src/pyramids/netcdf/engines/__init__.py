"""Engine collaborators for :class:`pyramids.netcdf.NetCDF`.

Mirrors the :mod:`pyramids.dataset.engines` architecture: the
6,700-line ``netcdf.py`` god-object is decomposed into focused
collaborator classes, each owning the bodies of one public-API
family. ``NetCDF`` exposes thin façade methods that delegate to a
collaborator, so a façade call and the ``nc.<collaborator>.<method>()``
it delegates to are equivalent.

The collaborators use **distinct** attribute names from the eight
``Dataset`` engines (``io``, ``spatial``, …) that ``NetCDF`` already
inherits — ``interop``, ``variables``, ``selection`` — so wiring them
in ``NetCDF.__init__`` never clobbers an inherited engine. They reuse
:class:`pyramids.dataset.engines._base._Engine` for the weakref-proxy
back-reference and pickle placeholder contract (issue #615, STR-1).
"""
