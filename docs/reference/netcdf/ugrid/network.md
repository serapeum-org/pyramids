# Network1d

UGRID `topology_dimension == 1` data: river networks and 1-D hydraulic models.

Nodes plus an edge-node connectivity table, with **no faces** — the core element is the edge, so
there are no centroids or areas, and there is no `n_face`. `Network1d` is a standalone sibling of
`Mesh2d`, holding plain numpy arrays with no GDAL objects and no dependency on `Mesh2d`.

It offers edge lengths and midpoints, bounds, nearest-neighbour KD-trees on both nodes and edge
midpoints, and a topological sort for upstream/downstream ordering.

**Sorting an undirected network.** `sort()` uses the stored edge directions when the network was
built with `directed=True`. Otherwise name the outlet — `sort(outlet=<node>)` — and the orientation
is derived from it: every edge is directed from its end farther from the outlet towards its nearer
end. A network stores no elevation, so the outlet cannot be guessed; with no `outlet` the error
lists the degree-1 nodes that are candidates. An edge whose two ends are the same distance from the
outlet (an odd cycle) has no defensible direction and is refused by name.

**Reads come from a driver.** `Network1d.read_file` dispatches to a registered
`pyramids.mesh_drivers` plugin, including for a `.nc` network such as a D-Flow FM `_net.nc` file.
There is no built-in route: the UGRID topology parser builds a `Mesh2d` whatever
`topology_dimension` a file declares, so a network always comes from a plugin. Writing a network is
still unimplemented, so it does not travel through `to_file`.

::: pyramids.netcdf.ugrid.network.Network1d
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
