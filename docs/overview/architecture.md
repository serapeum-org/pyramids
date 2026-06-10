# Architecture

## High-Level Overview

```mermaid
graph TD
    subgraph IO["I/O Formats"]
        GeoTIFF["GeoTIFF"]
        NC["NetCDF"]
        UGRID_NC["UGRID NetCDF"]
        SHP["Shapefile / GeoJSON"]
        ZIP["Compressed (zip/gzip/tar)"]
    end

    subgraph Core["Core Components"]
        DS["<b>Dataset</b><br/>Single raster<br/>read · write · crop<br/>reproject · align"]
        NCD["<b>NetCDF</b><br/>extends Dataset<br/>time/variable dimensions<br/>CF conventions"]
        UGDS["<b>UgridDataset</b><br/>Unstructured mesh<br/>UGRID-1.0 conventions"]
        DC["<b>DatasetCollection</b><br/>Temporal raster stack<br/>multi-temporal analysis"]
        FC["<b>FeatureCollection</b><br/>Vector data<br/>GeoDataFrame + OGR"]
    end

    subgraph Ops["Operations"]
        Spatial["Spatial<br/>crop · reproject · align"]
        Math["Array / Band<br/>math · statistics · no-data"]
        Convert["Convert<br/>raster ↔ vector<br/>raster ↔ NetCDF"]
    end

    GeoTIFF --> DS
    NC --> NCD
    UGRID_NC --> UGDS
    SHP --> FC
    ZIP --> DS
    ZIP --> NCD

    DS --> Spatial
    DS --> Math
    DS --> Convert
    NCD --> Spatial
    NCD --> Math
    UGDS -->|"mesh-to-raster"| DS
    UGDS -->|"mesh-to-vector"| FC
    DC -->|"contains aligned<br/>Dataset stack"| DS
    FC --> Convert
    FC -->|"rasterize"| DS
    Convert --> FC
    Convert --> DS
```

## Class Relationships & Internal Layers

How the classes relate to each other, the engine mixins behind the `Dataset` facade, and the
plotting layer's delegation to cleopatra:

```mermaid
graph LR
    GeoTIFF & NetCDF & Shapefile & UGRID -->|read| pyramids
    subgraph pyramids
        direction TB
        Dataset
        NetCDF_class[NetCDF]
        UgridDataset
        DatasetCollection
        FeatureCollection
        subgraph Engines["Dataset engines (ds.io · ds.spatial · ds.bands · ds.analysis · ds.cell · ds.vectorize · ds.cog)"]
        end
        subgraph Plotting["Plotting layer — _plot_helpers.render_array · mesh_render · NetCDFPlot · Selectors / ColourOpts / FacetSpec · basemap"]
        end
    end
    Dataset -->|crop · reproject · align| Dataset
    Dataset --- Engines
    FeatureCollection -->|rasterize| Dataset
    UgridDataset -->|interpolate| Dataset
    Dataset -->|vectorize| FeatureCollection
    DatasetCollection -->|lazy temporal stack| Dataset
    NetCDF_class -->|extends| Dataset
    Dataset & NetCDF_class & DatasetCollection & UgridDataset -->|plot| Plotting
    Plotting -->|delegates| cleopatra(["cleopatra<br/>ArrayGlyph · MeshGlyph · tiles"])
```

## Class API Overview

```mermaid
graph LR
    subgraph Dataset
        DS_props["<b>Properties</b><br/>raster · rows · columns<br/>geotransform · epsg · crs<br/>band_count · band_names<br/>no_data_value · dtype"]
        DS_create["<b>Create</b><br/>read_file<br/>create_from_array<br/>dataset_like"]
        DS_spatial["<b>Spatial</b><br/>crop · to_crs · align<br/>resample · overlay"]
        DS_data["<b>Data Access</b><br/>read_array · extract<br/>get_variables"]
        DS_math["<b>Math</b><br/>apply · fill · normalize"]
        DS_convert["<b>Convert</b><br/>to_polygon<br/>to_geodataframe"]
        DS_io["<b>I/O</b><br/>to_file · plot"]
    end

    subgraph DatasetCollection
        DC_props["<b>Properties</b><br/>base · time_length"]
        DC_create["<b>Create</b><br/>read_multiple_files<br/>read_dataset"]
        DC_data["<b>Data Access</b><br/>iloc · extract<br/>get_variables"]
        DC_io["<b>I/O</b><br/>to_file · plot"]
    end

    subgraph FeatureCollection
        FC_props["<b>Properties</b><br/>feature · epsg · bounds"]
        FC_create["<b>Create</b><br/>read_file<br/>from GeoDataFrame"]
        FC_spatial["<b>Spatial</b><br/>xy · create_polygon<br/>create_point · concat"]
        FC_convert["<b>Convert</b><br/>to_dataset"]
        FC_io["<b>I/O</b><br/>to_file"]
    end

    subgraph UgridDataset
        UG_props["<b>Properties</b><br/>mesh · epsg · n_face<br/>data_variable_names"]
        UG_create["<b>Create</b><br/>read_file<br/>create_from_arrays"]
        UG_spatial["<b>Spatial</b><br/>subset_by_bounds<br/>clip"]
        UG_convert["<b>Convert</b><br/>to_dataset<br/>to_geodataframe"]
        UG_io["<b>I/O</b><br/>plot · plot_outline<br/>sel_time"]
    end

    DC_props -.->|"base is a"| DS_props
    FC_convert -.->|"rasterize"| DS_props
    UG_convert -.->|"interpolate"| DS_props
    DS_convert -.->|"vectorize"| FC_props
```

## Plotting layer

`Dataset.plot` / `NetCDF.plot` / `DatasetCollection.plot` / `UgridDataset.plot` are thin facades
over a shared core in `pyramids.dataset._plot_helpers` (`render_array` for arrays, `mesh_render`
for meshes), which builds a [cleopatra](https://github.com/serapeum-org/cleopatra) glyph and
dispatches to `plot` / `facet` / `animate`. `NetCDF.plot` adds the NetCDF-specific resolver
`pyramids.netcdf._plot.NetCDFPlot` (variable / selector / curvilinear-coord / facet / animate
resolution) and the grouped option dataclasses `Selectors` / `ColourOpts` / `FacetSpec`
(`pyramids.netcdf.plot_options`, re-exported from `pyramids.netcdf`).
`pyramids.basemap.add_basemap` / `get_provider` are thin wrappers over `cleopatra.tiles`.

```mermaid
graph LR
    DS["Dataset.plot<br/>(band, rgb_options, basemap)"]
    NC["NetCDF.plot<br/>(variable, selectors, colour,<br/>facet, coords, kind, animate, chunks)"]
    DC["DatasetCollection.plot<br/>(animate mode)"]
    UG["UgridDataset.plot / plot_outline"]
    AN["Analysis.plot<br/>(generic engine)"]

    NCP["netcdf._plot.NetCDFPlot<br/>resolve variable · selectors<br/>curvilinear coords · facet · animate"]
    OPTS["netcdf.plot_options<br/>Selectors · ColourOpts · FacetSpec"]
    RC["dataset._plot_helpers<br/>render_array · mesh_render"]
    BM["basemap<br/>add_basemap · get_provider"]
    CLEO(["cleopatra<br/>ArrayGlyph (plot · facet · animate)<br/>MeshGlyph · styles.ColorScale · tiles"])

    DS --> AN --> RC
    NC --> NCP --> RC
    NCP --> OPTS
    DC --> RC
    UG --> RC
    RC --> BM
    RC --> CLEO
    BM --> CLEO
```
