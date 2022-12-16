from argparse import ArgumentParser
from copy import copy as cp_object
import contextily as cx
from enum import Enum
from functools import partial
import geopandas as gpd
import math
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import Rbf
import rasterio
from rasterio.transform import Affine
import rioxarray
from shapely.geometry import Point
import time

from qcore.im import IM
from qcore import srf, constants

script_dir = Path(__file__).parent.resolve()

# Supported countries, and country-specific data set
COUNTRY = Enum("Country", ["NZ", "KR", "JP"])

# Obtained from gshhg-shp-2.3.7, cropped with QGIS. Much faster than using the original. See README for details.
COASTLINES_TOPO = {
    "NZ": script_dir / "data" / "coastlines" / "nz-coastlines-topo-gshhg.shp",
    "KR": script_dir / "data" / "coastlines" / "kr-jp-coastlines-topo-gshhg.shp",
    "JP": script_dir / "data" / "coastlines" / "kr-jp-coastlines-topo-gshhg.shp",
}

BASEMAPS = {
    "NZ": script_dir / "data" / "basemap" / "NZ10.tif",  #
    "KR": script_dir / "data" / "basemap" / "KR10.tif",
    "JP": script_dir / "data" / "basemap" / "JP09.tif",
}
# these maps were obtained by the commands similar to below
# w, s, e, n = (166,-48.5,178.5,-34)
# cx.bounds2raster(w, s, e, n, ll=True, zoom=10, path="NZ10.tif", source=cx.providers.Stamen.TerrainBackground)
# Possible Map styles are here: https://xyzservices.readthedocs.io/en/stable/gallery.html


CITIES = {
    "NZ": script_dir / "data" / "city_nz.csv",
    "KR": script_dir / "data" / "city_kr.csv",
    "JP": script_dir / "data" / "city_jp.csv",
}


DEFAULT_MAP_WIDTH = 12
DEFAULT_MAP_HEIGHT = 15
DEFAULT_PLOT_ASPECT = 1.5

DEFAULT_COMP = "geom"

DEFAULT_POINT_SIZE = 10
DEFAULT_CONTOUR_LEVELS = 10
DEFAULT_CONTOUR_LINE_WIDTH = 1.0
DEFAULT_CONTOUR_LINE_COLOR = "black"

DEFAULT_SURFACE_OPACITY = 0.9
DEFAULT_COLORMAP = "afmhot_r"

DEFAULT_SRF_COLORMAP = "afmhot_r"
DEFAULT_SRF_OUTLINE_COLOR = "navy"
DEFAULT_SRF_OUTLINE_WIDTH = 1
DEFAULT_GRID_SIZE = 1000

DEFAULT_AXIS_FONT_SIZE = 12
DEFAULT_TITLE_FONT_SIZE = 25
DEFAULT_COLORBAR_FONT_SIZE = 15
DEFAULT_CITY_FONT_SIZE = 12

# zorder : higher for upper layer
LAYER_IMAGE = 3
LAYER_SRF = 5

marker_png = script_dir / "marker.png"

# TODO:
# Enable Roads
# Enable MAX/MIN values


def get_args():
    parser = ArgumentParser()
    arg = parser.add_argument
    arg("-t", "--title", help="title text", default="")
    arg(
        "-f",
        "--output-prefix",
        default="plot_items",
        help="output path combined with filename prefix. eg. XXXX/YYYY in XXXX/YYYY_PGA.png",
    )
    arg(
        "-s",
        "--srf-file",
        action="append",
        help="SRF files to plot, use quoted wildcards, repeat as needed",
    )

    arg(
        "--srf-only-outline",
        action="store_true",
        help="Only plot outline for SRF files",
    )
    arg(
        "--srf-outline-color",
        help="Outline color of SRF outline. See https://matplotlib.org/stable/tutorials/colors/colors.html",
        default=DEFAULT_SRF_OUTLINE_COLOR,
    )
    arg(
        "--srf-outline-width",
        help="Outline width of SRF outline.",
        type=float,
        default=DEFAULT_SRF_OUTLINE_WIDTH,
    )
    arg(
        "--srf-colormap",
        help="Colormap used for SRF surface",
        default=DEFAULT_SRF_COLORMAP,
    )

    arg(
        "--map-height",
        help="Height of the output map in inches",
        type=float,
        default=DEFAULT_MAP_HEIGHT,
    ),
    arg(
        "--map-width",
        help="Width of the output map in inches",
        type=float,
        default=DEFAULT_MAP_WIDTH,
    ),
    arg(
        "--map-aspect",
        help="Aspect of Height / Width of the plotted map",
        type=float,
        default=DEFAULT_PLOT_ASPECT,
    )

    arg("--country", default="NZ", choices=[c.name for c in COUNTRY])

    arg(
        "--basemap", help="Path to the local basemap", type=Path, default=None,
    )

    arg(
        "--coastline",
        type=Path,
        help="ESRI shapefile (multipolygon) of coastline definition",
        default=None,
    )

    arg(
        "--city-csv",
        help="City locations CSV file: name, lon, lat",
        type=Path,
        default=None,
    )
    arg(
        "--disable-city-labels",
        dest="enable_city_labels",
        help="Flag to disable city_labels - these are plotted by default",
        action="store_false",
    )

    arg(
        "--column",
        help="Column names to plot. If unspecified, all included. Repeat as needed",
        action="append",
    )

    arg(
        "-r",
        "--region",
        help="Region to plot in the form min_lon/max_lon/min_lat/max_lat.",
    )

    arg(
        "--comp",
        help="Component in IM file to plot",
        default=DEFAULT_COMP,
        choices=[x.str_value for x in list(constants.Components)],
    )

    arg(
        "--colormap",
        help="CPT to use for overlay data. Use '_r' to invert. See https://matplotlib.org/stable/tutorials/colors/colormaps.html",
        default=DEFAULT_COLORMAP,
    )

    arg(
        "--categorical",
        help="Display categorical/discrete values as points. Recommended to use with a cyclinc/qualitative colormaps (eg) --colormap hsv",
        action="store_true",
    )

    arg(
        "--points", help="Display values as points", action="store_true",
    )

    arg(
        "--contours", help="Add contour lines from CPT increments", action="store_true",
    )

    arg(
        "--point-size",
        help="Point size for points/categorical",
        type=float,
        default=DEFAULT_POINT_SIZE,
    )

    arg(
        "--contour-levels",
        help="number of contour levels",
        type=int,
        default=DEFAULT_CONTOUR_LEVELS,
    )
    arg(
        "--contour-line-width",
        help="contour line width",
        type=float,
        default=DEFAULT_CONTOUR_LINE_WIDTH,
    )
    arg(
        "--contour-line-color",
        help="contour line color",
        default=DEFAULT_CONTOUR_LINE_COLOR,
    )
    arg("--surface", help="interpolated grid surface", action="store_true")
    arg(
        "--surface-opacity",
        help="overlay opacity: transparent(0) - opaque(1)",
        type=float,
        default=DEFAULT_SURFACE_OPACITY,
    )

    arg("-n", "--nproc", help="max number of processes", type=int, default=1)

    arg(
        "--fast",
        help="Run faster with less resolution for interpolated surface",
        action="store_true",
    )

    arg(
        "--nan-is",
        help="NaN is replaced by specified value (eg. 0). By default, NaN is removed",
        type=float,
        default=None,
    ),

    arg(
        "--axis-font-size", type=int, default=DEFAULT_AXIS_FONT_SIZE,
    )
    arg(
        "--title-font-size", type=int, default=DEFAULT_TITLE_FONT_SIZE,
    )
    arg(
        "--colorbar-font-size", type=int, default=DEFAULT_COLORBAR_FONT_SIZE,
    )
    arg(
        "--city-font-size", type=int, default=DEFAULT_CITY_FONT_SIZE,
    )
    arg(
        "data_csv",
        help="Path to csv file to plot (eg. IM csv). Each line is station name, (optional: component), column_values... ",
    )

    arg(
        "--no-component-column",
        help="If data_csv doesn't have component column, set this option",
        dest="has_component_column",
        action="store_false",
    )

    arg(
        "station_file",
        help="station list file. name, lon, lat with a space as the delimiter by default",
    )

    arg("--station-sep", default=" ", help="the delimiter used in the station file")

    args = parser.parse_args()

    if args.region is not None:
        corners = args.region.split("/")
        assert len(corners) == 4, "Required format is min_lon/max_lon/min_lat/max_lat"

        try:
            corners = [float(x) for x in corners]
        except ValueError:
            print("Error: min_lon/max_lon/min_lat/max_lat should be numbers.")
            args.region = None

        else:
            args.region = corners

    if args.categorical:
        print("Categorical Points are displayed.")
        args.surface = False
        args.points = False
        args.contours = False
    else:
        if not args.surface and not args.contours and not args.points:
            print("Warning: No drawing mode selected. Points are displayed.")
            args.points = True

    # if country specific arguments are not specified, use default ones.

    if args.coastline is None:
        args.coastline = COASTLINES_TOPO[args.country]
    if args.basemap is None:
        args.basemap = BASEMAPS[args.country]
    if args.city_csv is None:
        args.city_csv = CITIES[args.country]

    return args


def city_labels(ax, city_csv, fontsize=DEFAULT_CITY_FONT_SIZE):
    lons = []
    lats = []

    city_df = pd.read_csv(city_csv, index_col=0)
    for city in city_df.index:
        lon = city_df.loc[city].lon
        lat = city_df.loc[city].lat
        ax.annotate(
            city,
            xy=(lon, lat),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=fontsize,
        )
        lons.append(lon)
        lats.append(lat)
    ax.plot(lons, lats, "o")


def pre_interp(X, Y, Z, xres, yres, grid_size=DEFAULT_GRID_SIZE, fast=False):

    assert grid_size > 0
    # Use xres or yres if specified. Otherwise determine on-the-fly.
    if xres is None:
        xres = (X.max() - X.min()) / grid_size
    if yres is None:
        yres = (Y.max() - Y.min()) / grid_size

    res = min(xres, yres)

    if fast:
        res = res * 2

    xCoords = np.arange(
        X.min(), X.max() + res, res
    )  # can use linspace with adaptive res
    yCoords = np.arange(Y.min(), Y.max() + res, res)
    zCoords = np.zeros([yCoords.shape[0], xCoords.shape[0]])

    return xCoords, yCoords, zCoords, res


def raster_to_tiff(Z, X, Y, xres, yres, crs, filename):
    """Export and save a kernel density raster."""

    # Set transform
    transform = Affine.translation(X[0] - xres / 2, Y[0] - yres / 2) * Affine.scale(
        xres, yres
    )

    # Export array as raster
    with rasterio.open(
        filename,
        mode="w",
        driver="GTiff",
        height=Z.shape[0],
        width=Z.shape[1],
        count=1,
        dtype=Z.dtype,
        crs=crs,
        transform=transform,
    ) as new_dataset:
        new_dataset.write(Z, 1)


def tri_interp(llv_array, tiff_name, crs, res=None, fast=False):

    # Studied wide materials as below, testing various interpolation algorithms including Kriging, Rbf etc.
    # Opted for this implementation as it is fastest, resource-light, and output is reasonably close to gmt surface
    # References:
    #  https://hatarilabs.com/ih-en/geospatial-triangular-interpolation-with-python-scipy-geopandas-and-rasterio-tutorial
    #  https://pygis.io/docs/e_interpolation.html
    #  https://www.delftstack.com/api/scipy/2d-interpolation-python/
    #  https://stackoverflow.com/questions/44922766/2d-linear-interpolation-data-and-interpolated-points
    #  https://pythonguides.com/matplotlib-2d-surface-plot/

    X, Y, Z = llv_array[:, 0], llv_array[:, 1], llv_array[:, 2]

    triFn = Triangulation(X, Y)
    linTriFn = LinearTriInterpolator(triFn, Z)

    xCoords, yCoords, zCoords, res = pre_interp(X, Y, Z, res, res, fast=fast)

    print(f"Rasterizing {tiff_name} at resolution {res}")

    # loop among each cell in the raster extension
    for indexX, x in np.ndenumerate(xCoords):
        for indexY, y in np.ndenumerate(yCoords):
            tempZ = linTriFn(x, y)
            # filtering masked values
            if tempZ.mask:  # also works with tempZ == tempZ, but counter-intuitive.
                zCoords[indexY, indexX] = np.nan  # masked
            else:
                zCoords[indexY, indexX] = tempZ

    raster_to_tiff(zCoords, xCoords, yCoords, res, res, crs, tiff_name)


def rbf(llv_array, tiff_name, crs, res=None, fast=False):
    # this is not used, and found to be inferior to tri_interp.
    # left here as an example to show how a different interpolation can be used

    X, Y, Z = llv_array[:, 0], llv_array[:, 1], llv_array[:, 2]
    rbf_fun = Rbf(X, Y, Z, function="linear")
    xCoords, yCoords, zCoords, res = pre_interp(X, Y, Z, res, res, fast=fast)

    xGrid, yGrid = np.meshgrid(xCoords, yCoords)
    zNew = rbf_fun(xGrid.ravel(), yGrid.ravel()).reshape(xGrid.shape)

    raster_to_tiff(zNew, xCoords, yCoords, res, res, crs, tiff_name)


def load_srf(srf_file):
    seg_llvs = srf.srf2llv_py(srf_file)
    # planes = srf.read_header(srf_file, idx=True)
    bounds = srf.get_bounds(srf_file)
    np_bounds = np.array(bounds)
    n_plane = len(bounds)
    cpt_percentile = 95
    all_vs = np.concatenate((seg_llvs))[:, -1]
    percentile = np.percentile(all_vs, cpt_percentile)
    # round percentile significant digits for colour palette
    if percentile < 1000:
        # 1 sf
        cpt_max = round(percentile, -int(math.floor(math.log10(abs(percentile)))))
    else:
        # 2 sf
        cpt_max = round(percentile, 1 - int(math.floor(math.log10(abs(percentile)))))
    regions = []
    for s in range(n_plane):
        x_min, y_min = np.min(np_bounds[s], axis=0)
        x_max, y_max = np.max(np_bounds[s], axis=0)
        regions.append((x_min, x_max, y_min, y_max))

    return seg_llvs, bounds, regions, cpt_max


def pre_plot_srfs(
    srf_files,
    outdir,
    crs,
    res=0.001,
    outline_only=False,
    outline_color=DEFAULT_SRF_OUTLINE_COLOR,
    outline_width=DEFAULT_SRF_OUTLINE_WIDTH,
    layer=LAYER_SRF,
):
    # pre-process SRF files ready for plotting.

    all_surfaces = []  # for all SRF files
    all_rectangles = []
    for srf_file in srf_files:
        seg_surfaces = []  # can be multiple segments per SRF file
        seg_rectangles = []
        seg_llvs, bounds, regions, cpt_max = load_srf(srf_file)
        srf_name = Path(srf_file).stem
        if not outline_only:
            for i in range(len(seg_llvs)):
                seg_tiff = outdir / f"srf_{srf_name}_{i}.tif"
                tri_interp(seg_llvs[i], seg_tiff, {"init": crs}, res=res)
                seg_surfaces.append(seg_tiff)

            all_surfaces.append((seg_surfaces, cpt_max))

        for bound in bounds:
            vertices = np.array(
                bound + [bound[0]]
            )  # add the first vertex to complete 4 edges
            xs = vertices[:, 0]
            ys = vertices[:, 1]
            rectangle = []
            for i in range(len(xs) - 1):
                edge = Line2D(
                    [xs[i], xs[i + 1]],
                    [ys[i], ys[i + 1]],
                    color=outline_color,
                    linewidth=outline_width,
                    linestyle="-",
                    zorder=layer,
                )
                rectangle.append(edge)
            seg_rectangles.append(rectangle)
        all_rectangles.append(seg_rectangles)
    return all_surfaces, all_rectangles


def plot_srfs(
    ax, all_surfaces, all_rectangles, cmap=DEFAULT_SRF_COLORMAP, layer=LAYER_SRF
):
    # actual SRF plotting done here

    for seg_surfaces, cpt_max, in all_surfaces:  # all SRF files
        for seg_tiff in seg_surfaces:
            seg_surface = rioxarray.open_rasterio(seg_tiff, masked=True)
            seg_surface.plot(
                ax=ax, cmap=cmap, zorder=layer, vmax=cpt_max, add_colorbar=False
            )

    for seg_rectangles in all_rectangles:
        for rectangle in seg_rectangles:
            for edge in rectangle:
                new_edge = cp_object(
                    edge
                )  # The same Line2D element can't be added to multiple figures. draw its copy
                ax.add_line(new_edge)


def plot_column(
    xyv_df,
    colname,
    crs,
    clip_with,
    outdir,
    prefix,
    title,
    height,
    width,
    aspect,
    enable_city_labels=True,
    region=None,
    # drawing type
    points=False,
    surface=True,
    contours=True,
    categorical=False,
    # colormap
    cmap=DEFAULT_COLORMAP,
    srf_colormap=DEFAULT_SRF_COLORMAP,
    # srf-related
    srf_surfaces=[],
    srf_outlines=[],
    # other things to add to the map
    city_csv=CITIES["NZ"],
    basemap=True,
    basemap_path=BASEMAPS["NZ"],
    # drawing style
    point_size=DEFAULT_POINT_SIZE,
    contour_levels=DEFAULT_CONTOUR_LEVELS,
    contour_line_width=DEFAULT_CONTOUR_LINE_WIDTH,
    contour_line_color=DEFAULT_CONTOUR_LINE_COLOR,
    surface_opacity=DEFAULT_SURFACE_OPACITY,
    # font sizes
    axis_font_size=DEFAULT_AXIS_FONT_SIZE,
    colorbar_font_size=DEFAULT_COLORBAR_FONT_SIZE,
    title_font_size=DEFAULT_TITLE_FONT_SIZE,
    city_font_size=DEFAULT_CITY_FONT_SIZE,
    # processing speed related params
    fast=False,
    nan_is=None,
):

    if nan_is is None:  # NaN is removed
        xyv_df = xyv_df.loc[xyv_df[colname].notna()]
    else:
        # xyv_df = xyv_df.fillna(0) # no data gets 0
        xyv_df = xyv_df.fillna(nan_is)  # nan_is is already known to be float

    min_lon = xyv_df["lon"].min()
    max_lon = xyv_df["lon"].max()
    min_lat = xyv_df["lat"].min()
    max_lat = xyv_df["lat"].max()

    if region is not None:
        # crop to a larger area than specified first (for better interpolation). Crop to the exact dimension later
        min_lon, max_lon, min_lat, max_lat = region
        hmargin = (max_lon - min_lon) * 0.05
        vmargin = (max_lat - min_lat) * 0.05

        xyv_df = xyv_df.loc[
            (xyv_df["lon"] >= min_lon - hmargin)
            & (xyv_df["lon"] <= max_lon + hmargin)
            & (xyv_df["lat"] >= min_lat - vmargin)
            & (xyv_df["lat"] <= max_lat + vmargin)
        ]

    fig, ax = plt.subplots(figsize=(width, height))

    if categorical:
        uniq = list(set(xyv_df[colname]))
        cNorm = Normalize(vmin=0, vmax=len(uniq))
        scalarMap = ScalarMappable(norm=cNorm, cmap=cmap)
        for i in range(len(uniq)):
            indx = xyv_df[colname] == uniq[i]
            ax.scatter(
                xyv_df["lon"][indx],
                xyv_df["lat"][indx],
                s=point_size,
                color=scalarMap.to_rgba(i),
                alpha=surface_opacity,
                label=uniq[i],
            )
            plt.legend(
                title=colname,
                title_fontsize=colorbar_font_size,
                fontsize=(colorbar_font_size - 2),
                shadow=True,
                ncols=len(uniq),
                loc="lower center",
                bbox_to_anchor=(0.0, -0.12, 1, 0.05),
            )  # push by 0.12 downwards

    else:
        if surface or contours:
            llv_array = xyv_df[["lon", "lat", colname]].to_numpy()
            surface_tiff = outdir / f"surface_{colname}.tif"
            # rbf(llv_array, surface_tiff, {"init": crs}, fast=fast)
            tri_interp(llv_array, surface_tiff, {"init": crs}, fast=fast)

            print(f"Clipping starts")

            begin = time.time()
            # clipped: xarray.DataArray
            clipped = rioxarray.open_rasterio(surface_tiff, masked=True).rio.clip(
                clip_with, crs, drop=False
            )
            print(f"Clipping completed {time.time()-begin} secs")
            if contours:
                clipped[0].plot.contour(
                    ax=ax,
                    colors=[contour_line_color],
                    linewidths=[contour_line_width],
                    levels=contour_levels,
                )  # draw contour

            if surface:
                # render clipped surface
                clipped[0].plot.pcolormesh(
                    ax=ax, cmap=cmap, alpha=surface_opacity, add_colorbar=False
                )  # add custom colorbar below

        if points:
            geometry = [Point(xy) for xy in zip(xyv_df["lon"], xyv_df["lat"])]
            geodata = gpd.GeoDataFrame(xyv_df, crs=crs, geometry=geometry)
            geodata.plot(
                colname, ax=ax, legend=False, markersize=point_size, cmap=cmap
            )  # draw points, no default colorbar

        # Place a custom colorbar
        cax = ax.inset_axes([0.0, -0.1, 1, 0.05])
        cNorm = ScalarMappable(
            norm=Normalize(vmin=xyv_df[colname].min(), vmax=xyv_df[colname].max()),
            cmap=cmap,
        )
        cbar = plt.colorbar(
            mappable=cNorm,
            ax=ax,
            cax=cax,
            orientation="horizontal",
            ticklocation="bottom",
            shrink=0.8,
        )

        imname_prefix = colname.split("_")[0]  # in case we have pSA_0.1 etc
        try:
            unit = IM(imname_prefix).get_unit()  # get the unit for this IM type
        except KeyError:  # may not be a recognized IM type. No unit
            unit_text = ""
        else:
            unit_text = f"({unit})"

        cbar.set_label(f"{colname} {unit_text}", fontsize=colorbar_font_size)

    if basemap:
        if basemap_path is not None:
            cx.add_basemap(ax, crs=crs, source=basemap_path, zoom=8)  # add basemap

        else:  # Download map from online (slow and quality may be low)
            cx.add_basemap(
                ax, crs=crs, source=cx.providers.Stamen.TerrainBackground, zoom="auto"
            )

    if enable_city_labels:
        city_labels(ax, city_csv, city_font_size)

    # Show SRFs. (Has to be done here to prevent weird basemap cropping)
    # srf_surfaces, srf_outlines are empty if no SRFs specified.
    plot_srfs(ax, srf_surfaces, srf_outlines, cmap=srf_colormap)

    # Show axis labels
    ax.set_xlabel("Longitude", fontsize=axis_font_size)
    ax.set_ylabel("Latitude", fontsize=axis_font_size)

    # Show Title
    plt.title(f"{title}", fontsize=title_font_size)

    # Crop to the specified region
    ax.set_xbound(lower=min_lon, upper=max_lon)
    ax.set_ybound(lower=min_lat, upper=max_lat)

    # Make sure file name doesn't have confusing dots.
    colname_p = colname.replace(".", "p")

    # If image should be added to the map, do here
    # plot_image(ax, marker_png, 167,-44.0, 0.03) # Just an example

    # Set the aspect as specified.
    ax.set_aspect(aspect)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_{colname_p}.png")
    plt.close(fig)


def plot_image(
    ax, imagefile, lon, lat, zoom=1, layer=LAYER_IMAGE
):  # zoom = 1 is full size of image
    # may be used for displaying external markers eg. beachballs
    # In fact, beachballls can be generated on-the-fly.
    # See https://github.com/ucgmsim/sim_atlas/blob/main/backend/utils/beachball_creation.py

    img = plt.imread(imagefile)
    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax
    # create bbox for the images
    ab = AnnotationBbox(im, (lon, lat), frameon=False, pad=0.0, zorder=layer)
    ax.add_artist(ab)


if __name__ == "__main__":
    crs = "EPSG:4326"  # projection

    args = get_args()
    print(f"Using {args.nproc} CPU cores")

    data_df = pd.read_csv(
        args.data_csv, index_col=0
    )  # each line is station_name, (optional: component), column_values...
    stations = pd.read_csv(
        args.station_file,
        sep=args.station_sep,
        header=None,
        index_col=2,
        names=["lon", "lat", "station"],
    )

    output_prefix = Path(args.output_prefix)
    basename = output_prefix.name
    out_dir = output_prefix.parent.resolve()
    out_dir.mkdir(exist_ok=True)

    column_names = list(data_df.columns)

    if args.has_component_column:  # Typical IM.csv
        column_names.remove("component")
        assert args.comp in set(
            data_df.component
        ), f"Specified component {args.comp} is unavailable in {args.data_csv}"
        data_to_plot_df = data_df[
            data_df.component == args.comp
        ]  # select the specified component
        print(f"Plotting component {args.comp}")
    else:  # generic data CSV with each line
        column_names = list(data_df.columns)
        data_to_plot_df = data_df

    columns_to_plot = []
    if args.column is None:
        columns_to_plot = column_names
    else:
        for colname in set(args.column):  # remove duplicates
            if colname in column_names:
                columns_to_plot.append(colname)
            else:
                print(f"Warning: No such column present: {colname}")
    print(f"Columns to plot: {columns_to_plot}")

    xyv_df = data_to_plot_df.join(stations, how="right")[
        ["lon", "lat"] + columns_to_plot
    ]  # joining 2 dataframes on index "station" name.

    coastlines_polygon = gpd.read_file(args.coastline)

    srf_surfaces = []
    srf_outlines = []

    if args.srf_file:
        srf_surfaces, srf_outlines = pre_plot_srfs(
            args.srf_file,
            out_dir,
            crs=crs,
            res=0.001,
            outline_only=args.srf_only_outline,
            outline_width=args.srf_outline_width,
            outline_color=args.srf_outline_color,
        )

    pfn = partial(
        plot_column,
        xyv_df,
        crs=crs,
        clip_with=coastlines_polygon.geometry.values,
        outdir=out_dir,
        prefix=basename,
        title=args.title,
        height=args.map_height,
        width=args.map_width,
        aspect=args.map_aspect,
        enable_city_labels=args.enable_city_labels,
        region=args.region,
        points=args.points,
        surface=args.surface,
        contours=args.contours,
        categorical=args.categorical,
        point_size=args.point_size,
        contour_levels=args.contour_levels,
        contour_line_width=args.contour_line_width,
        contour_line_color=args.contour_line_color,
        axis_font_size=args.axis_font_size,
        title_font_size=args.title_font_size,
        colorbar_font_size=args.colorbar_font_size,
        surface_opacity=args.surface_opacity,
        cmap=args.colormap,
        srf_surfaces=srf_surfaces,
        srf_outlines=srf_outlines,
        srf_colormap=args.srf_colormap,
        basemap_path=args.basemap,
        city_csv=args.city_csv,
        fast=args.fast,
        nan_is=args.nan_is,
    )

    # Parallel
    with Pool(args.nproc) as pool:
        plot_properties = pool.map_async(pfn, columns_to_plot)
        plot_properties = plot_properties.get()

    # Serial
    # for im in columns_to_plot:
    #     pfn(im)
