import rasterio
from rasterio.crs import CRS

from rasterio.transform import Affine

import rioxarray
import contextily as cx



import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from scipy.interpolate import Rbf


import math

from copy import copy as cp_object


import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon

import pandas as pd

from pathlib import Path

from argparse import ArgumentParser
from functools import partial

from multiprocessing import Pool

from qcore.im import IM
from qcore import srf, constants

script_dir = Path(__file__).parent.resolve()
COASTLINES_TOPO_POLYGON = script_dir / 'nz_coastlines/nz-coastlines-topo-150k_polygon.shp'
DEFAULT_LOCAL_BASEMAP= script_dir /"NZ10.tif"
DEFAULT_CITY_CSV = script_dir/"city.csv"

DEFAULT_MAP_WIDTH = 12
DEFAULT_MAP_HEIGHT = 15
DEFAULT_PLOT_ASPECT = 1.5

DEFAULT_COMP = "geom"
DEFAULT_CONTOUR_LEVELS = 10
DEFAULT_OPACITY = 0.7
DEFAULT_COLORMAP = "CMRmap_r"

DEFAULT_GRID_SIZE = 1000
beachball = script_dir/"beachball.png"


LAYER_IMAGE = 3
LAYER_SRF = 5


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
        help="outline color of SRF outline. See https://matplotlib.org/stable/tutorials/colors/colors.html",
        default="navy",
    )

    arg("--map-height", help="Height of the output map in inches", type=float, default=DEFAULT_MAP_HEIGHT),
    arg("--map-width", help="Width of the output map in inches", type=float, default=DEFAULT_MAP_WIDTH),
    arg("--map-aspect", help="Aspect of Height / Width of the plotted map", type=float, default=DEFAULT_PLOT_ASPECT)
    arg("--basemap", help="Path to the local basemap", type=Path, default=DEFAULT_LOCAL_BASEMAP)

    arg("-r", "--region", help="Region to plot in the form xmin/xmax/ymin/ymax.")

    arg("--comp", help="component in IM file to plot", default=DEFAULT_COMP,choices=[x.str_value for x in list(constants.Components)])

    arg(
        "--opacity",
        help="overlay opacity: transparent(0) - opaque(1)",
        type=float,
        default=DEFAULT_OPACITY,
    )
    arg("--colormap", help="CPT to use for overlay data. Use '_r' to invert. See https://matplotlib.org/stable/tutorials/colors/colormaps.html", default=DEFAULT_COLORMAP)


    # arg(
    #     "--categorical",
    #     help="colour scale as discreet values",
    #     action="store_true",
    # )

    arg(
        "--contours",
        help="add contour lines from CPT increments",
        action="store_true",
    )

    arg(
        "--contour-levels",
        help="number of contour levels",
        type=int,
        default=DEFAULT_CONTOUR_LEVELS,
    )


    arg("--surface", help="interpolated grid surface", action="store_true")

    arg(
        "--disable-city-labels",
        dest="enable_city_labels",
        help="Flag to disable city_labels - these are plotted by default",
        default=True,
        action="store_false",
    )
    arg(
        "--city-csv",
        help="City locations CSV file: name, lon, lat",
        type=Path,
        default=DEFAULT_CITY_CSV,
    )

    # arg(
    #     "--disable-roads",
    #     dest="enable_roads",
    #     help="Flag to disable roads/highways - these are plotted by default",
    #     default=True,
    #     action="store_false",
    # )
    arg("-n", "--nproc", help="max number of processes", type=int, default=1)

    arg(
        "--fast",
        help="Run faster with less resolution for interpolated surface",
        action="store_true"
    )

    arg(
        "--nan-is",
        help="NaN is replaced by specified value (eg. 0). By default, NaN is removed", type=float,
        default=None,
    ),

    arg(
        "imcsv", help="path to im csv file"
    )

    arg(
        "station_file", help="station list file. name, lon, lat with a space as the delimiter by default"
    )

    arg(
        "--station-sep", default=" ", help="the delimiter used in the station file"
    )

    args = parser.parse_args()

    if args.region is not None:
        corners = args.region.split("/")
        assert len(corners) == 4, "Required format is xmin/xmax/ymin/ymax"

        try:
            corners = [float(x) for x in corners]
        except ValueError:
            print("Error: xmin/xmax/ymin/ymax should be numbers.")
            args.region = None

        else:
            args.region = corners

    return args


def city_labels(ax, city_csv):
    lons = []
    lats = []

    city_df=pd.read_csv(city_csv,index_col=0)
    for city in city_df.index:
        lon = city_df.loc[city].lon
        lat = city_df.loc[city].lat
        ax.annotate(city, xy=(lon, lat), xytext=(3, 3), textcoords="offset points")
        lons.append(lon)
        lats.append(lat)
    ax.plot(lons, lats, 'o')

def pre_interp(X,Y,Z, xres, yres, grid_size=DEFAULT_GRID_SIZE, fast=False):

    assert grid_size > 0
    # Use xres or yres if specified. Otherwise determine on-the-fly.
    if xres is None:
        xres = (X.max() - X.min()) / grid_size
    if yres is None:
        yres = (Y.max() - Y.min()) / grid_size

    res = min(xres, yres)

    if fast:
        res = res * 2

    xCoords = np.arange(X.min(), X.max() + res, res) #can use linspace with adaptive res
    yCoords = np.arange(Y.min(), Y.max() + res, res)
    zCoords = np.zeros([yCoords.shape[0], xCoords.shape[0]])

    return xCoords, yCoords, zCoords, res
def raster_to_tiff(Z, X,Y, xres, yres, crs, filename):
    '''Export and save a kernel density raster.'''

    # Set transform
    transform = Affine.translation(X[0] - xres / 2, Y[0] - yres / 2) * Affine.scale(xres, yres)

    # Export array as raster
    with rasterio.open(
            filename,
            mode = "w",
            driver = "GTiff",
            height = Z.shape[0],
            width = Z.shape[1],
            count = 1,
            dtype = Z.dtype,
            crs = crs,
            transform = transform,
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

    X,Y,Z = llv_array[:,0], llv_array[:,1], llv_array[:,2]

    triFn = Triangulation(X, Y)
    linTriFn = LinearTriInterpolator(triFn, Z)

    xCoords, yCoords, zCoords, res = pre_interp(X,Y,Z, res, res, fast=fast)

    print(f"Rasterizing {tiff_name} at resolution {res}")

    # loop among each cell in the raster extension
    for indexX, x in np.ndenumerate(xCoords):
        for indexY, y in np.ndenumerate(yCoords):
            tempZ = linTriFn(x, y)
            # filtering masked values
            if tempZ.mask: # also works with tempZ == tempZ, but counter-intuitive.
                zCoords[indexY, indexX] = np.nan  # masked
            else:
                zCoords[indexY, indexX] = tempZ

    raster_to_tiff(zCoords,xCoords, yCoords, res,res,crs, tiff_name)



def rbf(llv_array, tiff_name, crs, res=None, fast=False):
# this is not used, and found to be inferior to tri_interp.
# left here as an example to show how a different interpolation can be used

    X, Y, Z = llv_array[:, 0], llv_array[:, 1], llv_array[:, 2]
    rbf_fun = Rbf(X,Y,Z, function='linear')
    xCoords, yCoords, zCoords, res = pre_interp(X, Y, Z, res, res, fast=fast)

    xGrid, yGrid = np.meshgrid(xCoords,yCoords)
    zNew = rbf_fun(xGrid.ravel(),yGrid.ravel()).reshape(xGrid.shape)

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

    return seg_llvs, bounds, regions,  cpt_max

def pre_plot_srfs(srf_files, outdir, crs, res=0.001, outline_only=False, outline_color='navy', linewidth=1, layer=LAYER_SRF):

    all_surfaces = [] # for all SRF files
    all_rectangles = []
    for srf_file in srf_files:
        seg_surfaces = [] # can be multiple segments per SRF file
        seg_rectangles = []
        seg_llvs, bounds, regions,  cpt_max = load_srf(srf_file)
        srf_name=Path(srf_file).stem
        if not outline_only:
            for i in range(len(seg_llvs)):
                seg_tiff = outdir / f'srf_{srf_name}_{i}.tif'
                tri_interp(seg_llvs[i],seg_tiff, {"init": crs}, res=res)
                seg_surfaces.append(seg_tiff)

            all_surfaces.append((seg_surfaces,cpt_max))

        for bound in bounds:
            vertices=np.array(bound+[bound[0]]) # add the first vertex to complete 4 edges
            xs = vertices[:, 0]
            ys = vertices[:, 1]
            rectangle =[]
            for i in range(len(xs)-1):
                edge=Line2D([xs[i],xs[i+1]],[ys[i],ys[i+1]],color=outline_color, linewidth=linewidth, linestyle='-', zorder=layer)
                rectangle.append(edge)
            seg_rectangles.append(rectangle)
        all_rectangles.append(seg_rectangles)
    return all_surfaces, all_rectangles

def plot_srfs(ax, all_surfaces,all_rectangles,layer=LAYER_SRF):
    for seg_surfaces, cpt_max, in all_surfaces: #all SRF files
        for seg_tiff in seg_surfaces:
            seg_surface = rioxarray.open_rasterio(seg_tiff, masked=True)
            seg_surface.plot(ax=ax,cmap="afmhot_r", zorder=layer, vmax=cpt_max, add_colorbar=False)

    for seg_rectangles in all_rectangles:
        for rectangle in seg_rectangles:
            for edge in rectangle:
                new_edge = cp_object(edge) # The same Line2D element can't be added to multiple figures
                ax.add_line(new_edge)


def plot_im(xyz, im_name, crs, clip_with, outdir, prefix, title, height, width, aspect,
                  enable_city_labels=True,
                  region=None,
                  surface=True,
                  contours=True,
                  contour_levels=DEFAULT_CONTOUR_LEVELS,
                  cmap=DEFAULT_COLORMAP,
                  opacity=DEFAULT_OPACITY,
                  srf_surfaces = [],
                  srf_outlines = [],
                  city_csv = DEFAULT_CITY_CSV,
                  basemap=True,
                  local_basemap= DEFAULT_LOCAL_BASEMAP,
                  fast=False,
                  nan_is=None):
    # print(f"{im_name} {crs} {clip_with} {surface} {contours} {basemap} {local_basemap}")

    # if not surface:
    #     xyz = xyz.fillna(0)

    if nan_is is None:  #NaN is removed
        xyz = xyz.loc[xyz[im_name].notna()]
    else:
        # xyz = xyz.fillna(0) # no data gets 0
        xyz=xyz.fillna(nan_is) #nan_is is already known to be float


    xmin = xyz['lon'].min()
    xmax = xyz['lon'].max()
    ymin = xyz['lat'].min()
    ymax = xyz['lat'].max()

    if region is not None:
        #TODO: Crop to a larger area than specified first (for better interpolation). Crop to the exact dimension later
        xmin, xmax, ymin, ymax = region
        xmargin = (xmax-xmin)* 0.05
        ymargin = (ymax-ymin)* 0.05

        xyz = xyz.loc[(xyz['lon'] >= xmin-xmargin) & (xyz['lon'] <= xmax+xmargin) & (xyz['lat'] >= ymin-ymargin) & (xyz['lat'] <= ymax+ymargin)]

    if surface or contours:

        llv_array = xyz[['lon', 'lat', im_name]].to_numpy()

        surface_tiff = outdir / f'surface_{im_name}.tif'
        #rbf(llv_array, surface_tiff, {"init": crs}, fast=fast)
        tri_interp(llv_array, surface_tiff, {"init": crs}, fast=fast)





    fig, ax = plt.subplots(figsize=(width, height))

    plt.xlim([xmin,xmax])
    plt.ylim([ymin, ymax])


    if not surface:
        geometry = [Point(xy) for xy in zip(xyz["lon"], xyz["lat"])]
        geodata = gpd.GeoDataFrame(xyz, crs={"init": crs}, geometry=geometry)
        geodata.plot(im_name, ax=ax, legend=True, markersize=1, cmap=cmap)  # draw points


    plot_srfs(ax,srf_surfaces,srf_outlines)



    if surface or contours:
        # clipped: xarray.DataArray
        clipped = rioxarray.open_rasterio(surface_tiff, masked=True).rio.clip(clip_with, crs, drop=False)
        if contours:
            clipped[0].plot.contour(ax=ax, colors=['black'], linewidths=[0.3], levels=contour_levels)  # draw contour

        if surface:
            # render clipped surface
            p = clipped[0].plot.pcolormesh(ax=ax, cmap=cmap, alpha=opacity, add_colorbar=False) # add custom colorbar below
            cax = ax.inset_axes([0.0, -0.1, 1, 0.05])
            cbar = plt.colorbar(p, ax=ax, cax=cax, orientation="horizontal", ticklocation='bottom', shrink=0.8)
            # unit = "g" #
            imname_prefix = im_name.split("_")[0]  # in case we have pSA_0.1 etc
            unit = IM(imname_prefix).get_unit()
            cbar.set_label(f"{im_name} ({unit})", fontsize=15)





    if basemap:
        # add basemap
        if local_basemap is not None:

            cx.add_basemap(ax, crs=crs, source=local_basemap, zoom=8)  # add basemap

            # this map was obtained by the following lines
            # w, s, e, n = (166,-48.5,178.5,-34)
            # _ = cx.bounds2raster(w, s, e, n,ll=True, zoom=10, path="NZ10.tif",
            # source=cx.providers.Stamen.TerrainBackground)

            # Possible Map styles are here: https://xyzservices.readthedocs.io/en/stable/gallery.html
        else:  # Download map from online (slow and quality may be low)
            cx.add_basemap(ax, crs=crs, source=cx.providers.Stamen.TerrainBackground, zoom="auto")


    if enable_city_labels:
        city_labels(ax, city_csv)



    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    # ax.legend(loc='lower center',mode='expand')

    plt.title(f"{title}", fontsize=20)
    #    plt.show()

    im_name_p = im_name.replace(".", "p")


    # plot_image(ax, beachball, 167,-45.5, 0.1)


    ax.set_aspect(aspect)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_{im_name_p}.png")
    plt.close(fig)


def plot_image(ax, imagefile, lon, lat, zoom=1, layer=LAYER_IMAGE): #zoom = 1 is full size of image
    # may be used for displaying external markers eg. beachballs
    # In fact, beachballls can be generated on-the-fly.
    # See https://github.com/ucgmsim/sim_atlas/blob/main/backend/utils/beachball_creation.py

    img = plt.imread(imagefile)
    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax
    # create bbox for the images
    ab = AnnotationBbox(im, (lon,lat), frameon=False, pad=0.0, zorder=layer)
    ax.add_artist(ab)

if __name__ == "__main__":
    rasterCrs = CRS.from_epsg(4326)

    args = get_args()

    ims_df = pd.read_csv(args.imcsv, index_col=0)
    stations = pd.read_csv(args.station_file, sep=' ', header=None,
                           index_col=2, names=['lon', 'lat', 'station'])

    output_prefix = Path(args.output_prefix)
    basename = output_prefix.name
    out_dir = output_prefix.parent.resolve()
    out_dir.mkdir(exist_ok=True)

    ims = list(ims_df.columns)[1:]  # [1:] to remove component column
    assert args.comp in set(ims_df.component), f"Specified component {args.comp} is unavailable in {args.imcsv}"
    ims_with_one_comp=ims_df[ims_df.component==args.comp] # select the specified component
    print(f"Plotting component {args.comp}")

    xyz_df = ims_with_one_comp.join(stations, how='right')[['lon', 'lat'] + ims]  # joining 2 dataframes on index "station" name.
    coastlines_polygon = gpd.read_file(COASTLINES_TOPO_POLYGON)


    print(args.nproc)
    srf_surfaces = []
    srf_outlines = []

    if args.srf_file:
        srf_surfaces, srf_outlines = pre_plot_srfs(args.srf_file,out_dir, crs=rasterCrs.to_string(), res=0.001, outline_only=args.srf_only_outline, outline_color=args.srf_outline_color)


    pfn = partial(plot_im, xyz_df, crs=rasterCrs.to_string(), clip_with=coastlines_polygon.geometry.values,
                  outdir=out_dir, prefix=basename,
                  title=args.title,
                  height=args.map_height,
                  width=args.map_width,
                  aspect=args.map_aspect,
                  enable_city_labels=args.enable_city_labels,
                  region=args.region,
                  surface=args.surface,
                  contours=args.contours,
                  contour_levels=args.contour_levels,
                  opacity=args.opacity,
                  cmap=args.colormap,
                  srf_surfaces=srf_surfaces, srf_outlines=srf_outlines,
                  city_csv=args.city_csv,
                  fast=args.fast,
                  nan_is=args.nan_is)


    # Parallel
    with Pool(args.nproc) as pool:
        plot_properties = pool.map_async(pfn, ims[-10:])
        plot_properties = plot_properties.get()

    # Serial
    # for im in ims[10:]:
    #     pfn(im)