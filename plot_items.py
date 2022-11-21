
import rasterio
from rasterio.crs import CRS
from rasterio.plot import show
from rasterio.transform import Affine


import rioxarray
import contextily as cx

import xarray.plot

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon

import pandas as pd

from pathlib import Path

from argparse import ArgumentParser
from functools import partial
from glob import glob
from multiprocessing import Pool
import os
import pkg_resources
from shutil import copy, rmtree
from tempfile import TemporaryDirectory

script_dir = os.path.abspath(os.path.dirname(__file__))
MAP_WIDTH = 7


def get_args():
    parser = ArgumentParser()
    arg = parser.add_argument
    # arg("-t", "--title", help="title text", default="")
    arg(
        "-f",
        "--filename",
        default="plot_items",
        help="output filename excluding extension",
    )
    # arg("--fast", help="no topography, low resolution coastlines", action="store_true")
    # arg(
    #     "-s",
    #     "--srf-files",
    #     action="append",
    #     help="SRF files to plot, use quoted wildcards, repeat as needed",
    # )
    # arg(
    #     "--fault-colour",
    #     help="outline colour of faults, repeat for different colour per location given",
    #     action="append",
    # )
    # arg(
    #     "-c",
    #     "--srf-only-outline",
    #     action="append",
    #     help="SRF and SRF corners files to plot only outline for, use quoted wildcards, repeat as needed",
    # )
    # arg(
    #     "--outline-fault-colour",
    #     help="outline colour of only-outline faults, repeat for different colour per location given",
    #     action="append",
    # )
    # arg("--logo", help="include logo", action="store_true")
    # arg("--logo-pos", help="logo position LCR, TMB eg: 'LT'", default="LT")
    # arg("-b", "--bb-scale", help="beachball scale", type=float, default=0.05)
    # arg(
    #     "--slip-max",
    #     help="maximum slip (cm/s) on colour scale",
    #     type=float,
    #     default=1000.0,
    # )
    # arg("-r", "--region", help="Region to plot in the form xmin/xmax/ymin/ymax.")
    # arg(
    #     "-v",
    #     "--vm-corners",
    #     action="append",
    #     help="VeloModCorners.txt to plot, use quoted wildcards, repeat as needed",
    # )
    # arg(
    #     "-x",
    #     "--xyts-corners",
    #     action="append",
    #     help="xyts.e3d to plot outlines for, use quoted wildcards, repeat as needed",
    # )
    # arg(
    #     "--ll-file", help="add longitude latitude station file to plot", action="append"
    # )
    # arg("--ll-colour", help="colour of ll-file points", action="append")
    # arg("--ll-outline", help="outline colour of ll-file points", action="append")
    # arg("--ll-thickness", help="outline thickness of ll-file points", action="append")
    # arg("--ll-size", help="size of ll-file points", action="append")
    # arg("--ll-shape", help="shape of ll-file points", action="append")
    #
    # arg("--xyz-landmask", help="only show overlay over land", action="store_true")
    # arg(
    #     "--xyz-distmask",
    #     help="mask areas more than (km) from nearest point",
    #     type=float,
    # )
    # arg("--xyz-size", help="size of points or grid spacing eg: 1c or 1k")
    # arg("--xyz-shape", help="shape of points eg: t,c,s...", default="t")
    # arg(
    #     "--xyz-model-params",
    #     help="crop xyz overlay with VeloModCorners or model_params",
    # )
    # arg(
    #     "--xyz-transparency",
    #     help="overlay transparency 0-100 (invisible)",
    #     type=float,
    #     default=30,
    # )
    # arg("--xyz-cpt", help="CPT to use for overlay data", default="hot")
    # arg("--xyz-cpt-invert", help="inverts CPT", action="store_true")
    # arg(
    #     "--xyz-cpt-continuous",
    #     help="generate continuous colour change CPT",
    #     action="store_true",
    # )
    # arg(
    #     "--xyz-cpt-continuing",
    #     help="background/foreground matches colors at ends of CPT",
    #     action="store_true",
    # )
    # arg("--xyz-cpt-asis", help="don't processes input CPT", action="store_true")
    # arg(
    #     "--xyz-cpt-categorical",
    #     help="colour scale as discreet values, implies --xyz-cpt-asis",
    #     action="store_true",
    # )
    # arg(
    #     "--xyz-cpt-gap",
    #     help="if categorical: gap between CPT scale values, centre align labels",
    #     default="",
    # )
    # arg(
    #     "--xyz-cpt-intervals",
    #     help="if categorical, display value intervals",
    #     action="store_true",
    # )
    # arg("--xyz-cpt-labels", help="colour scale labels", default=["values"], nargs="+")
    # arg("--xyz-cpt-min", help="CPT minimum values, '-' to keep automatic", nargs="+")
    # arg("--xyz-cpt-max", help="CPT maximum values, '-' to keep automatic", nargs="+")
    # arg("--xyz-cpt-inc", help="CPT colour increments, '-' to keep automatic", nargs="+")
    # arg(
    #     "--xyz-cpt-tick",
    #     help="CPT legend annotation spacing, '-' to keep automatic",
    #     nargs="+",
    # )
    # arg("--xyz-cpt-bg", help="overlay colour below CPT min, above max if invert")
    # arg("--xyz-cpt-fg", help="overlay colour above CPT max, below min if invert")
    # arg("--xyz-grid", help="display as grid instead of points", action="store_true")
    # arg("--xyz-grid-automask", help="crop area further than dist from points eg: 8k")
    # arg(
    #     "--xyz-grid-contours",
    #     help="add contour lines from CPT increments",
    #     action="store_true",
    # )
    # arg(
    #     "--xyz-grid-contours-inc",
    #     help="add contour lines with this increment",
    #     type=float,
    # )
    # arg("--xyz-grid-type", help="interpolation program to use", default="surface")
    # arg(
    #     "--xyz-grid-search",
    #     help="search radius for interpolation eg: 5k (only m|s units for surface)",
    # )
    # arg("--labels-file", help="file containing 'lat lon label' to be added to the map")
    # arg(
    #     "--disable-city-labels",
    #     dest="enable_city_labels",
    #     help="Flag to disable city_labels - these are plotted by default",
    #     default=True,
    #     action="store_false",
    # )
    # arg(
    #     "--disable-roads",
    #     dest="enable_roads",
    #     help="Flag to disable roads/highways - these are plotted by default",
    #     default=True,
    #     action="store_false",
    # )
    # arg("-n", "--nproc", help="max number of processes", type=int, default=1)
    # arg("-d", "--dpi", help="render DPI", type=int, default=300)
    # arg(
    #     "--downscale",
    #     help="DPI render multiplier for better pixel accuracy",
    #     type=int,
    #     default=8,
    # )

    arg(
        "imcsv", help="path to im csv file"
    )

    arg(
        "station", help="station list. name, lon, lat with a space as the delimiter by default"
    )

    arg(
        "--station-sep", default=" ", help="the delimiter used in the station file"
    )

    return parser.parse_args()


if __name__ == "__main__":


    rasterCrs = CRS.from_epsg(4326)

    args = get_args()

    ims_df = pd.read_csv(args.imcsv, index_col=0)
    stations = pd.read_csv(args.station, sep=' ', header=None,
                           index_col=2, names=['lon', 'lat', 'station'])

    filename = Path(args.filename)
    basename = filename.name
    out_dir = filename.parent.resolve()
    out_dir.mkdir(exist_ok=True)

    im='PGV'



    pgv_data = ims_df.join(stations, how='right')[['lon', 'lat', im]]  # joining 2 dataframes on index "station" name
    pgv_data = pgv_data.fillna(0)

    pgv_array = pgv_data[['lon', 'lat', 'PGV']].to_numpy()

    triFn = Triangulation(pgv_array[:, 0], pgv_array[:, 1])
    linTriFn = LinearTriInterpolator(triFn, pgv_array[:, 2])
    rasterRes = 0.01
    xCoords = np.arange(pgv_array[:, 0].min(), pgv_array[:, 0].max() + rasterRes, rasterRes)
    yCoords = np.arange(pgv_array[:, 1].min(), pgv_array[:, 1].max() + rasterRes, rasterRes)
    print(xCoords.shape)
    print(yCoords.shape)
    zCoords = np.zeros([yCoords.shape[0], xCoords.shape[0]])
    # loop among each cell in the raster extension
    for indexX, x in np.ndenumerate(xCoords):
        for indexY, y in np.ndenumerate(yCoords):
            tempZ = linTriFn(x, y)
            # filtering masked values
            if tempZ == tempZ:
                zCoords[indexY, indexX] = tempZ
            else:
                zCoords[indexY, indexX] = np.nan

    # preliminary representation of the interpolated values
    #plt.imshow(zCoords)

    transform = Affine.translation(xCoords[0] - rasterRes / 2, yCoords[0] - rasterRes / 2) * Affine.scale(rasterRes,
                                                                                                          rasterRes)

    triInterpRaster = rasterio.open('triangleInterpolation.tif',
                                    'w',
                                    driver='GTiff',
                                    height=zCoords.shape[0],
                                    width=zCoords.shape[1],
                                    count=1,
                                    dtype=zCoords.dtype,
                                    # crs='+proj=latlong',
                                    crs=rasterCrs.data,
                                    transform=transform,
                                    )

    triInterpRaster.write(zCoords, 1)
    triInterpRaster.close()
    type(triInterpRaster)


    geometry = [Point(xy) for xy in zip(stations["lon"], stations["lat"])]



    geodata = gpd.GeoDataFrame(stations, crs={"init":rasterCrs.to_string()}, geometry=geometry)

    grid_surface = rioxarray.open_rasterio('triangleInterpolation.tif', masked=True)
    type(grid_surface)
    coastlines_polygon = gpd.read_file('nz_coastlines/nz-coastlines-topo-150k_polygon.shp')
    clipped = grid_surface.rio.clip(coastlines_polygon.geometry.values, rasterCrs.to_string(), drop=False)

    fig, ax = plt.subplots(figsize=(14, 14))

    clipped[0].plot.contour(ax=ax, colors=['black'], linewidths=[0.3])  # draw contour
    clipped.plot(ax=ax, cmap="CMRmap_r", alpha=0.7)  # render clipped surface
    # geodata2.plot(im,ax=ax, legend=True, markersize=1, cmap="jet"); # draw points
    # Download map from online
    cx.add_basemap(ax, crs=rasterCrs.to_string(), source=cx.providers.Stamen.TerrainBackground)  # add basemap

    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize='medium')
    plt.title("PGV at stations", fontsize=12)
#    plt.show()

    fig.savefig(out_dir/f"{basename}_{im}.png")

