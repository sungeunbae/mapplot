import rasterio
from rasterio.crs import CRS
from rasterio.plot import show
from rasterio.transform import Affine

import rioxarray
import contextily as cx

import xarray.plot

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from matplotlib.lines import Line2D
import math


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
from qcore.im import IM
from qcore import srf

script_dir = Path(__file__).parent.resolve()

MAP_WIDTH = 7

CITIES = {"Christchurch": (172.63622540000006, -43.5320544),
          "Wellington": (174.77623600000004, -41.2864603),
          "Auckland": (174.76333150000005, -36.8484597),
          "New Plymouth": (174.0752278, -39.0556253),
          "Rotorua": (176.249746, 38.136848),
          "Taupo": (176.070447, -38.684286),
          "Napier": (176.91201780000006, -39.4928444),
          "Palmerston North": (175.60821450000003, -40.3523065),
          "Masterton": (175.6573502, -40.9511118),
          "Nelson": (173.283965, -41.270632),
          "Blenheim": (173.96125, -41.513443),
          "Kaikoura": (173.679911, -42.399448),
          "Tekapo": (170.477121, -44.004674),
          "Timaru": (171.254973, -44.396972),
          "Queenstown": (168.662644, -45.031162),
          "Dunedin": (170.5027976, -45.8787605),
          "Haast": (169.042437, -43.881107),
          "Greymouth": (171.21076229999994, -42.4503925),
          "Westport": (171.60589, -41.754522)

          }


# TODO:
# Crop the map : Done
# Color size : Done
# Set the aspect / size
# Draw SRF or a box
# Roads


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
    arg("-r", "--region", help="Region to plot in the form xmin/xmax/ymin/ymax.")
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
    arg(
        "--grid-contours",
        help="add contour lines from CPT increments",
        action="store_true",
    )
    # arg(
    #     "--xyz-grid-contours-inc",
    #     help="add contour lines with this increment",
    #     type=float,
    # )
    arg("--grid-surface", help="interpolated grid surface", action="store_true")
    # arg(
    #     "--xyz-grid-search",
    #     help="search radius for interpolation eg: 5k (only m|s units for surface)",
    # )
    # arg("--labels-file", help="file containing 'lat lon label' to be added to the map")

    arg(
        "--disable-city-labels",
        dest="enable_city_labels",
        help="Flag to disable city_labels - these are plotted by default",
        default=True,
        action="store_false",
    )

    # arg(
    #     "--disable-roads",
    #     dest="enable_roads",
    #     help="Flag to disable roads/highways - these are plotted by default",
    #     default=True,
    #     action="store_false",
    # )
    arg("-n", "--nproc", help="max number of processes", type=int, default=1)
    # arg("-d", "--dpi", help="render DPI", type=int, default=300)
    # arg(
    #     "--downscale",
    #     help="DPI render multiplier for better pixel accuracy",
    #     type=int,
    #     default=8,
    # )

    arg(
        "--fast",
        help="Run faster with slightly less resolution for interpolated surface",
        action="store_true"
    )

    arg(
        "--crop-na",
        help="Crop out no data area. Faster processing",
        action="store_true"
    ),

    arg(
        "imcsv", help="path to im csv file"
    )

    arg(
        "station", help="station list. name, lon, lat with a space as the delimiter by default"
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


def city_labels(ax):
    lons = []
    lats = []
    for city in CITIES:
        lon, lat = CITIES[city]
        ax.annotate(city, xy=(lon, lat), xytext=(3, 3), textcoords="offset points")
        lons.append(lon)
        lats.append(lat)
    ax.plot(lons, lats, 'o')


def triangle_interpolation(llv_array, tiff_name, crs, overwrite=True, res=0.01):
    #also see https://www.delftstack.com/api/scipy/2d-interpolation-python/
    #https://stackoverflow.com/questions/44922766/2d-linear-interpolation-data-and-interpolated-points
    #https://pythonguides.com/matplotlib-2d-surface-plot/

    if Path(tiff_name).exists() and not overwrite:
        return

    triFn = Triangulation(llv_array[:, 0], llv_array[:, 1])
    linTriFn = LinearTriInterpolator(triFn, llv_array[:, 2])

    xCoords = np.arange(llv_array[:, 0].min(), llv_array[:, 0].max() + res, res)
    yCoords = np.arange(llv_array[:, 1].min(), llv_array[:, 1].max() + res, res)
    # print(xCoords.shape)
    # print(yCoords.shape)
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

    transform = Affine.translation(xCoords[0] - res / 2, yCoords[0] - res / 2) * Affine.scale(res,res)

    triInterpRaster = rasterio.open(tiff_name,
                                    'w',
                                    driver='GTiff',
                                    height=zCoords.shape[0],
                                    width=zCoords.shape[1],
                                    count=1,
                                    dtype=zCoords.dtype,
                                    # crs='+proj=latlong',
                                    crs=crs,
                                    transform=transform,
                                    )

    triInterpRaster.write(zCoords, 1)
    triInterpRaster.close()

def load_srf(srf_file):
    seg_llvs = srf.srf2llv_py(srf_file)
    # planes = srf.read_header(srf_file, idx=True)
    bounds = srf.get_bounds(srf_file)
    np_bounds = np.array(bounds)
    n_plane = len(bounds)
    cpt_percentile=95
    all_vs = np.concatenate((seg_llvs))[:, -1]
    percentile = np.percentile(all_vs, cpt_percentile)
    # round percentile significant digits for colour pallete
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


def plot_property(xyz, prop_name, crs, clip_with, outdir, prefix, enable_city_labels=True,
                  region=None,
                  grid_surface=True,
                  grid_contours=True, basemap=True,
                  local_basemap= script_dir /"NZ10.tif",
                  cmap="CMRmap_r", # https://matplotlib.org/stable/tutorials/colors/colormaps.html
                  interp_overwrite=True,
                  fast=False,
                  crop_na=False):
    # print(f"{prop_name} {crs} {clip_with} {grid_surface} {grid_contours} {basemap} {local_basemap}")

    if crop_na:
        # xyz = xyz.fillna(0) # no data gets 0
        xyz = xyz.loc[xyz[prop_name].notna()]

    if region is not None:
        xmin, xmax, ymin, ymax = region
        xyz = xyz.loc[(xyz['lon'] >= xmin) & (xyz['lon'] <= xmax) & (xyz['lat'] >= ymin) & (xyz['lat'] <= ymax)]

    if grid_surface:

        llv_array = xyz[['lon', 'lat', prop_name]].to_numpy()

        surface_tiff = outdir / f'triangleInterpolation_{prop_name}.tif'
        triangle_interpolation(llv_array, surface_tiff, {"init": crs}, overwrite=interp_overwrite, res=0.03 if fast else 0.001)

    else:
        xyz = xyz.loc[xyz[prop_name].notna()]

    geometry = [Point(xy) for xy in zip(xyz["lon"], xyz["lat"])]

    #    geodata = gpd.GeoDataFrame(stations, crs={"init":rasterCrs.to_string()}, geometry=geometry)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect(1)

    # Playing with SRF. srf->grdfile then loaded onto QGIS, masked, converted to tiff.
    # srf_surface = rioxarray.open_rasterio(script_dir / "TeAnau/seg0.tif", masked = True)
    # srf_surface[0].plot.pcolormesh(ax=ax, add_colorbar=False, zorder=10, alpha=1)
    # l=Line2D([168,170],[-46,-44],linewidth=3, linestyle='--', zorder=10)
    # ax.add_line(l) # '-', '--', '-.', ':', '',
    seg_llvs, bounds, regions,  cpt_max = load_srf("/Users/sungbae/mapplot/TeAnau/TeAnau_REL01.srf")
    for i in range(len(seg_llvs)):
        srf_tiff = outdir / f'srf{i}.tif'
        triangle_interpolation(seg_llvs[i],srf_tiff, {"init": crs}, res=0.001)
        srf_surface = rioxarray.open_rasterio(srf_tiff, masked=True)
        srf_surface.plot(ax=ax,cmap="afmhot_r", zorder=10, vmax=cpt_max, add_colorbar=False) 

    for bound in bounds:
        vertices=np.array(bound+[bound[0]]) # add the first vertex
        xs = vertices[:, 0]
        ys = vertices[:, 1]
        for i in range(len(xs)-1):
            l=Line2D([xs[i],xs[i+1]],[ys[i],ys[i+1]],linewidth=2, linestyle='-', zorder=10)
            ax.add_line(l)



    if grid_surface:

        grid_surface = rioxarray.open_rasterio(surface_tiff, masked=True)

        # clipped: xarray.DataArray
        clipped = grid_surface.rio.clip(clip_with, crs, drop=False)
        if grid_contours:
            clipped[0].plot.contour(ax=ax, colors=['black'], linewidths=[0.3])  # draw contour
        # render clipped surface
        # cbar_kwargs : https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.colorbar

        # clipped[0].plot.pcolormesh(ax=ax, cmap=cmap, alpha=0.8,
        #                            cbar_kwargs={"orientation":"horizontal",
        #                                         "anchor":(0.4,1.5), # Try less than 1.5 for more spacing
        #                                         "label":"PGA (g)"})

        # for more customization, try below.
        p = clipped[0].plot.pcolormesh(ax=ax, cmap=cmap, alpha=0.8, add_colorbar=False)

        # cbar = plt.colorbar(p, ax=ax, orientation="horizontal", ticklocation='bottom')
        cax = ax.inset_axes([0.0, -0.1, 1, 0.05])
        cbar = plt.colorbar(p, ax=ax, cax=cax, orientation="horizontal", ticklocation='bottom', shrink=0.8)
        # unit = "g" #
        imname_prefix = prop_name.split("_")[0]  # in case we have pSA_0.1 etc
        unit = IM(imname_prefix).get_unit()
        cbar.set_label(f"{prop_name} ({unit})", fontsize=15)

    else:
        geodata = gpd.GeoDataFrame(xyz, crs={"init": crs}, geometry=geometry)
        geodata.plot(prop_name, ax=ax, legend=True, markersize=1, cmap=cmap)  # draw points

    if basemap:
        # add basemap
        if local_basemap is not None:

            # cx.add_basemap(ax, crs=crs, source=cx.providers.Stamen.TerrainLabels, zoom="auto")

            cx.add_basemap(ax, crs=crs, source=local_basemap, zoom=8)  # add basemap
            # cx.add_basemap(ax, crs=crs, source="NZ10_roads.tif", zoom="10") #not quite good

            # this map was obtained by the following lines
            # w, s, e, n = (166,-48.5,178.5,-34)
            # _ = cx.bounds2raster(w, s, e, n,ll=True, zoom=10, path="NZ10.tif",
            # source=cx.providers.Stamen.TerrainBackground)

            # Possible Map styles are here: https://xyzservices.readthedocs.io/en/stable/gallery.html
        else:  # Download map from online (slow and quality may be low)
            cx.add_basemap(ax, crs=crs, source=cx.providers.Stamen.TerrainBackground, zoom="auto")


    if enable_city_labels:
        city_labels(ax)



    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    # ax.legend(loc='lower center',mode='expand')

    plt.title(f"{prop_name} at stations", fontsize=20)
    #    plt.show()

    prop_name_p = prop_name.replace(".", "p")

    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_{prop_name_p}.png")
    plt.close(fig)


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

    # TODO: should select relevant component
    ims = list(ims_df.columns)[1:]  # [1:] to remove component column
    xyz_df = ims_df.join(stations, how='right')[['lon', 'lat'] + ims]  # joining 2 dataframes on index "station" name.
    coastlines_polygon = gpd.read_file(script_dir / 'nz_coastlines/nz-coastlines-topo-150k_polygon.shp')

    # for im in ims:
    #
    #     plot_property(xyz_df, im, rasterCrs.to_string(),coastlines_polygon.geometry.values,
    #                   out_dir, basename, grid_surface=args.grid_surface, grid_contours=args.grid_contours)

    print(args.nproc)
    pfn = partial(plot_property, xyz_df, crs=rasterCrs.to_string(), clip_with=coastlines_polygon.geometry.values,
                  outdir=out_dir, prefix=basename,
                  enable_city_labels=args.enable_city_labels,
                  region=args.region,
                  grid_surface=args.grid_surface, grid_contours=args.grid_contours,
                  fast=args.fast, interp_overwrite=True, crop_na=args.crop_na)


    with Pool(args.nproc) as pool:
        plot_properties = pool.map_async(pfn, ims[:1])

        plot_properties = plot_properties.get()
