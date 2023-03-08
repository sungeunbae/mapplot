# mapplot
# Introduction

Plots numerical/categorical values on the map based on matplotlib/geopandas/pandas etc. Python-native and not dependent on GMT, so it can be easily extended to utilize myriads of Python visualisation tools. It does not need intermediate steps to create an input file - a CSV file can be directly fed to exeucte the code.

## Input
Two mandatory input files are needed.

### IM csv
An IM csv file (or it can be a generic CSV - to be discussed later) is made of two index columns (station name, component) followed by various IM columns (eg. PGV, PGA etc).

 ![Screenshot from 2022-12-14 14-16-37](https://user-images.githubusercontent.com/466989/207481131-f2a2fde3-d5eb-44b2-97c9-6087efdb3113.png)

### Station list file 
A station list file is made of three columns (lon, lat, station name) seperated by a white space.

 ![Screenshot from 2022-12-14 14-22-10](https://user-images.githubusercontent.com/466989/207481959-f16cef91-1476-4a5e-bd18-96dc9ed70526.png)

# Installation
1. `git clone https://github.com/sungeunbae/mapplot.git`
2. Download basemap file and place it in the `data/basemap` directory.
* NZ : https://www.dropbox.com/s/lwq9o1z2wep4vq2/NZ10.tif?dl=0
* Japan: https://www.dropbox.com/s/eplw68s9i3dnpq2/JAPAN09.tif?dl=0
* South Korea: https://www.dropbox.com/s/u5nhkabobwbzmpv/KOREA10.tif?dl=0

3. Use pip or conda to install the following packages.
```
numpy pandas scipy contextily geopandas matplotlib rasterio rioxarray shapely fiona
```
4. Install qcore if not presently installed. https://github.com/ucgmsim/qcore


# Examples
 
 ## Numerical points
 
  ![plot_items_PGA](https://user-images.githubusercontent.com/466989/207484539-1c96633d-f572-4e75-8d90-ec481f07587a.png)
 
 To display numerical values, use `--points` drawing mode. You can specify columns to plot with `--column` option, such as `--column PGA --column PGV`. If not specified, it will plot all available columns. If more than a few columns are plotted, you could use `-n NUM_CORES` to utilize multiple CPU cores. 
 If the column to be plotted is a known IM type, it will automatically place a correct unit, eg) "g" for PGA and "cm/s" for PGV.
 
 ```
 python plot_items.py TeAnau_REL01.csv non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll --title TeAnau --points --column PGA
 ```
 
 
 ## Surface and contours, added SRF faults.
 
 The input CSV file contains discrete points. However, you could interplate this and obtain a smooth 2D surface.
 ![plot_items_pSA_10p0](https://user-images.githubusercontent.com/466989/207482252-3be6ea2e-66a1-4915-be7b-2debdde2f018.png)

To display a surface instead of points, use `--surface` option. `--contour` option adds contour lines. 
It uses Triangulation and LinearTriInterpolator for fast, yet a nice looking surface. While this is generally more efficient than other benchmarked interpolation algorithms (eg. Kriging, RBF), it can take a little longer than points plotting. Consider using `-n` option to best utilize available CPU cores, and `--fast` for slightly lower resolution. 

You can also supply SRF files and display them on the map too.

 ```
 python plot_items.py TeAnau_REL01.csv non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll --title TeAnau -n 4 --fast --srf-file TeAnau_REL01.srf --surface --contours
 ```

## Categorical data

You may have non-numerical categorical data to plot. 

![Screenshot from 2022-12-14 14-48-59](https://user-images.githubusercontent.com/466989/207485134-90d07cd0-4fc7-4798-b34d-e1c99514ffc5.png)

Note that this CSV file is not a standard IM csv, and has no `component` column. You can use `--no-component-column` option to remedy this.
Specify a colormap that has a wide spectrum of different colours (eg. hsv). Each unique category gets assigned a color dynamically. 
To try out a different color map, see https://matplotlib.org/stable/tutorials/colors/colormaps.html

 ![plot_items_Vendors](https://user-images.githubusercontent.com/466989/207482322-93412ae0-46db-4bd5-994a-11223cdb598f.png)

```
 python plot_items.py popular_phones.csv non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll --title "Most Popular Phone Vendors" --no-component-column --colormap hsv --categorical --column Vendors
```

## Support for other countries
By default, it works with NZ, but you can switch to other countires (currently, KR and JP are supported) with `--country` option.
![plot_items_PGA](https://user-images.githubusercontent.com/466989/207758667-6c906774-c491-4219-815a-58c40fb498f2.png)

```
python plot_items.py Pohang.csv Busan_2km_stats_20211001.ll --title Pohang -n 4 --surface-opacity 0.6 --surface --contours --column PGA --country KR --fast
```
One can extend the support for other countries by adding basemap, coastlines ESRI shapefile, and a city csv file.

A basemap can be easily obtained by 

```
import contextily as cx
w, s, e, n = (166,-48.5,178.5,-34)
cx.bounds2raster(w, s, e, n, ll=True, zoom=10, path="NZ10.tif", source=cx.providers.Stamen.TerrainBackground)
```

Coastlines ESRI shapefile was extracted from GSHHG (https://www.soest.hawaii.edu/pwessel/gshhg/) data downloaded from this link: http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip

GSHHS_f_L1.shp contains full resolution coastlines of most countries, and this can be used as itself. However, test proved that cropping out individual countries gives the best performance (15sec -> 0.15sec) for each IM. QGIS can be used to clip the relevant area following these steps.

1. Add a vector layer and import GSHHS_f_L1
2. Create a new shapefile layer, give a name and choose "Polygon" for Geometry type.
3. Right-click on the layer and select Toggle Editing. 
4. Select Add Polygon Feature and draw the area covering the desired area.
5. Vector > Geoprocessing tools > Clip. Select GSHHS_f_L1 layer as Input layer and the new layer as Overlay layer. 
6. If it complains about GSHHS_f_L1 has invalid geometry, Processing> Toolbox > Vector geometry > Check validity. Use GEOS method and extract Valid output. Try Clip again using Valid output as Inpuy layer.
7. Export > Save Features as and save it as a ESRI Shapefile.


city csv file is simply in this format.
```
name,lon,lat
Christchurch,172.63622540000006,-43.5320544
Wellington, 174.77623600000004, -41.2864603
Auckland, 174.76333150000005, -36.8484597
New Plymouth, 174.0752278, -39.0556253
Rotorua, 176.249746, 38.136848
Taupo, 176.070447, -38.684286
...
```

# Detailed Usages

```
usage: plot_items.py [-h] [-t TITLE] [-f OUTPUT_PREFIX] [-s SRF_FILE] [--srf-only-outline] [--srf-outline-color SRF_OUTLINE_COLOR]
                     [--srf-outline-width SRF_OUTLINE_WIDTH] [--srf-colormap SRF_COLORMAP] [--map-height MAP_HEIGHT] [--map-width MAP_WIDTH] [--map-aspect MAP_ASPECT]
                     [--country {NZ,KR,JP}] [--basemap BASEMAP] [--coastline COASTLINE] [--city-csv CITY_CSV] [--disable-city-labels] [--column COLUMN] [-r REGION]
                     [--comp {090,000,ver,H1,H2,geom,rotd50,rotd100,rotd100_50,norm,EAS}] [--colormap COLORMAP] [--categorical] [--points] [--contours]
                     [--point-size POINT_SIZE] [--contour-levels CONTOUR_LEVELS] [--contour-line-width CONTOUR_LINE_WIDTH] [--contour-line-color CONTOUR_LINE_COLOR]
                     [--surface] [--surface-opacity SURFACE_OPACITY] [-n NPROC] [--fast] [--nan-is NAN_IS] [--axis-font-size AXIS_FONT_SIZE]
                     [--title-font-size TITLE_FONT_SIZE] [--colorbar-font-size COLORBAR_FONT_SIZE] [--city-font-size CITY_FONT_SIZE] [--no-component-column]
                     [--station-sep STATION_SEP]
                     data_csv station_file
```

Four different drawing modes are available. 
(1) points (numerical values) 
(2) surface (2D interpolated from point data) 
(3) categorical 
(4) contours 

`--contours` can be added alongside `--points` or `--surface`. While `--categorical` can not be mixed with other drawing types. 

Choose a drawing mode, and stick to default values. See the initial output, and explore all the options to best-suit your plotting requirements.

```


positional arguments:
  data_csv              Path to csv file to plot (eg. IM csv). Each line is station name, (optional: component), column_values...
  station_file          station list file. name, lon, lat with a space as the delimiter by default

optional arguments:
  -h, --help            show this help message and exit
  -t TITLE, --title TITLE
                        title text
  -f OUTPUT_PREFIX, --output-prefix OUTPUT_PREFIX
                        output path combined with filename prefix. eg. XXXX/YYYY in XXXX/YYYY_PGA.png
  -s SRF_FILE, --srf-file SRF_FILE
                        SRF files to plot, use quoted wildcards, repeat as needed
  --srf-only-outline    Only plot outline for SRF files
  --srf-outline-color SRF_OUTLINE_COLOR
                        Outline color of SRF outline. See https://matplotlib.org/stable/tutorials/colors/colors.html
  --srf-outline-width SRF_OUTLINE_WIDTH
                        Outline width of SRF outline.
  --srf-colormap SRF_COLORMAP
                        Colormap used for SRF surface
  --map-height MAP_HEIGHT
                        Height of the output map in inches
  --map-width MAP_WIDTH
                        Width of the output map in inches
  --map-aspect MAP_ASPECT
                        Aspect of Height / Width of the plotted map
  --country {NZ,KR,JP}
  --basemap BASEMAP     Path to the local basemap
  --coastline COASTLINE
                        ESRI shapefile (multipolygon) of coastline definition
  --city-csv CITY_CSV   City locations CSV file: name, lon, lat
  --disable-city-labels
                        Flag to disable city_labels - these are plotted by default
  --column COLUMN       Column names to plot. If unspecified, all included. Repeat as needed
  -r REGION, --region REGION
                        Region to plot in the form min_lon/max_lon/min_lat/max_lat.
  --comp {090,000,ver,H1,H2,geom,rotd50,rotd100,rotd100_50,norm,EAS}
                        Component in IM file to plot
  --colormap COLORMAP   CPT to use for overlay data. Use '_r' to invert. See https://matplotlib.org/stable/tutorials/colors/colormaps.html
  --categorical         Display categorical/discrete values as points. Recommended to use with a cyclinc/qualitative colormaps (eg) --colormap hsv
  --points              Display values as points
  --contours            Add contour lines from CPT increments
  --point-size POINT_SIZE
                        Point size for points/categorical
  --contour-levels CONTOUR_LEVELS
                        number of contour levels
  --contour-line-width CONTOUR_LINE_WIDTH
                        contour line width
  --contour-line-color CONTOUR_LINE_COLOR
                        contour line color
  --surface             interpolated grid surface
  --surface-opacity SURFACE_OPACITY
                        overlay opacity: transparent(0) - opaque(1)
  -n NPROC, --nproc NPROC
                        max number of processes
  --fast                Run faster with less resolution for interpolated surface
  --nan-is NAN_IS       NaN is replaced by specified value (eg. 0). By default, NaN is removed
  --axis-font-size AXIS_FONT_SIZE
  --title-font-size TITLE_FONT_SIZE
  --colorbar-font-size COLORBAR_FONT_SIZE
  --city-font-size CITY_FONT_SIZE
  --no-component-column
                        If data_csv doesn't have component column, set this option
  --station-sep STATION_SEP
                        the delimiter used in the station file

```
 
