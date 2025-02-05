# Digital Elevation Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13272772.svg)](https://doi.org/10.5281/zenodo.13272772)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen)](https://opensource.org/license/mit)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/numpy)](https://python.org)

Contents
- [Funding & Support](#funding---support)
- [Installation](#installation)
- [Configuration](#configuration)
- [Scripts](#scripts)
  * [Mosaic Strips](#mosaic-strips)
  * [Running](#running)
  * [Coregistration of DEM to csv](#coregistration-of-dem-to-csv)
  * [Compute Inundation](#compute-inundation)
  * [Correct DEMs with ICESat-2.py](#correct-dems-with-icesat-2py)
  * [Find Cloudy DEMs](#find-cloudy-dems)
  * [Global DEMs](#global-dems)
    <!-- + [Plot_Angles.py](#plot-anglespy) -->
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [To Do](#to-do)

## Funding & Support

This work was funded by the NASA Sea Level Change Team (N-SLCT) and performed at the Colorado Center for Astrodynamics Research (CCAR), part of the Ann & H.J. Smead Department of Aerospace Engineering at the University of Colorado Boulder.

<p float "middle">
    <a href="https://www.colorado.edu/aerospace">
        <img src="https://raw.githubusercontent.com/EduardHeijkoop/EduardHeijkoop.github.io/refs/heads/main/Assets/Images/CU_Smead_Logo.png" height=200 alt="CU Aerospace Engineering"/>
    </a>
    <a href="https://www.colorado.edu/ccar">
        <img src="https://raw.githubusercontent.com/EduardHeijkoop/EduardHeijkoop.github.io/refs/heads/main/Assets/Images/CCAR_Logo.jpg" height=200 alt="Colorado Center for Astrodynamics Research"/>
    </a>
</p>


<p float "middle">
    <a href="https://sealevel.nasa.gov">
        <img src="https://raw.githubusercontent.com/EduardHeijkoop/EduardHeijkoop.github.io/refs/heads/main/Assets/Images/NSLCT_Logo.png" height=200 alt="NASA Sea Level Change Team" />
    </a>
</p>


## Installation

Clone the repository:

    git clone https://github.com/EduardHeijkoop/DEM.git

Create a new conda environment with the right packages:

    conda env create -f dem.yml

### Ames Stereo Pipeline
The `Mosaic_Strips.py` tool requires `dem_mosaic` from [NASA Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/en/latest/index.html). To install, follow [these steps](https://stereopipeline.readthedocs.io/en/latest/installation.html):

#### Linux

Download the latest stable version (3.4.0):

    wget https://github.com/NeoGeographyToolkit/StereoPipeline/releases/download/3.4.0/StereoPipeline-3.4.0-2024-06-19-x86_64-Linux.tar.bz2
Unzip with `tar`:

    tar xvf StereoPipeline-3.4.0-2024-06-19-x86_64-Linux.tar.bz2
Add to `bashrc`:

    echo "export PATH=\"</ASP/install/dir>/StereoPipeline-3.4.0-2024-06-19-x86_64-Linux/bin:\$PATH\"" >> ~/.bashrc

Test the installation by running:

    dem_mosaic -h

#### macOS

On Apple Silicon ASP requires [Rosetta 2](https://support.apple.com/en-us/102527). Download the latest stable version (3.4.0):

    wget https://github.com/NeoGeographyToolkit/StereoPipeline/releases/download/3.4.0/StereoPipeline-3.4.0-2024-06-19-x86_64-OSX.tar.bz2

Unzip with `tar`:

    tar xvf StereoPipeline-3.4.0-2024-06-19-x86_64-OSX.tar.bz2

Add to `zshrc`:

    echo "export PATH=\"</ASP/install/dir>/StereoPipeline-3.4.0-2024-06-19-x86_64-OSX/bin:\$PATH\"" >> ~/.zshrc

Test the installation by running:

    dem_mosaic -h


Make sure you have an active [NASA EarthData](https://urs.earthdata.nasa.gov/) account. Other Unix CLI tools that are needed are: `wget` and `unzip`.

## Configuration

### Config File
The scripts in this repository depend on a number of other files & variables that need to be defined in the configuration file. Edit the `dem_config.ini` file in this repository or create your own and point to that when you run the script. The structure (e.g. `MOSAIC_CONSTANTS`) must be the same as the original. The common entries will be listed here and the entries specific to a particular script will be listed in the respective section.

- `GENERAL_PATHS/tmp_dir` : Path to a directory where temporary files will be written to. All temporary files will be deleted upon completion, but in case of errors it may be useful to see which files were or weren't created.
- `GENERAL_PATHS/gsw_dir` : 
- `GENERAL_PATHS/landmask_c_file` : Point to the `pnpoly_function.c` file that is used to run the landmask algorithm. This repository comes with a file in the `C_Code` directory, so easiest is to point to that. A corresponding `.so` file will be made, ***do not*** point to that.
- `GENERAL_PATHS/osm_shp_file` : OpenStreetMap's land polygons are used to define the boundary between land and water. Click [here](https://osmdata.openstreetmap.de/download/land-polygons-complete-4326.zip) to download the zip file. Extract this and point to the `land_polygons.shp` file.

- `GENERAL_CONSTANTS/earthdata_username` : Enter your NASA EarthData username here.

## Scripts

### Mosaic Strips

Given a set of overlapping DEM tiles (often called strips), this script will stitch them all together into a (vertically) coherent product.

#### Running

    python Mosaic_Strips.py [optional flags]

<p float "middle">
    <a>
        <img src="https://raw.githubusercontent.com/EduardHeijkoop/EduardHeijkoop.github.io/refs/heads/main/Assets/Images/Mosaic_Result.png" alt="Mosaic Results" />
    </a>
</p>

##### Optional Flags

- `--config` `<configuration_file>` : Path to configuration file. Default is dem_config.ini in the script's directory.
- `--list` `<list_file>` : Specify a list of files to be turned into a mosaic. This can be useful for testing or using a subset of a directory's files. This can be used with an empty input file, as it effectively appends the list as an extra row to the input file.
- `--output_dir` `<output_dir>` : To be used in conjunction with a list_file. Specifies the output directory of the mosaic to be created from the list file.
- `--loc_name` `<loc_name>` : To be used in conjunction with a list file. Specifies the name of the mosaic to be created from the list file.
- `--dir_structure` `<dir_structure>` : Specifies the directory structure that should be expected. The input directory can either all be in the input directory (use "simple"), in the "sealevel" structure (explained below), or when "scenes" is used all files named "*dem.tif" in the subdirectories are used.
- `--gsw` `<gsw_path>` : Point to a specific (global) surface water product to use. Note, this product will be selected for all entries in the input file, so best to use with either a single entry or with the manual input.
- `--N_cpus` `<N_cpus>` : How many CPU cores to use. Each generation of alignment may include independent steps, which can be run in parallel.
- `--horizontal` : This flag will turn on horizontal shifting. There may be geolocation errors in each individual strip, and incorporating relative shift in x and y directions may improve the fit of the alignment, at the cost of (significantly) increased run time.
- `--cloud_water_filter` : This will include results from [Find_Cloudy_DEMs.py](#correct-dems-with-icesat-2py). This allows for exclusion of strips that are too cloudy or too much over open water.
- `--corrected` : This flag is used to select corrected strips only. If [Correct_DEM_Strip_ICESat2.py](#correct-dems-with-icesat-2py) was run on the strips, this flag will use the corrected strips rather than the original ones.
- `--all_strips` : This flag will use all strips to create a mosaic and not perform any geometric filtering.
- `--no_gsw` : Skip the (global) surface water filtering. Applicable when a mosaic is built (almost) entirely over land, rather than a coastal region.
- `--simplify` : Simplify the geometry used to populate points for strip-to-strip alignment. If the POLYGON_SIMPLIFY_VALUE in the configuration file is smaller than X_SPACING/Y_SPACING, this should not lead to accuracy loss. 

#### Configuration

The first step is to create an input file (with its path listed in the configuration file), which gives the location of the files to build the mosaic and the location where to output the final product:

<p float "middle">
    <a>
        <img src="https://raw.githubusercontent.com/EduardHeijkoop/EduardHeijkoop.github.io/refs/heads/main/Assets/Images/Mosaic_Input.png" alt="Mosaic Input File" />
    </a>
</p>

The following list briefly explains the functionality of each relevant parameter in the configuration file.

- `GENERAL_PATHS/gsw_dir` : Directory where extent tiles of the Global Surface Water [Pekel et al., 2016](https://www.nature.com/articles/nature20584) are located. [This link](https://global-surface-water.appspot.com/download) has a number of ways to download the dataset.
- `MOSAIC_PATHS/input_file` : 
- `MOSAIC_CONSTANTS/POLYGON_AREA_THRESHOLD` :  250.0
- `MOSAIC_CONSTANTS/POLYGON_SIMPLIFY_VALUE` :  10.0
- `MOSAIC_CONSTANTS/STRIP_AREA_THRESHOLD` :  4e6
- `MOSAIC_CONSTANTS/GSW_POCKET_THRESHOLD` :  0.01
- `MOSAIC_CONSTANTS/GSW_CRS_TRANSFORM_THRESHOLD` :  0.05
- `MOSAIC_CONSTANTS/GSW_OVERLAP_THRESHOLD` :  0.95
- `MOSAIC_CONSTANTS/STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD` :  0.01
- `MOSAIC_CONSTANTS/STRIP_CONTAINMENT_THRESHOLD` :  0.75
- `MOSAIC_CONSTANTS/STRIP_DELTA_TIME_THRESHOLD` :  0
- `MOSAIC_CONSTANTS/STRIP_CLOUD_THRESHOLD` :  0.15
- `MOSAIC_CONSTANTS/STRIP_WATER_THRESHOLD` :  0.75
- `MOSAIC_CONSTANTS/N_STRIPS_CONTAINMENT` :  2
- `MOSAIC_CONSTANTS/AREA_OVERLAP_THRESHOLD` :  2.5e5
- `MOSAIC_CONSTANTS/GSW_INTERSECTION_THRESHOLD` :  0.667
- `MOSAIC_CONSTANTS/X_SPACING` :  20.0
- `MOSAIC_CONSTANTS/Y_SPACING` :  20.0
- `MOSAIC_CONSTANTS/X_MAX_SEARCH` :  12.0
- `MOSAIC_CONSTANTS/Y_MAX_SEARCH` :  12.0
- `MOSAIC_CONSTANTS/MOSAIC_TILE_SIZE` :  25000.0


When the `sealevel` option is used for the `--dir_structure` flag, the following directory structure is required:

```bash
└── WV0N_YYYYMMDD_*/
    └── strips/
        ├── WV0N_YYYYMMDD_*dem.tif
        └── WV0N_YYYYMMDD_*dem_smooth.tif
```

When `simple` is used for this flag, only files in the input directory will be used. When `scenes` is used, the subdirectories will be recursively searched for files that end with `dem.tif`. 

The strips used to build the mosaic must satisfy a few conditions in terms of filenames. They must start with `WV0N` (where `N` is 1, 2 or 3, denoting the WorldView satellite used to acquire the images). Alternatively, they can start with `GE01` when GeoEye-1 was used for the acquisition. Next, the date of acquisition must be attached to the satellite idenitifier with one underscore and in the YYYYMMDD format. A filename that starts with `WV02_20210507` then represents a strip acquired by WorldView-2 on May 7th. The timestamp is required to apply the appropriate weights to the minimum spanning tree when building the mosaic.


### Coregistration of DEM to csv

This script iteratively co-registers a raster to a csv. The most relevant use here is to co-register a previously built mosaic to ICESat-2 data. 

#### Running

    python Simple_Coregistration.py --raster </path/to/raster> --csv </path/to/csv> [optional flags]

##### Mandatory flags:

- --raster <raster path> : Path to raster that is to be co-registered to the csv. 
- --csv <csv path> : Path to csv with ground truth. Currently, the csv lon, lat and height values must be relative to the WGS 84 ellipsoid. First to columns of the csv must be longitude and latitude, in that order.

The output file will be a co-registered product in the same directory as the input raster file, with the amount shifted appended to its name. For example, a mosaic `Sudan_PortSudan_Full_Mosaic_0_32637.tif` will yield `Sudan_PortSudan_Full_Mosaic_0_32637_Shifted_z_2p07m.tif`.

<p float "middle">
    <a>
        <img src="https://raw.githubusercontent.com/EduardHeijkoop/EduardHeijkoop.github.io/refs/heads/main/Assets/Images/Coregistration_Result.png" alt="Coregistration Results" />
    </a>
</p>

##### Optional Flags

- `--median` : Flag to select zero median rather than zero mean as target for the co-registration.
- `--sigma` `<value>` : How many standard deviations to use to detect outliers. Value must be an integer. Default is 2. 
- `--threshold` `<value>` : Iterative threshold to detect when convergence has been achieved. Default value is 0.05 m.
- `--resample` : Flag to sample the newly co-registered raster with the original input csv; this is in addition to the sampling of the points after outlier removal.
- `--keep_original_sample` : Keep csv file that samples the original raster.
- `--no_writing` : Don't sample the newly co-registered file with the csv after outlier removal.
- `--nodata`` <nodata value>` : Define value for nodata in the raster. Default is -9999.
- `--print` : Print statistics results to the screen.
- `--write_file` `<file to write statistics to>` : Write statistics results to a file.
- `--output_dir` `<output directory>` : Create file in this directory. Default is the directory the raster is in.
- `--N_iterations` `<number of iterations>` : Define maximum number of iterations, before iteration is stopped if convergence is not achieved. Default is 15.

### Compute Inundation

Once a DSM has been aligned in the vertical, it can be used to compute inundation projections.

#### Running

    python Compute_Inundation.py --input_file <input_dem> --grid_extents <lon_min lon_max lat_min lat_max> --coastline <coastline_file> [mandatory flags] [optional flags]

##### Mandatory Flags

- `--input_file` <`input_dem`> : Input digital elevation model to compute inundation with. Must reference the same datum as the sea level grid used to calculate mean sea level.
- `--grid_extents` `<lon_min lon_max lat_min lat_max>`: Technically not mandatory, but used to subset a global or larger sea level grid that is used as input. If omitted, the full grid (which may be global!) is used.
- `--coastline` <`coastline_file`> : Path to file defining the coastline, onto which sea level (extremes) are interpolated. 

##### Optional Flags

- `--loc_name` <`loc_name`> : Name of location being considered. This name will be the base for all output files. If nothing is defined, the name will be determined from the input DEM file.
- `--config` <`config_file`> : Path to configuration file to read constants, paths, ancillary files, etc. If not defined, the default `dem_config.ini` in the same directory as the script will be used.
- `--N_cpus` <`N_cpus`> : Number of CPU threads to use to parallelize the inundation computation.
- `--downsample_res` <`downsample_res`> : Final output resolution of the inundation. Must be coarser than the input file. If not defined it will default to the native resolution of the input file.
- `--geoid` <`geoid_file`> : Path to geoid file to perform all calculations will orthometric heights, rather than the default ellipsoidal heights.
- `--vlm` <`vlm_file`/`vlm_value`> : Either (a) path to vlm file (GeoTiff) that defines a spatially varying vertical land motion (e.g. from InSAR), or (b) a single value (in m/yr) that is applied to the entire grid. This flag (both grid and value) can only be used in conjunction with the temporally varying AR6 projections, and is incompatible with single values for sea level change.
- `--clip_vlm` : Toggle to clip the input DEM file to the extents of the given VLM file.
- `--t0` <`t0`> : Select a t0, relative to which the DEM will be augmented in time with VLM to match the timestamps of AR6. Default is 2020.
- `--clip_coast` : Toggle to clip the input DEM file to the input coastline file.
- `--ssp` <`ssp`> : Select which AR6 SSP (Shared Socioeconomic Pathway) to use for projected sea level rise. Must be used in conjunction with the `--years` flag.
- `--years` <`year`> : Select which year(s) to calculate sea level rise at, using the selected SSP. Multiple years may be given, but they must all be decades between 2020 and 2150, e.g. 2080 and not 2078.
- `--confidence` <`confidence_level`> : Select a confidence level (medium or low) for AR6 projections to be used.
- `--slr` <`sealevel_rise`> : Set amount(s) of sea level rise to compute inundation at. Incompatible with `--ssp`.
- `--return_period` <`return_period`> : Select return period (in years) of sea level extreme from [Muis et al. (2020)](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2020.00263/full). Values must be 2, 5, 10, 25, 50, 100, 250, 500 or 1000 years. Incompatible with FES-derived high tide values.
- `--fes2014` : Use FES2014 model-derived highest astronomical tides (HAT). 
- `--fes2022` : Use FES2022 model-derived highest astronomical tides (HAT). 
- `--mhhw` : Use mean higher high water (MHHW) instead of HAT, must use a `--fes2014/22` flag. 
- `--high_tide` <`high_tide`> : Use a set value for high tide, e.g. derived from a tide gauge.
- `--connectivity` : Create separate inundation files that are connected to the ocean.
- `--separate` : Create additional files of inundation pockets that are disconnected.
- `--surface_water` <`surface_water_file`> : Select user-defined surface water file to do the connectivity analysis with. If not given, dataset from [Pekel et al. (2016)](https://www.nature.com/articles/nature20584) will be used.
- `--uncertainty` : Toggle to compute uncertainty +/- N sigma around the median projection, defined by the `--sigma` flag below.
- `--sigma` <`sigma`> : Define how many standard deviations away from the mean to calculate uncertainty. Must be used in conjunction with `--uncertainty`.
- `--of` <`output_format`> : Define the output file format. Default will be shapefile, but GeoJSON is an option too.


#### Configuration

- `INUNDATION_PATHS/sealevel_grid` : Path to sea level grid for mean sea level. Must be a GeoTiff with the same datum as the input DEM.
- `INUNDATION_PATHS/AR6_dir` : Path to directory containing AR6 sea level projections. Directory must contain a "Regional" subdirectory, which itself contains "low_confidence" and "medium_confidence" directories with the SSP projections.
- `INUNDATION_PATHS/CODEC_file` : Path to CoDEC file from which to compute different sea level extremes, based on desired return period (as defined with the `--return_period` flag). Available [here](https://zenodo)
- `INUNDATION_PATHS/fes2014_file` : Path to FES2014 file with at least the columns lon, lat, tide_max and MHHW. Can be computed with the [Compute_Global_FES_Grid.py script](https://www.github.com/EduardHeijkoop/Utilities/Create_Global_FES_Grid.py) 
- `INUNDATION_PATHS/fes2022_file` : Same as above, but for FES2022.
- `VLM_CONSTANTS/VLM_NODATA` : Nodata value embedded in VLM file.
- `INUNDATION_CONSTANTS/INTERPOLATE_METHOD` : Method to interpolate points (e.g. from FES). Options are "Smooth" or "LSQ", with "Smooth" recommended.
- `INUNDATION_CONSTANTS/GRID_NUM_THREADS` : Number of threads to use when running `gdal_grid`.
- `INUNDATION_CONSTANTS/GRID_INTERMEDIATE_RES` : Intermediate spatial resolution (in meters) of grids, which are then upsampled to the native resolution of the input DEM.
- `INUNDATION_CONSTANTS/GRID_ALGORITHM` : Algorithm used to run `gdal_grid`. Recommended to use "invdistnn".
- `INUNDATION_CONSTANTS/GRID_SMOOTHING` : Smoothing value when running `gdal_grid`. Recommended to use 0.0.
- `INUNDATION_CONSTANTS/GRID_POWER` : Power value when running `gdal_grid`. Recommended to use 2.0.
- `INUNDATION_CONSTANTS/GRID_NODATA` : Nodata vlaue for both input DEM grid and intermediate grids obtained throught `gdal_grid`.
- `INUNDATION_CONSTANTS/GRID_MAX_PTS` : Max points value when running `gdal_grid`. Recommended to use at least 100.
- `INUNDATION_CONSTANTS/INUNDATION_NODATA` : Nodata value when running `gdal_calc` to compute inundation. Recommended to use 0.0 for ease of `gdal_polygonize` that follows.
- `INUNDATION_CONSTANTS/GSW_BUFFER` : Buffer distance (inland) of the surface water product to compute connectivity.
- `INUNDATION_CONSTANTS/REGGRID_INTERPOLATE_METHOD` : Method to interpolate regular grids (with SciPy RegularGridInterpolator), either sea level or geoid, onto irregular points, like a coastline. Recommended to use "linear".
<!-- - `INUNDATION_CONSTANTS/INUNDATION_GRIDDING_RESOLUTION` : 10.0 -->


### Correct DEMs with ICESat-2.py

Optional pre-processing step before creating a mosaic.

### Find Cloudy DEMs 


### Global DEMs

#### Running

    python Global_DEMs.py --product <> --extents <lon_min lon_max lat_min lat_max> [optional flags]

Mandatory flags:

- `--product` `<dem product>` : Select which DEM product to download. Options are "srtm", "aster" or "copernicus".
- `--extents` `<lon_min lon_max lat_min lat_max>` : Define the spatial extents of the requested DEM.

#### Optional Flags

- `--config` `<configuration_file>` : Path to configuration file. Default is dem_config.ini in the script's directory.
- `--copy_nan` : Flag to copy NaN values from the DEM when converting to WGS 84 heights; this is particularly useful when working in coastal regions.
- `--datum` `<dem datum>` : Select which vertical datum to use for the DEM. Geoid heights are EGM96 for SRTM and ASTER, and EGM2008 for Copernicus. Options are "geoid" or "wgs84".  Default is "wgs84". 
- `--output_file` `<output file>` : Define output file. Default is "tmp.tif".

#### Configuration

If the Copernicus DEM is selected, the AWS CLI tool must be installed, as individual tiles are obtained from `s3://copernicus-dem-30m/`.

<!-- ### Plot_Angles.py

Given a set of .ntf and their associated xml files, will plot spatial extent of the image and the satellite's look angle. This can be a visual aid to get the best set of files to build a DSM.

Usage:

    python Plot_Angles.py --input_file <input file> [--coast <coastline file>]

The input file will contain a list of full paths to the different NTF files to be plotted, without a header line. This requires each .ntf/.NTF file to have an associated .xml file with the same file name in the same directory. Adding a coastline is optional, but may aid in visualization. -->



## Citation

To cite this repository, please use:

Eduard Heijkoop. (2024). EduardHeijkoop/ICESat-2: First release (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.12691578


## Acknowledgments

I'd like to acknowledge the following organizations for their open source and/or freely available tools that help make this software work:
- OpenStreetMap and its contributors for producting the coastline.
- Sinergise for hosting the Copernicus DEM on AWS S3.
- Danish Technical University (DTU) for creating DTU21 MSS.
- CNES for creating FES2014/22.


## To Do
- [ ] Update readme for other scripts.
- [ ] Remove AR5 option from inundation.
- [ ] Add option to get GSW tiles as needed. 
- [ ] Add option to ingest csv in same projection as raster for co-registration.
- [ ] Update functionality to compute final DSM uncertainty in co-registration process when csv doesn't include sigma column. 