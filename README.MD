# Digital Elevation Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13272772.svg)](https://doi.org/10.5281/zenodo.13272772)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen)](https://opensource.org/license/mit)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/numpy)](https://python.org)

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

General Paths & Constants
- `tmp_dir` : Path to a directory where temporary files will be written to. All temporary files will be deleted upon completion, but in case of errors it may be useful to see which files were or weren't created.
- `earthdata_username` : Enter your NASA EarthData username here.
- `osm_shp_file` : OpenStreetMap's land polygons are used to define the boundary between land and water. Click [here](https://osmdata.openstreetmap.de/download/land-polygons-complete-4326.zip) to download the zip file. Extract this and point to the `land_polygons.shp` file.
- `landmask_c_file` : Point to the `pnpoly_function.c` file that is used to run the landmask algorithm. This repository comes with a file in the `C_Code` directory, so easiest is to point to that. A corresponding `.so` file will be made, ***do not*** point to that.


## Mosaic_Strips.py

Given a set of overlapping DEM tiles (often called strips), this script will stitch them all together into a (vertically) coherent product.

### Configuration

The first step is to create an input file (with its path listed in the configuration file), which gives the location of the files to build the mosaic and the location where to output the final product:

<p float "middle">
    <a>
        <img src="https://raw.githubusercontent.com/EduardHeijkoop/EduardHeijkoop.github.io/refs/heads/main/Assets/Images/Mosaic_Input.png" alt="Mosaic Input File" />
    </a>
</p>

The following list briefly explains the functionality of each relevant parameter in the configuration file.

    GENERAL_PATHS/gsw_dir : Directory where extent tiles of the Global Surface Water [Pekel et al., 2016] are located. [This link](https://global-surface-water.appspot.com/download) has a number of ways to download the dataset.
    MOSAIC_PATHS/input_file : 
    POLYGON_AREA_THRESHOLD : 

### Optional Flags

    --config <configuration_file> : Path to configuration file. Default will be dem_config.ini in the same directory.
    --list <list_file> : Specify a list of files to be turned into a mosaic. This can be useful for testing or using a subset of a directory's files. This can be used with an empty input file, as it effectively appends the list as an extra row to the input file.
    --output_dir <output_dir> : To be used in conjunction with a list_file. Specifies the output directory of the mosaic to be created from the list file.
    --loc_name <loc_name> : To be used in conjunction with a list file. Specifies the name of the mosaic to be created from the list file.
    --dir_structure <dir_structure> : Specifies the directory structure that should be expected. The input directory can either all be in the input directory (use `simple`), in the `sealevel` structure (explained below), or when `scenes` is used all files named `*dem.tif` in the subdirectories are used.
    --gsw <gsw_path> : Point to a specific (global) surface water product to use. Note, this product will be selected for all entries in the input file, so best to use with either a single entry or with the manual input.
    --N_cpus <N_cpus> : How many CPU cores to use. Each generation of alignment may include independent steps, which can be run in parallel.
    --horizontal : This flag will turn on horizontal shifting. There may be geolocation errors in each individual strip, and incorporating relative shift in x and y directions may improve the fit of the alignment, at the cost of (significantly) increased run time.
    --cloud_water_filter : This will include results from Find_Cloudy_DEMs.py. This allows for exclusion of strips that are too cloudy or too much over open water.
    --corrected : This flag is used to select corrected strips only. If Correct_DEM_Strip_ICESat2.py was run on the strips, this flag will use the corrected strips rather than the original ones.
    --all_strips : This flag will use all strips to create a mosaic and not perform any geometric filtering.
    --no_gsw : Skip the (global) surface water filtering. Applicable when a mosaic is built (almost) entirely over land, rather than a coastal region.
    --simplify : Simplify the geometry used to populate points for strip-to-strip alignment. If the POLYGON_SIMPLIFY_VALUE in the configuration file is smaller than X_SPACING/Y_SPACING, this should not lead to accuracy loss. 



When the `sealevel` option is used for the `--dir_structure` flag, the following directory structure is required:

```bash
└── WV0N_YYYYMMDD_*/
    └── strips/
        ├── WV0N_YYYYMMDD_*dem.tif
        └── WV0N_YYYYMMDD_*dem_smooth.tif
```

When `simple` is used for this flag, only files in the input directory will be used. When `scenes` is used, the subdirectories will be recursively searched for files that end with `dem.tif`. 

The strips used to build the mosaic must satisfy a few conditions in terms of filenames. They must start with `WV0N` (where `N` is 1, 2 or 3, denoting the WorldView satellite used to acquire the images). Alternatively, they can start with `GE01` when GeoEye-1 was used for the acquisition. Next, the date of acquisition must be attached to the satellite idenitifier with one underscore and in the YYYYMMDD format. A filename that starts with `WV02_20210507` then represents a strip acquired by WorldView-2 on May 7th. The timestamp is required to apply the appropriate weights to the minimum spanning tree when building the mosaic.


### Running

To run: python Mosaic_Strips.py

<p float "middle">
    <a>
        <img src="https://raw.githubusercontent.com/EduardHeijkoop/EduardHeijkoop.github.io/refs/heads/main/Assets/Images/Mosaic_Result.png" alt="Mosaic Results" />
    </a>
</p>

## Simple_Coregistration.py

This script iteratively co-registers a raster to a csv. The most relevant use here is to co-register a previously built mosaic to ICESat-2 data. 

Usage:

    python Simple_Coregistration.py --raster </path/to/raster> --csv </path/to/csv> 

The output file will be a co-registered product in the same directory as the input raster file, with the amount shifted appended to its name. For example, a mosaic `Sudan_PortSudan_Full_Mosaic_0_32637.tif` to `Sudan_PortSudan_Full_Mosaic_0_32637_Shifted_z_2p07m.tif`.

<p float "middle">
    <a>
        <img src="https://raw.githubusercontent.com/EduardHeijkoop/EduardHeijkoop.github.io/refs/heads/main/Assets/Images/Coregistration_Result.png" alt="Coregistration Results" />
    </a>
</p>

### Optional Flags

    --median : Flag to select zero median rather than zero mean as target for the co-registration. Default is false.
    --sigma <value> : How many standard deviations to use to detect outliers. Value must be an integer. Default is 2. 
    --threshold <value> : Iterative threshold to detect when convergence has been achieved. Default value is 0.05 m.
    --resample : Flag to sample the newly co-registered raster with the original input csv; this is in addition to the sampling  Default is false.
    --keep_original_sample
    --no_writing
    --nodata <nodata value> : 
    --print
    --write_file <file to write statistics to>
    --output_dir <output directory>
    --N_iterations <number of iterations>

## Compute_Inundation.py

Once a DSM has been aligned in the vertical, it can be used to compute inundation projections.

Usage

    python Compute_Inundation.py



## Correct_DEM_Strip_ICESat2.py

Optional pre-processing step before creating a mosaic.

## Find_Cloudy_DEMs.py 


## Global_DEMs.py




## Interpolate_VLM.py


## Plot_Angles.py

Given a set of .ntf and their associated xml files, will plot spatial extent of the image and the satellite's look angle. This can be a visual aid to get the best set of files to build a DSM.

Usage:

    python Plot_Angles.py --input_file <input file> [--coast <coastline file>]

The input file will contain a list of full paths to the different NTF files to be plotted, without a header line. This requires each .ntf/.NTF file to have an associated .xml file with the same file name in the same directory. Adding a coastline is optional, but may aid in visualization.



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
- [ ] Add option to get GSW tiles as needed. 
- [ ] Update functionality to compute final DSM uncertainty in co-registration process when csv doesn't include sigma column. 