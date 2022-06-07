A repository to work with high resolution Digital Elevation Models  
  
Mosaic_Strips:  
  
To run: python Mosaic_Strips.py [--input_file </path/to/input_file.txt>]  
  
given a directory, find orthos in subdirectories  
find unique EPSGs, may be more than one  
for given EPSG, find strips (output of SETSM) based off of the orthos  
find extents of strips to subset GSW tiles and then:  
    merge GSW tiles  
    clip merged GSW tile to match strips extent  
    find sea only, i.e. GSW==1  
    transform that from EPSG:4326 to the unique EPSG  
    select main sea only, may be more than one  
In this EPSG:  
    load a strip, find the "real" borders (i.e. >-9999)  
    turn that into a shapefile and load this multipolygon  
    apply filters:  
        smaller than 1 km^2 -> discard the whole strip  
        within a strip, remove 1x1 pixel -> artifacts will just slow things down  
        combine with GSW and find polygons of the strip that are >95% contained by GSW -> discard those  
        "reset" strip and combine with GSW again, find intersection with GSW now, > threshold -> discard  
    combine to produce final shapefile of all strips' "real" outlines:  
        strip name and geometry (polygon or multipolygon) will be in there  
    go through each strip, find strips that are older *and* fully contained by another -> discard that strip  
    find overlap of each strip with each other strip  
        find exceptions too:  
            1. intersection of two strips is fully within GSW  
            2. intersection of two strips is covered by GSW too much  
            3. intersection of two strips is too small  
    create minimum spanning tree, weighted by delta time  
    given starting path (largest one of newest strips), find generations to link strips together  
    given this path, "co-register" next strip to the current one:  
        populate intersection with points that are sampled from the current strip  
            regular grid, "land"masked by intersection polygon  
        run point2dem on this to create a grid to which we can co-register with dem_align  
    Once all strips have been co-registered to each other, we can create N>=1 unique mosaics with demmosaic, sorting by date (newest on top)  
    Output is in the form of tiles which are merged together with gdalbuildvrt  

  
Dependencies:  
Github:  
David Shean's demcoreg, pygeotools & imview:  
https://github.com/dshean/demcoreg  
https://github.com/dshean/pygeotools  
https://github.com/dshean/imview  
NASA Ames Stereo Pipeline (ASP):  
https://github.com/NeoGeographyToolkit/StereoPipeline/releases  
  
Python packages:  
numpy, scipy, pandas, geopandas, gdal/osgeo, shapely, datetime  
subprocess, os, sys, argparse, configparser, warnings  
  
Datasets:  
Global Surface Water:  
https://global-surface-water.appspot.com/download  
Download the extent dataset  
