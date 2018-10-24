This neural network will predict land cover classification from Landsat-8 and Sentinel-2 data.  This supports the NASA SBIR "Open-Source Deep Learning Classification and Visualization of Multi-Temporal Multi-Source Satellite Data",

https://sbir.nasa.gov/SBIR/abstracts/18/sbir/phase1/SBIR-18-1-S5.03-4282.html

DATA REPRESENTATION:
The raw format of data used here are .tif files from the HLS dataset with one file for each spectral band (19).  Each file contains data for 26 timestamps, of which 19 are used.  These derive from four separate regions Arkansas, Ohio, California, Vermont(?).

There are also ground truth files for each of these regions (tifs) containing category information.

Reformatted versions of the satellite and ground truth tif files have been saved, which have been converted to numpy arrays.  In the first case, a file has been saved for each spectral band and time stamp, for a specific geographical subregion.  In the second case, for a given geographical subregion *all* the data from that subregion has been combined in a single HT x WD x BANDS x TIMEPTS array.



