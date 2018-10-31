This neural network will predict land cover classification from Landsat-8 and Sentinel-2 data.  This supports the NASA SBIR "Open-Source Deep Learning Classification and Visualization of Multi-Temporal Multi-Source Satellite Data",

https://sbir.nasa.gov/SBIR/abstracts/18/sbir/phase1/SBIR-18-1-S5.03-4282.html

DATA REPRESENTATION:
The raw format of data used here are .tif files from the HLS dataset with one file for each spectral band (19).  Each file contains data for 26 timestamps, of which 19 are used.  These derive from four separate regions Arkansas, Ohio, California, Vermont(?).

There are also ground truth files for each of these regions (tifs) containing category information.

Reformatted versions of the satellite and ground truth tif files have been saved, which have been converted to numpy arrays.  In the first case, a file has been saved for each spectral band and time stamp, for a specific geographical subregion.  In the second case, for a given geographical subregion *all* the data from that subregion has been combined in a single HT x WD x BANDS x TIMEPTS array.

HOW TO RUN CROPNET: We assume that you have a directory containing .tif files with a file for each spectral band of interest, e.g. hls_cls_ark_01.tif, hls_cls_ark_02, etc.  We also assume that you have a *corresponding* .tif file which contains the CDL labels, e.g. cdl_2016_neAR.tif.  Let these be (e.g.)

HLS_DIR=~/Datasets/HLS/ark
CDL_PATH=~/Datasets/HLS/ground_truth/cdl_2016_neAR.tif

To make a 4D numpy array out of a particular subregion, run

DEEPRES=<path to deepres repo>
cd $DEEPRES/cropnet
OUTPUT_SUPDIR=~/Datasets/HLS/test_imgs
python make_images.py --hls-dir $HLS_DIR --cdl $CDL_PATH -o $OUTPUT_SUPDIR

This will put .npy and .png files into $OUTPUT_SUPDIR/hls, including the 4D band x time x ht x wd file, which will begin with "hls_tb". The command given above will chip out the default subregion, which is the500 pixel NW corner of the region.  To chip out a custom (square) region, say as defined by the bounding box [500, 1000, 1250, 1750], run

python make_images.py --hls-dir $HLS_DIR --cdl $CDL_PATH -o $OUTPUT_SUPDIR -x 500 -y 1000 --image-size 750


