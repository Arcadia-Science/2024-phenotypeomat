import os as os
import sys as sys

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy import ndimage
from skimage import filters, io

"""This script was written to create the chlorophyll fluorescence decay
figure in our pub ‘The phenotype-o-mat: A flexible tool for collecting
visual phenotypes.’ For each of two species, C. reinhardtii and C. smithii,
it takes in single .TIFF formatted image of a petri dish, segments that image to
identify colonies, uses that segmentation mask to import intensity data at
each colony from a .AVI formatted video of the chlorophyll fluorescence decay.
It produces an unsegmented and a segmented image of Petri dish.
It then plots the mean intensity for each of the species over time.
The script expects the .AVI video and the .TIFF to be in the same folder.

Usage:
python3 colony_segment_figure_chr_fl_fig.py [PATH TO SINGLE C. REINHARDTII TIFF] \
    [PATH TO SINGLE C. SMITHII TIFF]"""


# helper functions


def segment_colonies(
    image, min_size=None, max_size=None, lw_ratio=None, img_occupancy=None, min_variance=None
):
    """Takes a reasonably clean image of a petri dish or omni-well agar plate
    and returns images of individual colonies and their positions. Uses a global Otus's threshold
    then filters based on size, aspect ratio, area of the image thatis occupied by the object
    and the variance of the thresholded image."""
    if min_size is None:
        min_size = 200
    if max_size is None:
        max_size = 10000
    if lw_ratio is None:
        lw_ratio = 1.5
    if img_occupancy is None:
        img_occupancy = 0.0
    if min_variance is None:
        min_variance = 0.5
    g_image = ndimage.gaussian_filter(image, sigma=0.55, order=0)
    thr_val = filters.threshold_otsu(g_image)
    labels = ndimage.label(g_image > thr_val)
    objs = ndimage.find_objects(labels[0])
    objs = [x for x in objs if get_size_obj_2d(x) < max_size and get_size_obj_2d(x) > min_size]
    objs = [x for x in objs if get_obj_lw_ratio(x) < lw_ratio]
    objs = [
        x for x in objs if len(get_dat_above_thr(x, image)) / get_size_obj_2d(x) > img_occupancy
    ]
    objs = [x for x in objs if sc.ndimage.variance(image[x]) / np.mean(image[x]) > min_variance]
    obj_locs = [
        (x[1].start + ((x[1].stop - x[1].start) / 2), x[0].start - ((x[0].start - x[0].stop) / 2))
        for x in objs
    ]
    raw_dat = [image[y] for y in objs]
    return raw_dat, obj_locs, objs, labels


def get_dat_above_thr(obj, image):
    """helper function returning the non-zero data in an image using a slice format object"""
    data = get_foreground(image[obj])
    data = data[np.nonzero(data)]
    return data


def get_size_obj_2d(obj):
    """helper function to calculate the area of a slice format object"""
    x_size = obj[0].stop - obj[0].start
    y_size = obj[1].stop - obj[1].start
    return x_size * y_size


def get_obj_lw_ratio(obj):
    """helper fucntion to calculate the length to width ratio of a slice format object"""
    x_size = obj[0].stop - obj[0].start
    y_size = obj[1].stop - obj[1].start
    if x_size > y_size:
        out = x_size / y_size
    elif y_size > x_size:
        out = y_size / x_size
    elif y_size == x_size:
        out = 1.0
    return out


def get_foreground(image):
    """Takes an image with a single object (e.g. one output well from 'segment_96_well_plate_wells)
    and returns a thresholded image where the signal is separated from the background. The output
    image is the same size and shape, but the background is set to 0"""
    val = filters.threshold_otsu(image)
    out = np.where(image > val, image, 0)
    return out


# import images
if len(sys.argv) > 2:
    rein_image_path = sys.argv[1]
    smithii_image_path = sys.argv[2]
else:
    print("please indicate the paths to the data")
    sys.exit()

# pull out folder paths
rein_path = os.path.dirname(rein_image_path) + "/"
smithii_path = os.path.dirname(smithii_image_path) + "/"

# read in single images of petri dishes
rein_image = io.imread(rein_image_path)
smithii_image = io.imread(smithii_image_path)

# segment colonies for reinhardtii image
rein_dat, rein_locs, rein_objs, rein_lab_image = segment_colonies(
    rein_image, max_size=1200, min_size=10
)

# segmetn colonies for smithii image
smithii_dat, smithii_locs, smithii_objs, smithii_lab_image = segment_colonies(
    smithii_image, max_size=1200, min_size=20
)

# plot image of reinhardtii petri dish
plt.imshow(rein_image)
plt.axis("off")
plt.show()

# plot image of smithii petri dish
plt.imshow(smithii_image)
plt.axis("off")
plt.show()

# plot image of reinhardtii petri dish with markers for segmented colonies
plt.imshow(rein_image)
for x in rein_locs:
    plt.plot(x[0], x[1], "ro", markersize=2)
plt.axis("off")
plt.show()

# plot image of smithii petri dish with markers for segmented colonies
plt.imshow(smithii_image)
for x in smithii_locs:
    plt.plot(x[0], x[1], "ro", markersize=2)
plt.axis("off")
plt.show()

# prepare for analyzing videos
paths = [rein_path, smithii_path]
species_labels = ["rein", "smithii"]
data_labels = [r"$\it{C. reinhardtii}$", r"$\it{C. smithii}$"]
species_obj_sets = [rein_objs, smithii_objs]

# variables to contain data from the videos
species_string = []
images = []
colony_dat = []
times = []

# for each video segment the data from each colony and plot changes in
# fluorescence over time
for n in range(len(paths)):
    species = species_labels[n]
    species_path = paths[n]
    vid_files = [x for x in os.listdir(species_path) if "video" in x and ".avi" in x]
    species_objs = species_obj_sets[n]
    species_dat = []
    times = []
    vid_files = sorted(vid_files, key=lambda x: x.split("_")[-1].split(".")[0])
    for file in vid_files[1:]:
        file_handle = cv2.VideoCapture(species_path + file)
        time = int(file.split("_")[-1].split(".")[0])
        for x in range(30):
            _, img = file_handle.read()
            img = img[:, :, 0]
            times.append(time)
            time += 0.25
            dat = [np.mean(img[x]) for x in species_objs]
            species_dat.append(dat)
    species_dat = np.array(species_dat).T
    species_dat = [x / max(x) for x in species_dat]
    times = [x - min(times) for x in times]
    mean_species_dat = np.median(species_dat, axis=0)
    mean_species_dat = [x / np.max(mean_species_dat) for x in mean_species_dat]
    means_dat = [np.mean(x) for x in np.split(np.array(mean_species_dat), 19)]
    mean_time = [np.mean(x) for x in np.split(np.array(times), 19)]

    plt.plot(
        times,
        mean_species_dat,
        marker="o",
        linestyle="",
        ms=4,
        solid_capstyle="round",
        label=data_labels[n],
    )

    plt.plot(mean_time, means_dat, marker="_")

plt.tick_params(labelsize=11)
plt.legend()
plt.xlabel("Time since illumination (s)", fontsize=15)
plt.ylabel("Average fluorescence (au)", fontsize=15)
plt.show()
