import os as os
import sys as sys

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.ndimage as ndimage
import seaborn as sbn
from skimage import filters
from skimage.morphology import disk

"""This script was written to create the colony multi-wavelength reflectance images
in our pub ‘The phenotype-o-mat: A flexible tool for collecting visual phenotypes.’
For each of two species, C. reinhardtii and C. smithii, it takes in a series of
images with wavelength reflectance data. It requires a transillumination image of
the same plates. These images are used for colony segmentation based on intensity.
These segmented colonies are then filtered based on size, aspect ratio, and occupied
area. These segmentation masks are then used to capture reflectance intensity data
from the rest of the wavelengths. The script then plots images of the transillumintion
image, the segmentation mask, several examples of the segmented reflected light, and
then a violin plot comparison of the reflected intensity for each of the two species
for each of the reflected wavelengths. The species you would like to examine are
defined by the ‘strains’ variable and the reflected wavelength data are defined by
the ‘Incident_wavelength’ variable.  For each of those variables, the individual
values should be present in the names of the .TIFF images.

Usage:
python3 parent_strains_reflectance_fig.py [PATH TO FOLDER CONTAINING IMAGES]"""


# identify a specific input folder
if len(sys.argv) > 1:
    dat_fold = sys.argv[1]
else:
    print("please provide a data folder")
    sys.exit()

# variables to define the strains and wavelengths to analyze
strains = ["cc124", "cc1373"]

incident_wavelengths = ["460", "535", "590", "670"]


# helper functions
def get_aspectratio(obj):
    """helper function to grab the aspect ratio of a 2d object in slice format"""
    x_size = obj[0].stop - obj[0].start
    y_size = obj[1].stop - obj[1].start
    if x_size > y_size:
        aspr = x_size / y_size
    elif y_size > x_size:
        aspr = y_size / x_size
    elif y_size == x_size:
        aspr = 1
    return aspr


def get_size_obj_2d(obj):
    """helper function to grab the size of a 2d object in slice format"""
    x_size = obj[0].stop - obj[0].start
    y_size = obj[1].stop - obj[1].start
    return x_size * y_size


def crop_well(image, width=None):
    """helper function to crop out the central region of a small image.
    The intended target is an image of a well"""
    if width is None:
        width = 10
    r = int(width / 2)
    cent = ndimage.center_of_mass(image)
    cent = [int(x) for x in np.around(cent)]
    crop = image[cent[0] - r : cent[0] + r, cent[1] - r : cent[1] + r]
    return crop


def get_occupied_area(obj, img):
    """helper function to determine how much of an object crop is occupied by the object"""
    size = get_size_obj_2d(obj)
    occupied_area = np.count_nonzero(get_background_image(img[obj]))
    return occupied_area / size


def get_foreground_image(image):
    """Takes an image with a single object (e.g. one output well from 'segment_96_well_plate_wells)
    and returns a thresholded image where the signal is separated from the background. The image is
    the same size, but the background is set to 0"""
    val = filters.threshold_otsu(image)
    out = np.where(image > (val), image, 0)
    return out


def get_background_image(image):
    """Helper function to return the background of an image"""
    val = filters.threshold_otsu(image)
    out = np.where(image < (val), image, 0)
    return out


def get_background_data(image):
    """helper function to return the data from the background of an image"""
    val = filters.threshold_otsu(image)
    out = np.where(image < (val), image, 0)
    out = np.nonzero(out)
    return out


# File handling
fils = os.listdir(dat_fold)

fils = [x for x in fils if any([y for y in strains if y in x])]

trans_illumination_files = [x for x in fils if "TRANS" in x]

plate_names = [x.split("_")[1] for x in fils]

fil_handle = cv2.VideoCapture(dat_fold + trans_illumination_files[0])
_, img = fil_handle.read()

img = img[:, :, 0]

# make masks using the transillumination images of the plates
mask_dict = {}
for fil in trans_illumination_files:
    plate_number = fil.replace("-", "_").split("_")[2]
    strain_name = [x for x in strains if x in plate_number][0]
    fil_handle = cv2.VideoCapture(dat_fold + fil)
    _, img = fil_handle.read()
    img = img[:, :, 0]  # make single channel mono from 3 channel mono
    if strain_name == "cc124":
        cc_124_raw = img
    img = ndimage.gaussian_filter(img, sigma=(8, 8), order=0)
    thr = filters.rank.otsu(img, disk(13)) / 1.2
    labs = ndimage.label(img < thr)
    labss = list(range(labs[1]))
    objs = ndimage.find_objects(labs[0])
    labss = [
        labss[n]
        for n in range(len(objs))
        if get_size_obj_2d(objs[n]) > 20 and get_size_obj_2d(objs[n]) < 5000
    ]
    objs = [x for x in objs if get_size_obj_2d(x) > 20 and get_size_obj_2d(x) < 5000]

    labss = [labss[n] for n in range(len(objs)) if get_aspectratio(objs[n]) < 1.5]
    objs = [x for x in objs if get_aspectratio(x) < 1.5]

    labss = [labss[n] for n in range(len(objs)) if get_occupied_area(objs[n], img) > 0.4]
    objs = [x for x in objs if get_occupied_area(x, img) > 0.4]

    if strain_name == "cc124":
        lab_img = labs[0]
        lab_img = np.where(np.isin(lab_img, labss), lab_img + 20, 0)

    mask_dict[strain_name] = objs

# set up variables to store the data
wl_labels = []
strain_labels = []
reflectance_data = []
ref_dat_by_colony = []
concat_wl_images = []

# collect data from each wavelength and strain using the strain masks defined above
for wavelength in incident_wavelengths:
    for strain in strains:
        for file in [x for x in fils if strain in x and wavelength in x]:
            fil_handle = cv2.VideoCapture(dat_fold + file)
            _, img = fil_handle.read()
            img = img[:, :, 0]  # make single channel mono from 3 channel mono
            data = [np.mean(get_background_data(img[x])) for x in mask_dict[strain]]
            img_data = [get_background_image(img[x]) for x in mask_dict[strain]]
            concat_wl_images.append(img_data)
            reflectance_data.append(data)
            wl_labels.append(wavelength)
            strain_labels.append(strain)

# reorganize the data for plotting
data_pairs = []
for n in range(0, 8, 2):
    print(sc.stats.ttest_ind([x for x in reflectance_data[n] if x > 15], reflectance_data[n + 1]))
    data_pairs.append([x for x in reflectance_data[n] if x > 15])
    data_pairs.append(reflectance_data[n + 1])

# plot segmented colonies
fig, ax = plt.subplots(12, 4, figsize=(1.5, 8), sharex=True, sharey=True)
for n in range(4):
    for m in range(12):
        ax[m, n].imshow(concat_wl_images[2 * n][m], vmin=0, vmax=55)
        ax[m, n].axis("off")
ax[1, 1].set_ylabel(r"$\it{Chlamydomonas reinhardtii$}", fontsize=15)
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

fig, ax = plt.subplots(12, 4, figsize=(1.5, 8), sharex=True, sharey=True)
for n in range(4):
    for m in range(12):
        ax[m, n].imshow(concat_wl_images[(2 * n) + 1][m], vmin=0, vmax=55)
        ax[m, n].axis("off")
ax[1, 1].set_ylabel(r"$\it{Chlamydomonas smithii$}", fontsize=15)
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

# violin plot of intensity values
v_plot = sbn.violinplot(data_pairs, split=True)
v_plot.set_ylabel("Mean pixel intensity (au)", fontsize=15)
v_plot.tick_params(labelsize=11)
v_plot.set_xticks([0.5, 2.5, 4.5, 6.5])
v_plot.legend(strain_labels)
v_plot.set_xticklabels(wl_labels[::2], fontsize=11)
v_plot.set_xlabel("Incedent light wavelength (nm)", fontsize=15)
v_fig = v_plot.get_figure()
v_fig.savefig("/home/dmets/Desktop/v_fig.pdf")
plt.show()

# plots of transillumination and segmentation mask
plt.imshow(cc_124_raw)
plt.axis("off")
plt.show()

plt.imshow(lab_img)
plt.axis("off")
plt.show()
