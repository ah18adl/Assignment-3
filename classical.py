# classical.py, Task 2, Otsu thresholding, morphological cleanup, connected
# component labelling and a per-object feature table from regionprops_table.
#
# Two labelling modes are provided. Plain connected components treat any
# group of touching nuclei as a single object, which is wrong for the
# clustered and dense regimes in this dataset. Watershed on the distance
# transform splits those groups, so it is the default. Both modes are kept
# so the two can be compared, because this is the main reason simple
# pipelines undercount.

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import (binary_closing, binary_opening, disk,
                                remove_small_holes, remove_small_objects)
from skimage.segmentation import clear_border, watershed

MIN_AREA = 30          # nuclei smaller than this are noise, not cells
MIN_HOLE = 30          # fill pinholes inside a nucleus
# Minimum separation between watershed seeds. Tuned on the training split
# only (grid over distance 1 to 8 and area 15 to 50); distance 2 with area 30
# gave the lowest count error. Larger values merge touching nuclei and
# undercount the clustered and dense regimes.
MIN_DISTANCE = 2
FEATURES = ["label", "area", "eccentricity", "solidity", "extent",
            "perimeter", "equivalent_diameter", "mean_intensity",
            "max_intensity", "min_intensity", "major_axis_length",
            "minor_axis_length", "centroid-0", "centroid-1"]


def otsu_mask(image, min_area=MIN_AREA, drop_border=False):
    """Otsu threshold plus morphological cleanup.

    Opening removes single-pixel speckle, closing seals small gaps in a
    nucleus edge, then tiny objects and pinholes are removed. Border objects
    are kept by default because partial nuclei still count in this dataset.
    """
    thresh = threshold_otsu(image)
    binary = image > thresh
    binary = binary_opening(binary, disk(1))
    binary = binary_closing(binary, disk(2))
    binary = remove_small_holes(binary, area_threshold=MIN_HOLE)
    binary = remove_small_objects(binary, min_size=min_area)
    if drop_border:
        binary = clear_border(binary)
    return binary, thresh


def label_watershed(binary, min_distance=MIN_DISTANCE):
    """Split touching nuclei with a distance-transform watershed.

    The distance transform peaks at nucleus centres; those peaks seed the
    watershed, so a cluster of touching nuclei is divided along the valleys
    between centres rather than being labelled as one object.
    """
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=min_distance,
                            labels=binary, exclude_border=False)
    markers = np.zeros(distance.shape, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    if markers.max() == 0:                   # no peaks found, fall back
        return label(binary)
    return watershed(-distance, markers, mask=binary)


def segment(image, use_watershed=True, min_area=MIN_AREA):
    "Full classical segmentation: returns (label_map, binary, threshold)."
    binary, thresh = otsu_mask(image, min_area=min_area)
    labels = label_watershed(binary) if use_watershed else label(binary)
    labels = remove_small_objects(labels, min_size=min_area)
    return labels, binary, thresh


def feature_table(labels, image):
    """Per-object features via regionprops_table.

    Returns a dataframe with one row per detected nucleus. Intensity features
    are measured on the supplied grayscale image, so they are comparable
    across images only because every image is scaled to [0, 1].
    """
    if labels.max() == 0:
        return pd.DataFrame(columns=FEATURES)
    props = regionprops_table(labels, intensity_image=image,
                              properties=["label", "area", "eccentricity",
                                          "solidity", "extent", "perimeter",
                                          "equivalent_diameter",
                                          "mean_intensity", "max_intensity",
                                          "min_intensity",
                                          "major_axis_length",
                                          "minor_axis_length", "centroid"])
    df = pd.DataFrame(props)
    # circularity is a useful shape summary that regionprops does not provide
    df["circularity"] = np.where(
        df["perimeter"] > 0,
        4 * np.pi * df["area"] / df["perimeter"] ** 2, np.nan)
    return df


def summarise(df, image=None):
    """Collapse a per-object table into the image-level numbers an LLM sees.

    Deliberately small: counts, central tendency and spread of size and
    shape, plus crowding. These are the only inputs to the narration step.
    """
    if len(df) == 0:
        return {"n_objects": 0, "mean_area": 0.0, "median_area": 0.0,
                "std_area": 0.0, "min_area": 0.0, "max_area": 0.0,
                "mean_eccentricity": 0.0, "mean_solidity": 0.0,
                "mean_circularity": 0.0, "mean_intensity": 0.0,
                "area_fraction": 0.0}
    total_px = image.size if image is not None else 256 * 256
    return {
        "n_objects": int(len(df)),
        "mean_area": round(float(df["area"].mean()), 1),
        "median_area": round(float(df["area"].median()), 1),
        "std_area": round(float(df["area"].std(ddof=0)), 1),
        "min_area": round(float(df["area"].min()), 1),
        "max_area": round(float(df["area"].max()), 1),
        "mean_eccentricity": round(float(df["eccentricity"].mean()), 3),
        "mean_solidity": round(float(df["solidity"].mean()), 3),
        "mean_circularity": round(float(df["circularity"].mean()), 3),
        "mean_intensity": round(float(df["mean_intensity"].mean()), 3),
        "area_fraction": round(float(df["area"].sum() / total_px), 4),
    }


def analyse(image, use_watershed=True):
    "Segment one image and return (summary dict, per-object table, labels)."
    labels, binary, _ = segment(image, use_watershed=use_watershed)
    df = feature_table(labels, image)
    return summarise(df, image), df, labels
