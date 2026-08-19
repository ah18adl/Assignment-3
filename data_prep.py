# data_prep.py, Task 1: load the nuclei dataset, convert to grayscale,
# resize to a common 256x256, and provide split accessors.
#
# The images are DAPI-like blue-stained nuclei on a dark field, so almost all
# the signal sits in the blue channel. Converting to grayscale with the
# standard luminance weights would attenuate it (blue weight is only 0.114),
# so the loader offers both: a luminance grayscale for display and a blue
# channel option for measurement. See to_grayscale for the comparison.

from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nuclei_dataset"
TARGET_SIZE = (256, 256)
SPLITS = ("train", "val", "test")


def metadata():
    "Ground truth table: image_id, split, density, n_objects, intensity, area."
    return pd.read_csv(DATA / "metadata.csv")


def image_ids(split):
    "Sorted image ids for one split, taken from the images directory."
    files = sorted((DATA / split / "images").glob("*.png"))
    return [f.stem for f in files]


def to_grayscale(rgb, method="blue"):
    """Convert an RGB image to a single channel in [0, 1].

    method='blue' keeps the blue channel, which carries the DAPI signal.
    method='luminance' uses the standard 0.299/0.587/0.114 weighting, which
    is the conventional choice but suppresses blue by design.
    """
    if method == "blue":
        return rgb[..., 2].astype(np.float32) / 255.0
    return rgb2gray(rgb).astype(np.float32)


def load_image(split, image_id, gray="blue", corrupted=False):
    "Load one image, convert to grayscale and resize to TARGET_SIZE."
    folder = DATA / ("test_corrupted" if corrupted else split) / "images"
    rgb = io.imread(folder / f"{image_id}.png")
    g = to_grayscale(rgb, gray)
    if g.shape != TARGET_SIZE:
        g = resize(g, TARGET_SIZE, anti_aliasing=True, preserve_range=True)
    return g.astype(np.float32)


def load_mask(split, image_id):
    "Load the binary ground-truth mask as a boolean array at TARGET_SIZE."
    m = io.imread(DATA / split / "masks" / f"{image_id}.png")
    if m.shape != TARGET_SIZE:
        m = resize(m, TARGET_SIZE, order=0, preserve_range=True)
    return m > 127


def load_labels(split, image_id):
    "Load the 16-bit instance label map (each nucleus a distinct integer)."
    return io.imread(DATA / split / "labels" / f"{image_id}.png")


def load_split(split, gray="blue"):
    """Load a whole split.

    Returns (ids, images as float32 [n,256,256], masks as bool [n,256,256]).
    """
    ids = image_ids(split)
    images = np.stack([load_image(split, i, gray) for i in ids])
    masks = np.stack([load_mask(split, i) for i in ids])
    return ids, images, masks


def corrupted_ids():
    "Ids of the deliberately degraded test images (blur, low contrast)."
    return sorted(f.stem for f in
                  (DATA / "test_corrupted" / "images").glob("*.png"))


if __name__ == "__main__":
    md = metadata()
    print("metadata rows:", len(md))
    print(md.groupby(["split", "density"]).size().unstack(fill_value=0))
    for s in SPLITS:
        ids, imgs, masks = load_split(s)
        print(f"{s}: {len(ids)} images {imgs.shape} "
              f"range [{imgs.min():.2f}, {imgs.max():.2f}], "
              f"mask coverage {masks.mean():.3f}")
    print("corrupted:", corrupted_ids())
