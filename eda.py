# eda.py, Task 1: exploratory figures for the nuclei dataset.
#
# Produces a sample image grid, an intensity histogram, a comparison of the
# grayscale conversion choices, and a summary of the ground-truth metadata.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans",
                                          "DejaVu Sans"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io

import data_prep

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
MET = ROOT / "outputs" / "metrics"


def sample_grid(fname="eda_samples.png", per_density=2):
    """One row per density regime, with the ground-truth count in the title.

    Showing the regimes side by side makes the difficulty axis explicit:
    sparse nuclei are well separated, clustered ones touch.
    """
    md = data_prep.metadata()
    regimes = ["sparse", "normal", "dense", "clustered"]
    fig, axes = plt.subplots(len(regimes), per_density * 2,
                             figsize=(3 * per_density * 2, 3 * len(regimes)))
    for r, regime in enumerate(regimes):
        picks = md[(md.density == regime) & (md.split == "train")].head(per_density)
        for c, (_, row) in enumerate(picks.iterrows()):
            img = data_prep.load_image("train", row.image_id)
            mask = data_prep.load_mask("train", row.image_id)
            axes[r, c * 2].imshow(img, cmap="gray")
            axes[r, c * 2].set_title(f"{regime}, n={row.n_objects}", fontsize=9)
            axes[r, c * 2 + 1].imshow(mask, cmap="gray")
            axes[r, c * 2 + 1].set_title("ground-truth mask", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def intensity_histogram(fname="eda_intensity.png", split="train"):
    """Pooled intensity histogram, plus nucleus and background separately.

    The pooled histogram is strongly bimodal, which is why a global Otsu
    threshold works so well on this dataset.
    """
    ids = data_prep.image_ids(split)[:20]
    imgs = np.stack([data_prep.load_image(split, i) for i in ids])
    masks = np.stack([data_prep.load_mask(split, i) for i in ids])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].hist(imgs.ravel(), bins=100, color="#4C78A8")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Pooled pixel intensities ({len(ids)} {split} images)")
    axes[0].set_xlabel("intensity (blue channel, scaled to 0-1)")
    axes[0].set_ylabel("pixel count (log)")
    axes[1].hist(imgs[masks].ravel(), bins=60, alpha=0.7, density=True,
                 label="nucleus", color="#4C78A8")
    axes[1].hist(imgs[~masks].ravel(), bins=60, alpha=0.7, density=True,
                 label="background", color="#F58518")
    axes[1].set_yscale("log")
    axes[1].set_title("Intensity by ground-truth class")
    axes[1].set_xlabel("intensity")
    axes[1].set_ylabel("density (log)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def grayscale_comparison(fname="eda_grayscale.png", split="train"):
    """Blue channel against luminance grayscale.

    Standard luminance weighting gives blue only 0.114, which dims a
    DAPI-like stain. This figure justifies using the blue channel.
    """
    iid = data_prep.image_ids(split)[1]
    rgb = io.imread(data_prep.DATA / split / "images" / f"{iid}.png")
    blue = data_prep.to_grayscale(rgb, "blue")
    lum = data_prep.to_grayscale(rgb, "luminance")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, im, t in [(axes[0], rgb, "original RGB"),
                      (axes[1], blue, "blue channel"),
                      (axes[2], lum, "luminance grayscale")]:
        ax.imshow(im, cmap=None if t.endswith("RGB") else "gray",
                  vmin=None if t.endswith("RGB") else 0,
                  vmax=None if t.endswith("RGB") else 1)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    axes[1].set_xlabel(f"mean {blue.mean():.3f}")
    fig.suptitle(f"{iid}: blue channel retains the stain signal "
                 f"(mean {blue.mean():.3f}) that luminance suppresses "
                 f"(mean {lum.mean():.3f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def metadata_summary(fname="dataset_summary.csv"):
    "Ground-truth counts and areas per split and density regime."
    md = data_prep.metadata()
    out = (md.groupby(["split", "density"])
             .agg(images=("image_id", "count"),
                  n_objects_mean=("n_objects", "mean"),
                  n_objects_min=("n_objects", "min"),
                  n_objects_max=("n_objects", "max"),
                  area_fraction_mean=("area_fraction", "mean"))
             .round(3).reset_index())
    MET.mkdir(parents=True, exist_ok=True)
    out.to_csv(MET / fname, index=False)
    return out


def corrupted_comparison(fname="eda_corrupted.png"):
    """Clean test image against its blurred and low-contrast variants.

    These are the robustness cases used in the discussion: the same scene
    degraded in two different ways.
    """
    base = "test_000"
    variants = [(base, False, "clean"),
                (f"{base}_blur", True, "blurred"),
                (f"{base}_lowcontrast", True, "low contrast")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (iid, corrupt, title) in zip(axes, variants):
        img = data_prep.load_image("test", iid, corrupted=corrupt)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title}\nmean {img.mean():.3f}, sd {img.std():.3f}",
                     fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def run():
    "All Task-1 EDA outputs."
    FIG.mkdir(parents=True, exist_ok=True)
    sample_grid()
    intensity_histogram()
    grayscale_comparison()
    corrupted_comparison()
    summary = metadata_summary()
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
