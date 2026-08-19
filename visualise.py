# visualise.py, figures for Tasks 2, 3 and 4.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans",
                                          "DejaVu Sans"]
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries

import classical
import data_prep
import metrics

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"


def _show(ax, img, title, cmap="gray"):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def segmentation_steps(split="val", image_id=None,
                       fname="classical_steps.png"):
    """The classical pipeline stage by stage on one image.

    Shows why each morphological step is there: the raw threshold is speckled,
    cleanup removes the speckle, and watershed splits the touching nuclei that
    connected components would merge.
    """
    md = data_prep.metadata().set_index("image_id")
    if image_id is None:                       # pick a clustered image
        ids = [i for i in data_prep.image_ids(split)
               if md.loc[i, "density"] == "clustered"]
        image_id = ids[0] if ids else data_prep.image_ids(split)[0]
    img = data_prep.load_image(split, image_id)
    truth = data_prep.load_mask(split, image_id)

    from skimage.filters import threshold_otsu
    from skimage.measure import label
    raw = img > threshold_otsu(img)
    clean, _ = classical.otsu_mask(img)
    cc = label(clean)
    ws = classical.label_watershed(clean)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    _show(axes[0], img, f"{image_id}\ninput (blue channel)")
    _show(axes[1], raw, f"Otsu threshold\n({raw.sum()} px)")
    _show(axes[2], clean, f"after cleanup\n({clean.sum()} px)")
    _show(axes[3], label2rgb(cc, bg_label=0),
          f"connected components\nn = {cc.max()}", cmap=None)
    _show(axes[4], label2rgb(ws, bg_label=0),
          f"watershed split\nn = {ws.max()} (truth {md.loc[image_id, 'n_objects']})",
          cmap=None)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def prediction_grid(split, image_ids, pred_masks, fname, title,
                    n_show=3):
    """Input, ground truth and prediction side by side, with Dice per image.

    Required by Task 3 for at least three validation images; also used for
    the test set in Task 4.
    """
    n_show = min(n_show, len(image_ids))
    fig, axes = plt.subplots(n_show, 4, figsize=(13, 3.2 * n_show))
    if n_show == 1:
        axes = axes[None, :]
    for r in range(n_show):
        iid = image_ids[r]
        img = data_prep.load_image(split, iid)
        truth = data_prep.load_mask(split, iid)
        pred = np.asarray(pred_masks[r], bool)
        d = metrics.dice(pred, truth)
        overlay = np.dstack([pred.astype(float), truth.astype(float),
                             np.zeros_like(img)])
        _show(axes[r, 0], img, f"{iid}: input")
        _show(axes[r, 1], truth, "ground truth")
        _show(axes[r, 2], pred, f"prediction (Dice {d:.3f})")
        _show(axes[r, 3], overlay,
              "overlay: red = pred, green = truth,\nyellow = agreement",
              cmap=None)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def training_curves(history, fname="unet_training.png"):
    "Loss and validation Dice against epoch."
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("Dice + BCE loss")
    axes[0].set_title("Training and validation loss"); axes[0].legend(fontsize=8)
    axes[1].plot(history["epoch"], history["val_dice"], color="#54A24B")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("mean Dice")
    axes[1].set_title("Validation Dice")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def count_scatter(records, fname="count_accuracy.png"):
    """Predicted against true nucleus count, coloured by density regime.

    The diagonal is perfect counting; points below it are undercounting,
    which is what happens when touching nuclei are merged.
    """
    fig, ax = plt.subplots(figsize=(5.2, 5))
    colours = {"sparse": "#4C78A8", "normal": "#54A24B",
               "dense": "#F58518", "clustered": "#B279A2"}
    for regime, group in records.groupby("density_truth"):
        ax.scatter(group["true_n_objects"], group["n_objects"], s=45,
                   label=regime, color=colours.get(regime), alpha=0.85)
    top = max(records["true_n_objects"].max(), records["n_objects"].max()) + 5
    ax.plot([0, top], [0, top], "k--", lw=0.8, label="perfect count")
    ax.set_xlabel("true nucleus count")
    ax.set_ylabel("detected nucleus count")
    ax.set_title("Counting accuracy by density regime")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def feature_distributions(objects, fname="feature_distributions.png"):
    "Per-object area, eccentricity, solidity and circularity."
    cols = ["area", "eccentricity", "solidity", "circularity"]
    fig, axes = plt.subplots(1, len(cols), figsize=(13, 3))
    for ax, col in zip(axes, cols):
        ax.hist(objects[col].dropna(), bins=30, color="#4C78A8")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("nuclei")
    fig.suptitle("Per-object feature distributions across the test set",
                 fontsize=11)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def compact_overview(fname="eda_compact.png"):
    """One image per density regime with its mask, plus the intensity
    histogram, combined into a single compact figure.
    """
    md = data_prep.metadata()
    regimes = ["sparse", "normal", "dense", "clustered"]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.35]})
    for c, regime in enumerate(regimes):
        row = md[(md.density == regime) & (md.split == "train")].iloc[0]
        img = data_prep.load_image("train", row.image_id)
        mask = data_prep.load_mask("train", row.image_id)
        _show(axes[0, c], img, f"{regime}, n = {row.n_objects}")
        _show(axes[1, c], mask, "ground-truth mask")

    ids = data_prep.image_ids("train")[:20]
    imgs = np.stack([data_prep.load_image("train", i) for i in ids])
    masks = np.stack([data_prep.load_mask("train", i) for i in ids])
    axes[0, 4].hist(imgs.ravel(), bins=80, color="#4C78A8")
    axes[0, 4].set_yscale("log")
    axes[0, 4].set_title("pooled intensities", fontsize=9)
    axes[0, 4].set_xlabel("intensity"); axes[0, 4].set_ylabel("pixels (log)")
    axes[1, 4].hist(imgs[masks].ravel(), bins=50, alpha=0.7, density=True,
                    label="nucleus", color="#4C78A8")
    axes[1, 4].hist(imgs[~masks].ravel(), bins=50, alpha=0.7, density=True,
                    label="background", color="#F58518")
    axes[1, 4].set_yscale("log")
    axes[1, 4].set_title("by ground-truth class", fontsize=9)
    axes[1, 4].set_xlabel("intensity"); axes[1, 4].set_ylabel("density (log)")
    axes[1, 4].legend(fontsize=7)
    for ax in list(axes[0, :4]) + list(axes[1, :4]):
        ax.axis("off")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)
