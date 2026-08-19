# metrics.py, segmentation and counting metrics used in Tasks 2, 3 and 4.
#
# Dice and IoU are pixel-level and answer "is this pixel nucleus or
# background". Count error is object-level and answers "how many nuclei are
# there". A pipeline can score well on the first and badly on the second,
# which is exactly what happens on clustered images, so both are computed.

import numpy as np
import pandas as pd


def dice(pred, true, eps=1e-7):
    "Dice coefficient, 2|A and B| / (|A| + |B|), for boolean masks."
    pred, true = np.asarray(pred, bool), np.asarray(true, bool)
    inter = np.logical_and(pred, true).sum()
    return float((2 * inter + eps) / (pred.sum() + true.sum() + eps))


def iou(pred, true, eps=1e-7):
    "Intersection over union (Jaccard index) for boolean masks."
    pred, true = np.asarray(pred, bool), np.asarray(true, bool)
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return float((inter + eps) / (union + eps))


def pixel_scores(pred, true):
    "Dice, IoU, precision and recall for one mask pair."
    pred, true = np.asarray(pred, bool), np.asarray(true, bool)
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    return {"dice": dice(pred, true), "iou": iou(pred, true),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0}


def count_scores(pred_n, true_n):
    "Absolute and relative counting error for one image."
    return {"n_pred": int(pred_n), "n_true": int(true_n),
            "count_error": int(pred_n - true_n),
            "abs_count_error": int(abs(pred_n - true_n)),
            "pct_count_error": float(abs(pred_n - true_n) / true_n * 100)
            if true_n else np.nan}


def summarise_scores(rows, by=None):
    """Aggregate per-image score dicts into a mean table.

    by: optional column name (usually 'density') to group by, so that
    performance can be broken down per difficulty regime rather than
    left as a single average.
    """
    df = pd.DataFrame(rows)
    numeric = df.select_dtypes("number")
    if by is None:
        return numeric.mean().round(4)
    return df.groupby(by)[numeric.columns].mean().round(4)
