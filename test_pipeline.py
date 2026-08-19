# test_pipeline.py, sanity checks for the nuclei analysis pipeline.
# Run with:  python -m pytest tests -q
#
# These guard the properties most likely to break silently: image and mask
# alignment, the direction of the metrics, and the fact that watershed must
# never find fewer objects than plain connected components.

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import classical
import data_prep
import llm
import metrics


@pytest.fixture(scope="module")
def sample():
    "One validation image with its ground-truth mask."
    if not (data_prep.DATA / "val" / "images").exists():
        pytest.skip("dataset not unpacked")
    image_id = data_prep.image_ids("val")[0]
    return (image_id, data_prep.load_image("val", image_id),
            data_prep.load_mask("val", image_id))


def test_images_are_scaled_and_sized(sample):
    _, image, _ = sample
    assert image.shape == data_prep.TARGET_SIZE
    assert image.dtype == np.float32
    assert 0.0 <= image.min() and image.max() <= 1.0


def test_mask_matches_image_shape(sample):
    _, image, mask = sample
    assert mask.shape == image.shape
    assert mask.dtype == bool


def test_metadata_covers_every_image():
    md = data_prep.metadata().set_index("image_id")
    for split in data_prep.SPLITS:
        for image_id in data_prep.image_ids(split):
            assert image_id in md.index
            assert md.loc[image_id, "split"] == split


def test_dice_and_iou_are_perfect_for_identical_masks():
    mask = np.zeros((16, 16), bool)
    mask[4:10, 4:10] = True
    assert metrics.dice(mask, mask) == pytest.approx(1.0, abs=1e-4)
    assert metrics.iou(mask, mask) == pytest.approx(1.0, abs=1e-4)


def test_dice_is_zero_for_disjoint_masks():
    a, b = np.zeros((16, 16), bool), np.zeros((16, 16), bool)
    a[:4], b[12:] = True, True
    assert metrics.dice(a, b) == pytest.approx(0.0, abs=1e-4)


def test_dice_exceeds_iou_for_partial_overlap():
    "Dice is always at least IoU, which catches a swapped return value."
    a, b = np.zeros((16, 16), bool), np.zeros((16, 16), bool)
    a[4:12, 4:12], b[6:14, 6:14] = True, True
    assert metrics.dice(a, b) > metrics.iou(a, b)


def test_watershed_never_finds_fewer_objects_than_components(sample):
    "Splitting touching nuclei can only increase the count, never reduce it."
    from skimage.measure import label
    _, image, _ = sample
    binary, _ = classical.otsu_mask(image)
    assert classical.label_watershed(binary).max() >= label(binary).max()


def test_segmentation_recovers_most_nucleus_pixels(sample):
    "Otsu should be a strong pixel-level baseline on this dataset."
    _, image, truth = sample
    binary, _ = classical.otsu_mask(image)
    assert metrics.dice(binary, truth) > 0.9


def test_feature_table_has_one_row_per_object(sample):
    _, image, _ = sample
    summary, table, labels = classical.analyse(image)
    assert len(table) == summary["n_objects"]
    assert len(table) == len(np.unique(labels)) - 1
    assert (table["area"] >= classical.MIN_AREA).all()


def test_summary_is_empty_safe():
    "An image with no objects must not raise."
    blank = np.zeros((64, 64), np.float32)
    import pandas as pd
    assert classical.summarise(pd.DataFrame(), blank)["n_objects"] == 0


def test_json_extraction_survives_common_model_faults():
    "The parser must cope with fences, prose, bad commas and truncation."
    assert llm.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.extract_json('sure: {"a": 2} done') == {"a": 2}
    assert llm.extract_json('{"a": 1,234}') == {"a": 1234}
    assert llm.extract_json('{"a": 1, "b": ["x"') == {"a": 1, "b": ["x"]}


def test_paragraph_and_json_split():
    text = 'PARAGRAPH:\nThe nuclei are round.\n\nJSON:\n{"n_objects": 7}'
    paragraph, record = llm.split_paragraph_json(text)
    assert paragraph == "The nuclei are round."
    assert record == {"n_objects": 7}


def test_every_formatted_prompt_renders():
    """Prompts passed through .format() must not contain stray braces.

    A literal { in the prompt text is read as a placeholder and raises
    KeyError at run time, which is how the audit prompt broke once.
    """
    import prompts
    values = {"measurements": "n_objects: 5", "description": "text",
              "image_id": "test_000"}
    for name in ["NARRATION", "HYBRID_RECORD", "AUDIT"]:
        template = getattr(prompts, name)
        args = {k: v for k, v in values.items() if "{" + k + "}" in template}
        rendered = template.format(**args)
        assert "{measurements}" not in rendered
        assert len(rendered) > 100
