# pipeline.py, Task 4, the hybrid pipeline and the shared narration step.
#
# For each image the flow is: a mask, either from the U-Net or from Otsu,
# then instance labels by watershed, then a regionprops feature table, then
# a plain text block of measurements, then an LLM paragraph and JSON record,
# then an optional audit by a second model.
#
# The LLM never sees the image. Everything it says comes from numbers a
# classical algorithm measured, which is what makes the output auditable:
# any claim can be checked against the feature table.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import remove_small_objects

import classical
import data_prep
import llm
import prompts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TABLES = OUT / "tables"


def measurement_block(summary):
    """Format the image-level summary as the plain text the model receives.

    Written as 'name: value' lines rather than raw JSON because the smaller
    text models follow line-oriented input more reliably.
    """
    order = ["n_objects", "mean_area", "median_area", "std_area", "min_area",
             "max_area", "mean_eccentricity", "mean_solidity",
             "mean_circularity", "mean_intensity", "area_fraction"]
    units = {"mean_area": " pixels", "median_area": " pixels",
             "std_area": " pixels", "min_area": " pixels",
             "max_area": " pixels"}
    lines = []
    for k in order:
        if k in summary:
            lines.append(f"{k}: {summary[k]}{units.get(k, '')}")
    return "\n".join(lines)


def narrate(summary, model=None, template=prompts.NARRATION, image_id=None,
            temperature=0.0):
    """Send measurements to the text model and split the response.

    Returns (paragraph, record dict, raw response).
    """
    model = model or llm.TEXT_MODEL
    block = measurement_block(summary)
    prompt = (template.format(image_id=image_id, measurements=block)
              if image_id is not None else template.format(measurements=block))
    raw = llm.generate(prompt, model=model, temperature=temperature)
    paragraph, record = llm.split_paragraph_json(raw)
    return paragraph, record, raw


def audit(summary, description, model=None):
    "Second model checks the narration against the same measurements."
    model = model or llm.AUDIT_MODEL
    prompt = prompts.AUDIT.format(measurements=measurement_block(summary),
                                  description=description)
    raw = llm.generate(prompt, model=model, temperature=0.0)
    scores = llm.extract_json(raw) or {}
    return {k: scores.get(k) for k in
            ("faithfulness", "completeness", "restraint",
             "unsupported_claims", "comment")}


def labels_from_mask(mask, min_area=classical.MIN_AREA):
    """Turn a binary mask into instance labels with watershed.

    Used for both the U-Net output and the Otsu output, so the two are
    measured identically and any difference in the feature tables comes from
    the segmentation itself rather than from the labelling.
    """
    labels = classical.label_watershed(mask.astype(bool))
    return remove_small_objects(labels, min_size=min_area)


def analyse_mask(image, mask):
    "Measurements and per-object table for a given binary mask."
    labels = labels_from_mask(mask)
    table = classical.feature_table(labels, image)
    return classical.summarise(table, image), table, labels


def run_test_set(masks=None, split="test", narrate_records=True,
                 do_audit=True, save=True):
    """Full Task-4 run over a split.

    masks: predicted boolean masks aligned with data_prep.image_ids(split).
    If None, the classical Otsu mask is used, so the pipeline still runs
    end to end when torch is not available.
    Returns (records dataframe, per-object table, narratives dict).
    """
    ids = data_prep.image_ids(split)
    md = data_prep.metadata().set_index("image_id")
    records, all_objects, narratives = [], [], {}

    for i, image_id in enumerate(ids):
        image = data_prep.load_image(split, image_id)
        mask = (masks[i] if masks is not None
                else classical.otsu_mask(image)[0])
        summary, table, _ = analyse_mask(image, mask)

        row = {"image_id": image_id, **summary,
               "true_n_objects": int(md.loc[image_id, "n_objects"]),
               "density_truth": md.loc[image_id, "density"]}

        if narrate_records:
            paragraph, record, _ = narrate(summary,
                                           template=prompts.HYBRID_RECORD,
                                           image_id=image_id)
            narratives[image_id] = paragraph
            if record:
                row.update({f"llm_{k}": v for k, v in record.items()
                            if k != "image_id"})
            if do_audit:
                row.update({f"audit_{k}": v
                            for k, v in audit(summary, paragraph).items()
                            if k != "unsupported_claims"})
        records.append(row)
        table.insert(0, "image_id", image_id)
        all_objects.append(table)
        print(f"  {image_id}: {summary['n_objects']} objects "
              f"(truth {row['true_n_objects']})", flush=True)

    records_df = pd.DataFrame(records)
    objects_df = pd.concat(all_objects, ignore_index=True)
    records_df["count_error"] = (records_df["n_objects"]
                                 - records_df["true_n_objects"])

    if save:
        TABLES.mkdir(parents=True, exist_ok=True)
        records_df.to_csv(TABLES / f"{split}_records.csv", index=False)
        objects_df.to_csv(TABLES / f"{split}_objects.csv", index=False)
        if narratives:
            (TABLES / f"{split}_narratives.json").write_text(
                json.dumps(narratives, indent=1))
        print(f"saved {TABLES / f'{split}_records.csv'}")
    return records_df, objects_df, narratives
