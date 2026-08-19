# run_pipeline.py, end-to-end run of the deterministic parts of the project.
#
#   python run_pipeline.py             classical route, no LLM, no torch
#   python run_pipeline.py --llm       add the narration and audit steps
#   python run_pipeline.py --unet      train the U-Net and use its masks
#
# The LLM and U-Net stages need Ollama and torch respectively, which is why
# they are opt-in: the classical route alone reproduces every segmentation
# figure and metric in the analysis. notebooks/analysis.ipynb runs the full
# set of four tasks on Colab.

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import classical
import data_prep
import eda
import metrics
import pipeline
import visualise

TABLES = ROOT / "outputs" / "tables"


def classical_scores(splits=("val", "test")):
    "Pixel and count metrics for the Otsu route on held-out images."
    md = data_prep.metadata().set_index("image_id")
    rows = []
    for split in splits:
        for image_id in data_prep.image_ids(split):
            image = data_prep.load_image(split, image_id)
            truth = data_prep.load_mask(split, image_id)
            binary, _ = classical.otsu_mask(image)
            summary, _, _ = classical.analyse(image)
            rows.append({"image_id": image_id, "split": split,
                         "density": md.loc[image_id, "density"],
                         **metrics.pixel_scores(binary, truth),
                         **metrics.count_scores(summary["n_objects"],
                                                md.loc[image_id, "n_objects"])})
    return pd.DataFrame(rows)


def main(use_llm=False, use_unet=False, epochs=30):
    # Task 1: preparation and exploratory analysis
    print("Task 1: data preparation and EDA")
    eda.run()

    # Task 2: classical segmentation and features
    print("\nTask 2: classical segmentation")
    scores = classical_scores()
    TABLES.mkdir(parents=True, exist_ok=True)
    scores.to_csv(TABLES / "classical_scores.csv", index=False)
    numeric = scores.select_dtypes("number")
    print(numeric.mean().round(4).to_string())
    print("\nby density regime")
    print(scores.groupby("density")[["dice", "iou", "n_true", "n_pred",
                                     "abs_count_error"]].mean().round(3)
          .to_string())
    visualise.segmentation_steps()
    demo_ids = data_prep.image_ids("val")[:3]
    otsu_masks = [classical.otsu_mask(data_prep.load_image("val", i))[0]
                  for i in demo_ids]
    visualise.prediction_grid("val", demo_ids, otsu_masks,
                              "classical_predictions.png",
                              "Classical Otsu segmentation on validation images",
                              n_show=3)

    # Task 3: U-Net, optional because it needs torch
    masks = None
    if use_unet:
        print("\nTask 3: U-Net")
        import unet
        _, train_imgs, train_masks = data_prep.load_split("train")
        val_ids, val_imgs, val_masks = data_prep.load_split("val")
        loaders = unet.make_loaders(train_imgs, train_masks,
                                    val_imgs, val_masks)
        model, history = unet.train_unet(*loaders, epochs=epochs)
        history.to_csv(TABLES / "unet_history.csv", index=False)
        visualise.training_curves(history)

        val_pred = unet.predict_masks(model, val_imgs)
        rows = [metrics.pixel_scores(val_pred[i], val_masks[i])
                for i in range(len(val_ids))]
        print("validation:", pd.DataFrame(rows).mean().round(4).to_dict())
        visualise.prediction_grid("val", val_ids[:4], val_pred[:4],
                                  "unet_predictions.png",
                                  "U-Net segmentation on validation images",
                                  n_show=4)
        _, test_imgs, _ = data_prep.load_split("test")
        masks = unet.predict_masks(model, test_imgs)

    # Task 4: hybrid pipeline over the test split
    print("\nTask 4: pipeline on the test split")
    records, objects, _ = pipeline.run_test_set(masks=masks,
                                                narrate_records=use_llm,
                                                do_audit=use_llm)
    visualise.count_scatter(records)
    visualise.feature_distributions(objects)
    print(f"\ncount MAE {records.count_error.abs().mean():.2f}, "
          f"bias {records.count_error.mean():+.2f}")
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true",
                    help="run the narration and audit steps (needs Ollama)")
    ap.add_argument("--unet", action="store_true",
                    help="train the U-Net and use its masks (needs torch)")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    main(use_llm=args.llm, use_unet=args.unet, epochs=args.epochs)
