# Nuclei Image Analysis with Multimodal LLMs, Classical Features and a U-Net

Segmentation, measurement and language model interpretation of a synthetic
fluorescence microscopy dataset of stained cell nuclei.

The central comparison is between two ways of describing a microscopy image.
The first lets a vision model look at it. The second measures it with a
classical algorithm and lets a text model narrate the measurements. The
second is slower to build and far easier to defend, because every statement
traces back to a number in a feature table.

## How to re-run this

Everything except the U-Net and the language models runs on a laptop. The
U-Net needs torch and the models need Ollama, so the full pipeline is run on
Colab with a GPU.

### On Colab, all four tasks

1. Open notebooks/analysis.ipynb in Colab and set Runtime, Change runtime
   type, T4 GPU.
2. Run the first cell. It installs zstd, which the Ollama installer needs and
   the Colab image does not ship, then installs Ollama, starts the server as a
   background process and pulls the models. Allow about 10 minutes.
3. Run the second cell and upload nuclei-project-colab.zip, which contains
   the source code, the tests and the dataset archive. Alternatively set REPO
   at the top of that cell to a git URL and it clones instead.
4. Runtime, Run all.

The Ollama server sometimes stops when the cell that started it finishes. If
a later cell says it cannot connect, restart it with:

    import subprocess, urllib.request, time
    subprocess.Popen(["ollama", "serve"])
    time.sleep(5); urllib.request.urlopen("http://127.0.0.1:11434")

### Locally, the parts that need no GPU and no language model

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

    python run_pipeline.py          data prep, EDA, classical route, Task 4
    python run_pipeline.py --unet   also trains the U-Net, needs torch
    python run_pipeline.py --llm    also narrates and audits, needs Ollama
    python -m pytest tests -q       13 sanity checks

The dataset is expected at data/nuclei_dataset. Unzip nuclei_dataset.zip into
the data folder if it is not already there.

## Results

Segmentation and counting, held out images:

| route | split | Dice | IoU | count MAE |
|---|---|---|---|---|
| Otsu and watershed | validation | 0.9759 | 0.9530 | 4.20 |
| U-Net, Dice and BCE | validation | 0.9960 | 0.9920 | n/a |
| Otsu and watershed | test | 0.9754 | 0.9520 | 3.92 |
| U-Net, Dice and BCE | test | 0.9955 | 0.9910 | 3.25 |

Loss ablation, with the architecture, seed and epoch budget held fixed:

| loss | val Dice | val IoU | predicted area over true area | Dice at epoch 5 |
|---|---|---|---|---|
| Dice and BCE | 0.9959 | 0.9918 | 0.9974 | 0.9218 |
| BCE only | 0.9956 | 0.9913 | 0.9988 | 0.9361 |
| Dice only | 0.9956 | 0.9912 | 1.0007 | 0.8766 |

The three losses are indistinguishable at convergence and none of them under
segments, so the class imbalance argument for combining them does not hold on
this data. The cross entropy term only speeds up early training.

Performance by density regime:

| regime | Otsu Dice | U-Net Dice | true count | detected | count MAE |
|---|---|---|---|---|---|
| sparse | 0.9748 | 0.9953 | 8.0 | 8.0 | 0.0 |
| normal | 0.9768 | 0.9965 | 25.5 | 26.0 | 1.5 |
| dense | 0.9735 | 0.9958 | 76.5 | 73.0 | 3.5 |
| clustered | 0.9766 | 0.9956 | 39.5 | 28.0 | 11.5 |

This is the main finding. The pixel metrics barely move across regimes while
the counting error changes by a factor of ten. Dice answers whether a pixel
is a nucleus, which is easy on a high contrast stain. Counting answers how
many nuclei there are, which needs objects that touch to be separated.
Looking only at Dice would hide the failure completely.

## Models

| role | model | reason |
|---|---|---|
| image description | qwen2.5vl:7b | multimodal, runs locally through Ollama |
| measurement narration | llama3.2 | text only, never sees the image |
| audit | qwen2.5:3b | a different family from the narrator, so it cannot simply agree with itself |

The brief specifies llama3.2-vision. It downloads but fails at load with the
error unknown model architecture: mllama on current Ollama builds. The
function available_vision_model therefore tests each candidate with a one
token request and falls through llama3.2-vision, qwen2.5vl:7b, qwen2.5vl:3b,
llava:7b and moondream until one responds.

## Layout

    run_pipeline.py           end to end pipeline for the parts that need no GPU
    notebooks/analysis.ipynb  all four tasks, written for Colab
    src/data_prep.py          loading, grayscale conversion, resizing, splits
    src/eda.py                sample grids, intensity histograms, corrupted images
    src/classical.py          Otsu, morphology, watershed, regionprops features
    src/unet.py               small U-Net, three losses, training, ablation
    src/metrics.py            Dice, IoU, precision, recall, counting error
    src/llm.py                Ollama client, response cache, tolerant JSON parser
    src/prompts.py            every prompt, with the reason for each choice
    src/pipeline.py           hybrid pipeline, narration and audit steps
    src/visualise.py          segmentation, prediction and accuracy figures
    tests/test_pipeline.py    13 sanity checks including leakage guards
    outputs/                  figures, tables and cached model responses

## Method notes

The blue channel is used rather than luminance grayscale. Standard luminance
weights blue at 0.114, which suppresses exactly the stain signal the analysis
depends on.

Watershed on the distance transform separates touching nuclei. Plain
connected components merge them, which on one clustered image finds 15
objects where there are 48.

The watershed separation and the minimum object area were tuned on the
training split only, by grid search over distance 1 to 8 and area 15 to 50.

U-Net augmentation is flips and 90 degree rotations only. These are label
preserving for objects with no canonical orientation. Brightness jitter is
deliberately excluded, because intensity statistics are measured later as
features.

Every language model call is cached in outputs/cache, keyed on the model, the
prompt, the image and the temperature, so results are reproducible and each
raw response can be inspected.

The narration prompts state that the model has not seen the image and give
the classification cut offs, so the JSON fields are reproducible rather than
a guess.

## References

Milletari, F., Navab, N. and Ahmadi, S. (2016). V-Net: Fully Convolutional
Neural Networks for Volumetric Medical Image Segmentation. 3DV 2016.

Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms.
IEEE Transactions on Systems, Man and Cybernetics, 9(1), pages 62 to 66.

Ronneberger, O., Fischer, P. and Brox, T. (2015). U-Net: Convolutional
Networks for Biomedical Image Segmentation. MICCAI 2015, pages 234 to 241.
