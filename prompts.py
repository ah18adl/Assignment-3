# prompts.py, every prompt used in the project, with the reason for each
# design choice.
#
# There are three families. Vision prompts (Task 1) go to the multimodal
# model with an image. Narration prompts (Tasks 2 and 4) go to a text model
# with numbers only. An audit prompt lets a second model check the narration.
#
# Four rules are applied throughout. The model is anchored as descriptive
# rather than diagnostic, because vision models volunteer clinical claims
# that a synthetic microscopy image cannot support. An uncertainty escape is
# allowed, so the model does not have to invent a confident answer. The
# output contract names the exact JSON keys with no example values to copy.
# The narration prompts state that the numbers are the only evidence and
# that the model has not seen the image.

# Task 1
# Naive prompt, included as the comparison baseline. No role, no format, no
# permission to be unsure, so the model free-associates and often diagnoses.
VISION_NAIVE = "What is in this image?"

# Structured prompt. Anchors the model as descriptive, fixes the schema, and
# allows "uncertain" for every field.
VISION_STRUCTURED = """You are a microscopy image description assistant. You describe what is visually present in an image. You do not diagnose, and you do not infer patient information.

Look at the image and fill in the record below.

Rules:
- Describe only what is visible. Do not speculate about disease, prognosis or the patient.
- If a field cannot be determined from the image alone, write exactly "uncertain".
- notable_features must be a list of 2 to 5 short visual observations, each under 12 words.
- image_quality must be one of: "good", "moderate", "poor", "uncertain".
- Respond with ONLY a JSON object with exactly these keys and no others:
  modality, tissue_type, notable_features, image_quality
- Your entire response must start with { and end with }."""

# Used to demonstrate that vision models will answer questions the image
# cannot support, and that an escape hatch fixes it.
VISION_HALLUCINATION = """Looking at this image, what is the patient's age and sex? Give a specific numerical age and a sex."""

VISION_HALLUCINATION_SAFE = """Looking at this image, can you determine the patient's age and sex?

If the image does not contain enough information, reply with exactly:
"This image does not contain enough information to determine age or sex."
Otherwise, state which visual features you used."""

# Task 2
# The model receives measurements only. The prompt says so explicitly, so it
# cannot pretend to have seen the image, and gives the classification cuts so
# that density_class is reproducible rather than a guess.
NARRATION = """You are a quantitative microscopy analyst. You are given measurements computed from a fluorescence microscopy image of stained cell nuclei by a classical image-analysis pipeline. You have NOT seen the image itself. The numbers are your only evidence.

MEASUREMENTS
{measurements}

TASK
1. Write one paragraph, at most 90 words, describing what these measurements imply about the nuclei in this image. Refer to the numbers. Do not invent anything the measurements do not support, and do not diagnose.
2. Then give a JSON record.

Classification rules, apply exactly:
- density_class: "sparse" if n_objects < 15, "normal" if 15 to 44, "dense" if 45 or more.
- shape_regularity: "regular" if mean_circularity >= 0.85, "moderate" if 0.70 to 0.849, "irregular" if below 0.70.
- quality_flag: "ok" normally; "low_contrast" if mean_intensity < 0.45; "possible_oversegmentation" if n_objects > 60 and mean_area < 40.

Respond in exactly this form:

PARAGRAPH:
<your paragraph>

JSON:
{{"n_objects": <int>, "density_class": "<string>", "shape_regularity": "<string>", "quality_flag": "<string>"}}"""

# Task 4
# Same idea as NARRATION but the record carries the image id and mean area,
# because these records are aggregated into one table across the test set.
HYBRID_RECORD = """You are a quantitative microscopy analyst. The measurements below were computed from a U-Net segmentation of a fluorescence microscopy image of stained cell nuclei. You have NOT seen the image. The numbers are your only evidence.

IMAGE ID: {image_id}

MEASUREMENTS
{measurements}

TASK
1. Write one paragraph, at most 80 words, describing the nuclei population in this image using these numbers. Do not diagnose and do not invent detail.
2. Then give a JSON record.

Classification rules, apply exactly:
- density_class: "sparse" if n_objects < 15, "normal" if 15 to 44, "dense" if 45 or more.
- quality_flag: "ok" normally; "low_contrast" if mean_intensity < 0.45; "possible_oversegmentation" if n_objects > 60 and mean_area < 40.

Respond in exactly this form:

PARAGRAPH:
<your paragraph>

JSON:
{{"image_id": "{image_id}", "n_objects": <int>, "mean_area": <number>, "density_class": "<string>", "quality_flag": "<string>"}}"""

# Evaluation
# A second model audits the narration against the same measurements. It is a
# different model family from the narrator so it cannot simply agree with
# itself. No example scores are given, to stop it echoing a template.
AUDIT = """You are a strict auditor. You check whether a written description is supported by the measurements it was based on. You did not write the description.

MEASUREMENTS (ground truth for this check)
{measurements}

DESCRIPTION TO AUDIT
{description}

Score the description on three criteria, each an integer from 1 (very poor) to 5 (perfect):
- faithfulness: every number and claim in the description matches the measurements
- completeness: the description covers the important measurements
- restraint: the description avoids diagnosis, speculation and invented detail

Then list any specific unsupported claims.

Respond with ONLY a JSON object with exactly these keys and no others:
"faithfulness", "completeness", "restraint", "unsupported_claims" (a list of strings, empty if none), "comment" (one sentence).
Your entire response must start with {{ and end with }}."""
