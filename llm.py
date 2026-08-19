# llm.py, Ollama client for the vision and text models.
#
# Models:
#   VISION_MODEL     llama3.2-vision, describes an image directly (Task 1)
#   TEXT_MODEL       llama3.2, narrates measurements (Tasks 2 and 4)
#   AUDIT_MODEL      qwen2.5:3b, independently audits the narration
#
# The audit model is a different family from the narrator on purpose: a model
# grading its own output tends to agree with itself.
#
# Every call is cached on disk, keyed by model, prompt, image and temperature,
# so a rerun costs nothing and every raw response can be inspected afterwards.

import hashlib
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "cache"

VISION_MODEL = "llama3.2-vision"
TEXT_MODEL = "llama3.2"
AUDIT_MODEL = "qwen2.5:3b"

# Alternatives, tried in order. llama3.2-vision uses the mllama architecture,
# which recent Ollama builds no longer load ("unknown model architecture:
# 'mllama'"), so a working substitute is needed. Qwen2.5-VL is the course
# sanctioned alternative; llava and moondream are smaller still.
VISION_FALLBACKS = ["qwen2.5vl:7b", "qwen2.5vl:3b", "llava:7b", "moondream"]


def _key(model, prompt, temperature, images):
    "Cache filename from everything that can change the response."
    img_part = "".join(Path(p).name for p in (images or []))
    raw = f"{model}|{temperature}|{prompt}|{img_part}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:24]
    return CACHE / f"{model.replace(':', '_').replace('.', '_')}_{h}.json"


def generate(prompt, model=TEXT_MODEL, images=None, temperature=0.0,
             use_cache=True, num_predict=None):
    """Single-turn chat call to Ollama, returning the response text.

    images: list of file paths for a multimodal model, or None for text only.
    temperature 0 is used everywhere except the stochasticity demonstration,
    which deliberately disables the cache so repeated runs really do differ.
    """
    import ollama

    CACHE.mkdir(parents=True, exist_ok=True)
    path = _key(model, prompt, temperature, images)
    if use_cache and path.exists():
        return json.loads(path.read_text())["response"]

    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = [str(p) for p in images]
    options = {"temperature": temperature}
    if num_predict:
        options["num_predict"] = num_predict

    t0 = time.time()
    response = ollama.chat(model=model, messages=[message], options=options)
    text = response["message"]["content"].strip()

    path.write_text(json.dumps({"model": model, "temperature": temperature,
                                "images": [str(p) for p in (images or [])],
                                "elapsed_s": round(time.time() - t0, 1),
                                "prompt": prompt, "response": text}, indent=1))
    return text


def available_vision_model(preferred=VISION_MODEL, verify=True):
    """Return the first vision model that is installed and actually loads.

    Being installed is not sufficient: llama3.2-vision downloads happily and
    then fails at load time on Ollama builds that dropped the mllama
    architecture. With verify=True each candidate is given a one-token
    request, and the first that responds is returned.
    """
    import ollama

    try:
        listed = ollama.list()
        names = {m.get("model", m.get("name", "")) for m in listed["models"]}
    except Exception:
        return preferred

    for candidate in [preferred] + VISION_FALLBACKS:
        if not any(n.startswith(candidate) for n in names):
            continue
        if not verify:
            return candidate
        try:
            ollama.chat(model=candidate,
                        messages=[{"role": "user", "content": "hi"}],
                        options={"num_predict": 1})
            return candidate
        except Exception as exc:
            print(f"{candidate} is installed but will not load: "
                  f"{str(exc)[:120]}")
    return preferred


def extract_json(text):
    """Pull the first JSON object out of a model response.

    Handles the faults small local models actually produce: markdown fences,
    prose before the brace, thousands separators inside numbers, missing
    commas between string items, and truncated output with unclosed brackets.
    """
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"(\d),(?=\d{3})", r"\1", text)
    text = re.sub(r'"\s*\n(\s*)"', '",\n\\1"', text)
    start = text.find("{")
    if start == -1:
        return None
    frag = text[start:]

    depth = 0                                   # try a balanced object first
    for i, ch in enumerate(frag):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(frag[:i + 1])
                except json.JSONDecodeError:
                    break

    frag = frag.rstrip().rstrip(",")            # repair a truncated response
    stack, in_str, esc = [], False, False
    for ch in frag:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        frag += '"'
    for ch in reversed(stack):
        frag += "}" if ch == "{" else "]"
    try:
        return json.loads(frag)
    except json.JSONDecodeError:
        return None


def split_paragraph_json(text):
    """Split a 'PARAGRAPH: ... JSON: {...}' response into its two parts.

    Returns (paragraph string, parsed dict or None). Falls back to treating
    everything before the first brace as the paragraph.
    """
    if not text:
        return "", None
    record = extract_json(text)
    if "PARAGRAPH:" in text:
        body = text.split("PARAGRAPH:", 1)[1]
        paragraph = body.split("JSON:", 1)[0].strip()
    else:
        cut = text.find("{")
        paragraph = (text[:cut] if cut > 0 else text).strip()
    return paragraph, record
