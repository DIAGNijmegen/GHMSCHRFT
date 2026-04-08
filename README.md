# GHMSCHRFT — Dutch Medical Report Anonymizer

GHMSCHRFT anonymizes Dutch radiology and pathology reports by detecting and replacing personal health information (PHI) using a fine-tuned transformer model.

The model was trained on reports from Radboudumc (RUMC) and ZGT and is available on HuggingFace: [LMMasters/GHMSCHRFT-v1](https://huggingface.co/LMMasters/GHMSCHRFT-v1).

## What it detects

| Tag | Description |
|---|---|
| `<PERSOON>` | Person names |
| `<PERSOONAFKORTING>` | Name abbreviations |
| `<DATUM>` | Dates |
| `<TIJD>` | Times |
| `<LEEFTIJD>` | Ages |
| `<ADRES>` | Addresses |
| `<ZIEKENHUIS>` | Hospital names |
| `<TELEFOONNUMMER>` | Phone numbers |
| `<RAPPORT_ID>` | Report identifiers |
| `<DOCUMENTID>` / `<DOCUMENTNUMMER>` | Document numbers |
| `<PHINUMMER>` | Patient/PHI numbers |
| `<BIGNUMMER>` / `<AGBNUMMER>` | Practitioner registration numbers |
| `<BSN>` | Citizen service numbers |
| `<URL>` | URLs |
| `<STUDIE_NAAM>` | Study names |
| `<ACCREDATIE_NUMMER>` | Accreditation numbers |

## Output formats

Each report is returned in three forms:
- **`text_anon_internal`** — PHI replaced by tags (`<PERSOON>`, `<DATUM>`, etc.)
- **`text_anon_hips`** — PHI replaced by realistic fake data (names, dates, etc.)
- **`reports_orig_with_phi_predictions`** — Original text with character-level PHI annotations

---

## Quick start

### Option 1: Docker (recommended)

**Build the image** (downloads model weights from HuggingFace, ~2 GB):
```bash
docker build -t ghmschrft .
```

**Run on a file:**
```bash
docker run --rm --gpus all \
    -v /path/to/input:/input \
    -v /path/to/output:/output \
    ghmschrft \
    python process.py --input /input/reports.json --output /output/
```

**Run as an API server:**
```bash
docker compose up
```
The API will be available at `http://localhost:1986`.

---

### Option 2: Local Python

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run on a file:**
```bash
python process.py --input test-input/reports.json --output output/
```

The model is downloaded automatically from HuggingFace on first run and cached locally.

---

## Input format

JSON or JSONL file with a list of reports:

```json
[
  {
    "uid": "case1",
    "text": "Patiënt Jan de Vries werd gezien op 3 april 2024."
  },
  {
    "uid": "case2",
    "text": "Onderzoek uitgevoerd door Dr. P. Jansen in het Radboudumc."
  }
]
```

- `uid` — unique identifier (string or integer, optional)
- `text` — the report text (required)

A directory of `.txt` files is also accepted.

---

## API usage

With the server running (`docker compose up`):

```python
import json
import requests

with open("test-input/reports.json") as f:
    payload = json.load(f)

response = requests.post(
    "http://localhost:1986/process",
    json=payload,
)
result = response.json()

# Access results
for report in result["data"]["text_anon_hips"]:
    print(report["uid"], report["text"])
```

**Request only specific output types:**
```python
# Only HIPS-anonymized output
response = requests.post(
    "http://localhost:1986/process?output_type=anon_hips",
    json=payload,
)

# Only tag-replaced output
response = requests.post(
    "http://localhost:1986/process?output_type=anon_internal",
    json=payload,
)

# Only PHI span annotations
response = requests.post(
    "http://localhost:1986/process?output_type=orig_with_phi_predictions",
    json=payload,
)
```

**Example response:**
```json
{
  "result": "success",
  "data": {
    "text_anon_hips": [
      {"uid": "case1", "text": "Patiënt Pieter van den Berg werd gezien op 17 februari 2024."}
    ],
    "text_anon_internal": [
      {"uid": "case1", "text": "Patiënt <PERSOON> werd gezien op <DATUM>."}
    ],
    "reports_orig_with_phi_predictions": [
      {
        "uid": "case1",
        "text": "Patiënt Jan de Vries werd gezien op 3 april 2024.",
        "label": [[9, 22, "<PERSOON>"], [36, 49, "<DATUM>"]]
      }
    ]
  }
}
```

---

## Model

- **HuggingFace**: [LMMasters/GHMSCHRFT-v1](https://huggingface.co/LMMasters/GHMSCHRFT-v1)
- **Base model**: `joeranbosma/dragon-roberta-large-mixed-domain`
- **Training data**: Radboudumc radiology, Radboudumc pathology, ZGT radiology (~4,900 reports)
- **Test performance**: micro F1 0.94, detection recall 0.95

---

## Citation

If you use this model, please cite:

```
@software{GHMSCHRFT,
  author = {Builtjes, Luc},
  title  = {GHMSCHRFT: Dutch Medical Report Anonymizer},
  year   = {2025},
  url    = {https://github.com/LMMasters/GHMSCHRFT}
}
```
