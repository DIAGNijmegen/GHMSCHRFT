import argparse
import hashlib
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import spacy
from dragon_baseline import DragonBaseline
from dragon_baseline.nlp_algorithm import TaskDetails
from dragon_prep.ner import ner_tokenizer
from dutch_med_hips import HideInPlainSight, schema, settings
from dutch_med_hips.schema import PHIType
from tqdm import tqdm


# =============================================================================
# dutch-med-hips custom tag registration
# =============================================================================

def register_custom_phi_tags():
    schema.DEFAULT_PATTERNS.setdefault(PHIType.GENERIC_ID, []).extend(
        [
            r"<DOCUMENTNUMMER>",
            r"<DOCUMENTID>",
            r"<RAPPORT_ID>",
            r"<RAPPORT_ID\.T_NUMMER>",
            r"<RAPPORT_ID\.R_NUMMER>",
            r"<RAPPORT_ID\.C_NUMMER>",
            r"<RAPPORT_ID\.DPA_NUMMER>",
            r"<RAPPORT_ID\.RPA_NUMMER>",
            r"<PHINUMMER>",
            r"<PATIENTNUMMER>",
            r"<AGBNUMMER>",
            r"<BIGNUMMER>",
            r"<TNUMMER>",
            r"<ZNUMMER>",
        ]
    )
    settings.ID_TEMPLATES_BY_TAG["<DOCUMENTNUMMER>"] = "##########"
    settings.ID_TEMPLATES_BY_TAG["<DOCUMENTID>"] = "####"
    settings.ID_TEMPLATES_BY_TAG["<RAPPORT_ID>"] = "R-########"
    settings.ID_TEMPLATES_BY_TAG["<RAPPORT_ID.T_NUMMER>"] = "T-########"
    settings.ID_TEMPLATES_BY_TAG["<RAPPORT_ID.R_NUMMER>"] = "R-########"
    settings.ID_TEMPLATES_BY_TAG["<RAPPORT_ID.C_NUMMER>"] = "C-########"
    settings.ID_TEMPLATES_BY_TAG["<RAPPORT_ID.DPA_NUMMER>"] = "DPA-######"
    settings.ID_TEMPLATES_BY_TAG["<RAPPORT_ID.RPA_NUMMER>"] = "RPA-######"
    settings.ID_TEMPLATES_BY_TAG["<PHINUMMER>"] = "##########"
    settings.ID_TEMPLATES_BY_TAG["<PATIENTNUMMER>"] = "########"
    settings.ID_TEMPLATES_BY_TAG["<AGBNUMMER>"] = "#########"
    settings.ID_TEMPLATES_BY_TAG["<BIGNUMMER>"] = "###########"
    settings.ID_TEMPLATES_BY_TAG["<TNUMMER>"] = "T########"
    settings.ID_TEMPLATES_BY_TAG["<ZNUMMER>"] = "Z#######"


register_custom_phi_tags()


# =============================================================================
# HIPS helpers
# =============================================================================

PHI_TAG_PATTERN = re.compile(r"<[A-Z_\.]+>")


def apply_hips_to_anonymized_text(text: str, seed: int) -> str:
    tag_positions = [
        (m.start(), m.end(), m.group())
        for m in PHI_TAG_PATTERN.finditer(text)
    ]
    if not tag_positions:
        return text
    hips = HideInPlainSight(default_seed=seed, enable_header=False, enable_random_typos=False)
    return hips.run(text, ner_labels=tag_positions)["text"]


# =============================================================================
# Post-processing
# =============================================================================

def merge_adjacent_entities(labels: list[str]) -> list[str]:
    merged = labels.copy()
    i = 0
    while i < len(merged):
        if merged[i].startswith("B-"):
            tag = merged[i][2:]
            j = i + 1
            while j < len(merged) and merged[j] == f"I-{tag}":
                j += 1
            if j < len(merged) and merged[j] == f"B-{tag}":
                merged[j] = f"I-{tag}"
                continue
            i = j
        else:
            i += 1
    return merged


def tokenize(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokenizer = ner_tokenizer()
    processed = []
    for row in tqdm(data, desc="Tokenizing"):
        text = row["text"]
        tokenized_text = tokenizer(text)
        filtered_tokens = [t.text for t in tokenized_text if not t.text.isspace()]
        row = deepcopy(row)
        row["text_parts"] = filtered_tokens
        row["tokenized_text"] = tokenized_text
        processed.append(row)
    return processed


def align_predictions(
    tokenized_text: list[spacy.tokens.Token],
    predictions: list[str],
) -> list[str]:
    all_predictions = []
    filtered_index = 0
    for token in tokenized_text:
        if not token.text.isspace():
            all_predictions.append(predictions[filtered_index])
            filtered_index += 1
        else:
            if filtered_index >= len(predictions):
                all_predictions.append("O")
            elif predictions[filtered_index] == predictions[filtered_index - 1]:
                all_predictions.append(predictions[filtered_index])
            elif (
                predictions[filtered_index].startswith("I-")
                and predictions[filtered_index - 1].startswith("B-")
            ):
                all_predictions.append(predictions[filtered_index])
            else:
                all_predictions.append("O")
    return all_predictions


def token_text(token: spacy.tokens.Token) -> str:
    return token.text + token.whitespace_


def construct_anonymized_report(
    tokenized_text: list[spacy.tokens.Token],
    predictions: list[str],
) -> str:
    reconstructed = ""
    for i, (token, prediction) in enumerate(zip(tokenized_text, predictions)):
        if prediction == "O":
            reconstructed += token_text(token)
        elif prediction.startswith("B-"):
            reconstructed += f"{prediction[2:]}"
            whitespace = token.whitespace_
            for t, p in zip(tokenized_text[i + 1:], predictions[i + 1:]):
                if p == f"I-{prediction[2:]}":
                    whitespace = t.whitespace_
                else:
                    break
            reconstructed += whitespace
    return reconstructed


def construct_annotated_report(
    tokenized_text: list[spacy.tokens.Token],
    predictions: list[str],
) -> tuple[str, list[tuple[int, int, str]]]:
    reconstructed = ""
    annotations = []
    current_entity = None
    entity_start = None
    entity_length = 0

    for idx, (token, prediction) in enumerate(zip(tokenized_text, predictions)):
        if prediction == "O":
            if current_entity:
                annotations.append((entity_start, entity_start + entity_length, current_entity))
                current_entity = None
            reconstructed += token_text(token)
        elif prediction.startswith("B-"):
            if current_entity:
                annotations.append((entity_start, entity_start + entity_length, current_entity))
            current_entity = prediction[2:]
            entity_start = len(reconstructed)
            reconstructed += token_text(token)
            next_pred = predictions[idx + 1] if idx + 1 < len(predictions) else "O"
            if next_pred == "O" or next_pred.startswith("B-"):
                entity_length = len(token.text)
            else:
                entity_length = len(token_text(token))
        elif prediction.startswith("I-"):
            reconstructed += token_text(token)
            next_pred = predictions[idx + 1] if idx + 1 < len(predictions) else "O"
            if next_pred == "O" or next_pred.startswith("B-"):
                entity_length += len(token.text)
            else:
                entity_length += len(token_text(token))

    if current_entity:
        annotations.append((entity_start, entity_start + entity_length, current_entity))

    return reconstructed, annotations


# =============================================================================
# Main class
# =============================================================================

@dataclass
class Case:
    uid: str = ""
    text_annot: str = ""
    phi_predictions: list[tuple[int, int, str]] = field(default_factory=list)
    text_anon: str = ""
    text_anon_hips: str = ""


class ReportAnonymizer(DragonBaseline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df: pd.DataFrame = None
        self.results: list[Case] = None

        model_path = Path("/opt/app/model")
        if not model_path.exists():
            # Download model from HuggingFace on first run (cached locally)
            from huggingface_hub import snapshot_download
            print("Downloading model from HuggingFace (LMMasters/GHMSCHRFT-v1)...")
            model_path = Path(snapshot_download("LMMasters/GHMSCHRFT-v1"))

        self.model_save_dir = str(model_path)
        self.task = TaskDetails.from_json(Path(__file__).parent / "nlp-task-configuration.json")
        self.common_prefix = []

    def process(self):
        self.df = self.load()
        self.validate()
        self.results = self.process_cases()
        self.save(results=self.results)

    def load(self, path: Path | None = None) -> pd.DataFrame:
        if path is None:
            path = self._input_path
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_dir():
            filepaths = list(path.glob("*.txt"))
            if not filepaths:
                filepaths = list(path.glob("*.json*"))
                if len(filepaths) == 1:
                    return self.load(filepaths[0])
                raise FileNotFoundError(
                    f"No .txt files or single JSON(L) file found in {path}"
                )
            data = []
            for fn in sorted(filepaths):
                with open(fn) as f:
                    data.append({"text": f.read()})
            return pd.DataFrame(data)
        else:
            if path.suffix == ".json":
                return pd.read_json(path)
            elif path.suffix == ".jsonl":
                return pd.read_json(path, lines=True)
            else:
                raise ValueError(f"Expected .json or .jsonl file, got: {path}")

    def validate(self, df: pd.DataFrame | None = None):
        if df is None:
            df = self.df
        if len(df) == 0:
            raise ValueError("The provided data is empty")
        if "text" not in df.columns:
            raise ValueError("Input data must contain a 'text' field")

    def postprocess_predictions(
        self,
        tokenized_text: list[spacy.tokens.Token],
        predictions: list[str],
    ) -> list[str]:
        """Refine <RAPPORT_ID> predictions to specific subtypes."""
        refined = predictions.copy()
        for i, (token, prediction) in enumerate(zip(tokenized_text, predictions)):
            if prediction == "B-<RAPPORT_ID>":
                entity_text = token.text
                j = i + 1
                while j < len(tokenized_text) and predictions[j] == "I-<RAPPORT_ID>":
                    entity_text += token_text(tokenized_text[j])
                    j += 1
                if "T" in entity_text:
                    refined_label = "B-<RAPPORT_ID.T_NUMMER>"
                elif "R" in entity_text:
                    refined_label = "B-<RAPPORT_ID.R_NUMMER>"
                elif "C" in entity_text:
                    refined_label = "B-<RAPPORT_ID.C_NUMMER>"
                elif "DPA" in entity_text:
                    refined_label = "B-<RAPPORT_ID.DPA_NUMMER>"
                elif "RPA" in entity_text:
                    refined_label = "B-<RAPPORT_ID.RPA_NUMMER>"
                else:
                    refined_label = "B-<RAPPORT_ID>"
                refined[i] = refined_label
                for k in range(i + 1, j):
                    refined[k] = refined_label.replace("B-", "I-")
        return refined

    def process_cases(self, df: pd.DataFrame | None = None) -> list[Case]:
        if df is None:
            df = self.df

        data = df.to_dict(orient="records")
        data = tokenize(data)
        df = pd.DataFrame(data)

        if "uid" not in df.columns:
            df["uid"] = df.index

        predictions = self.predict_ner(df=df)
        df = pd.merge(df, predictions, on="uid")

        results = []
        for _, row in df.iterrows():
            tokenized_text = row["tokenized_text"]
            preds = list(row["named_entity_recognition"])

            preds = align_predictions(tokenized_text, preds)
            preds = merge_adjacent_entities(preds)
            preds = self.postprocess_predictions(tokenized_text, preds)

            reconstructed, phi_predictions = construct_annotated_report(tokenized_text, preds)
            report_anon = construct_anonymized_report(tokenized_text, preds)

            seed = int(hashlib.md5(report_anon.encode()).hexdigest(), 16) % 2**32
            report_anon_hips = apply_hips_to_anonymized_text(report_anon, seed=seed)

            results.append(Case(
                uid=row["uid"],
                text_annot=reconstructed,
                phi_predictions=phi_predictions,
                text_anon=report_anon,
                text_anon_hips=report_anon_hips,
            ))

        return results

    def save(self, results: list[Case] | None = None, output_dir: Path | None = None):
        if results is None:
            results = self.results
        if output_dir is None:
            output_dir = Path(self._output_path)

        pd.DataFrame([{"uid": c.uid, "text": c.text_anon} for c in results]).to_json(
            output_dir / "text_anon_internal.jsonl", lines=True, orient="records"
        )
        pd.DataFrame([{"uid": c.uid, "text": c.text_annot, "label": c.phi_predictions} for c in results]).to_json(
            output_dir / "reports_orig_with_phi_predictions.jsonl", lines=True, orient="records"
        )
        pd.DataFrame([{"uid": c.uid, "text": c.text_anon_hips} for c in results]).to_json(
            output_dir / "text_anon_hips.jsonl", lines=True, orient="records"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize Dutch medical reports")
    parser.add_argument("--input", type=str, default="/input", help="Path to input file or directory")
    parser.add_argument("--output", type=str, default="/output", help="Path to output directory")
    args = parser.parse_args()

    ReportAnonymizer(
        input_path=Path(args.input),
        output_path=Path(args.output),
    ).process()
