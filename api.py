import json
from pathlib import Path

from fastapi import FastAPI, Query
from pydantic import BaseModel

from process import ReportAnonymizer

app = FastAPI()


class Item(BaseModel):
    text: str
    uid: str | int | None = None


def process_function(
    data: list[dict],
    output_type: list[str] = ["anon_internal", "anon_hips", "orig_with_phi_predictions"],
):
    with open("/input/data.json", "w") as f:
        json.dump(data, f)

    ReportAnonymizer(
        input_path=Path("/input/data.json"),
        output_path=Path("/output"),
        workdir=Path("/workdir"),
    ).process()

    result = {}
    if "anon_hips" in output_type:
        with open("/output/text_anon_hips.jsonl") as f:
            result["text_anon_hips"] = [json.loads(line) for line in f]

    if "anon_internal" in output_type:
        with open("/output/text_anon_internal.jsonl") as f:
            result["text_anon_internal"] = [json.loads(line) for line in f]

    if "orig_with_phi_predictions" in output_type:
        with open("/output/reports_orig_with_phi_predictions.jsonl") as f:
            result["reports_orig_with_phi_predictions"] = [json.loads(line) for line in f]

    return {"result": "success", "data": result}


@app.post("/process", response_model=dict)
async def process_endpoint(
    data: list[Item],
    output_type: list[str] = Query(["anon_internal", "anon_hips", "orig_with_phi_predictions"]),
):
    processed_data = [item.model_dump(exclude_none=True) for item in data]
    return process_function(data=processed_data, output_type=output_type)
