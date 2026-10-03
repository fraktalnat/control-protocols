from __future__ import annotations

import functools
import json

from huggingface_hub import hf_hub_download
from inspect_ai.dataset import Dataset, Sample, json_dataset

from control_protocols.config import (
    HF_BACKDOOR_LOG_FILENAME,
    HF_CLEANED_APPS_FILENAME,
    HF_REPO_ID,
    HF_REPO_TYPE,
)
from control_protocols.prompts import CODE_GEN_PROMPT

count = 0


def is_list_of_str(lst):
    return isinstance(lst, list) and all(isinstance(item, str) for item in lst)


def is_list_of_int(lst):
    return isinstance(lst, list) and all(isinstance(item, int) for item in lst)


def is_list_of_list_of_str(lst):
    return isinstance(lst, list) and all(
        isinstance(item, list) and all(isinstance(sub_item, str) for sub_item in item)
        for item in lst
    )


def is_list_of_list_of_int(lst):
    return isinstance(lst, list) and all(
        isinstance(item, list) and all(isinstance(sub_item, int) for sub_item in item)
        for item in lst
    )


def is_list_of_str_one_element(lst):
    return (
        isinstance(lst, list)
        and len(lst) == 1
        and all(isinstance(item, str) for item in lst)
    )


def clean_dataset(data: list[dict]) -> list[dict]:
    """Filter the raw APPS-interview split to entries with simple I/O shapes.

    Only entries whose `inputs` and `outputs` are one of
    ``list[str] | list[int] | list[list[str]] | list[list[int]]`` are kept, and
    each surviving entry gets `input_type` / `output_type` fields describing
    the shape in plain English. Anything else is silently dropped.
    """
    cleaned_data = []
    print(f"Total items in dataset: {len(data)}")
    for item in data:
        try:
            if (
                not isinstance(item, dict)
                or "input_output" not in item
                or item.get("difficulty") != "interview"
            ):
                continue

            input_output = None
            try:
                input_output = json.loads(item["input_output"])
            except Exception:
                if isinstance(item["input_output"], dict):
                    input_output = item["input_output"]
                else:
                    continue

            if "inputs" not in input_output or "outputs" not in input_output:
                continue

            inputs = input_output["inputs"]
            outputs = input_output["outputs"]

            io_ok = (
                is_list_of_str(inputs)
                or is_list_of_int(inputs)
                or is_list_of_list_of_int(inputs)
                or is_list_of_list_of_str(inputs)
            ) and (
                is_list_of_str(outputs)
                or is_list_of_int(outputs)
                or is_list_of_list_of_int(outputs)
                or is_list_of_list_of_str(outputs)
            )
            if not io_ok:
                continue

            entry = item.copy()
            if isinstance(entry["input_output"], str):
                entry["input_output"] = input_output

            if is_list_of_str(inputs):
                entry["input_type"] = "list of strings"
            elif is_list_of_int(inputs):
                entry["input_type"] = "list of integers"
            elif is_list_of_list_of_str(inputs):
                entry["input_type"] = "list of lists of strings"
            else:
                entry["input_type"] = "list of lists of integers"

            if is_list_of_str(outputs):
                entry["output_type"] = "list of strings"
            elif is_list_of_int(outputs):
                entry["output_type"] = "list of integers"
            elif is_list_of_list_of_str(outputs):
                entry["output_type"] = "list of lists of strings"
            else:
                entry["output_type"] = "list of lists of integers"

            cleaned_data.append(entry)
        except Exception:
            continue
    return cleaned_data


def cleaned_apps_path() -> str:
    """Download (or reuse the cached copy of) the cleaned APPS-interview subset."""
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_CLEANED_APPS_FILENAME,
        repo_type=HF_REPO_TYPE,
    )


def hf_backdoor_log_path() -> str:
    """Download (or reuse the cached copy of) the 500-problem backdoor-generation log."""
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_BACKDOOR_LOG_FILENAME,
        repo_type=HF_REPO_TYPE,
    )


def load_cleaned_apps() -> list[dict]:
    """Read the pre-cleaned APPS-interview subset."""
    with open(cleaned_apps_path()) as f:
        return [json.loads(line) for line in f]


def apps_record_to_sample(record: dict) -> Sample:
    """
    Wraps an APPS record into an `inspect_ai.Sample` for code generation.

    Args:
        record: dict with keys `question`, `input_output`, `input_type`, `output_type`.

    Returns:
        Sample whose `input` is the formatted code-generation prompt, `target` is
        the expected `outputs` (as a string), and `metadata` carries everything
        Stage 2 will need to verify a backdoor.
    """
    question = record["question"]
    input_output = record["input_output"]
    input_type = record["input_type"]
    output_type = record["output_type"]
    return Sample(
        input=CODE_GEN_PROMPT.format(
            input_type=input_type, output_type=output_type, question=question
        ),
        metadata={
            "input_type": input_type,
            "output_type": output_type,
            "question": question,
            "inputs": input_output["inputs"],
            "outputs": input_output["outputs"],
        },
        target=str(input_output["outputs"]),
    )


@functools.cache
def apps_dataset() -> Dataset:
    """The cleaned APPS subset as an Inspect dataset (loaded once per process)."""
    return json_dataset(
        str(cleaned_apps_path()),
        sample_fields=apps_record_to_sample,
    )
