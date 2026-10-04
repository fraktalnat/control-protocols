from inspect_ai.dataset import Sample

from control_protocols.dataset import apps_record_to_sample


def test_apps_record_to_sample():
    """Every downstream solver reads state.metadata['question'|'inputs'|'outputs'|'input_type'|'output_type'].
    If apps_record_to_sample drops one, every Stage 2+ solver will KeyError mid-eval.
    """
    record = {
        "question": "Compute the sum of the two integers on stdin.",
        "input_output": {"inputs": [["2", "3"], ["10", "5"]], "outputs": [5, 15]},
        "input_type": "list of strings",
        "output_type": "list of integers",
    }
    sample = apps_record_to_sample(record)
    assert isinstance(sample, Sample), (
        f"apps_record_to_sample should return an inspect_ai.Sample, got {type(sample).__name__}. "
        "Make sure to construct and return a Sample, not a dict."
    )
    md = sample.metadata or {}
    required = {"question", "inputs", "outputs", "input_type", "output_type"}
    missing = required - md.keys()
    assert not missing, (
        f"Sample.metadata is missing required keys: {sorted(missing)}. "
        f"Got keys: {sorted(md.keys())}. "
        "Every Stage 2 solver reads these via state.metadata[...] and will KeyError without them."
    )
    assert md["question"] == record["question"], (
        f"metadata['question'] mismatch: expected the original question text, got {md['question']!r}. "
        "Question text is the join key across all Stage 2-4 logs in assemble_protocol_data."
    )
    assert md["inputs"] == record["input_output"]["inputs"], (
        f"metadata['inputs'] should be the raw inputs list, got {md['inputs']!r}. "
        "Don't stringify; downstream test_code iterates these directly."
    )
    assert md["outputs"] == record["input_output"]["outputs"], (
        f"metadata['outputs'] should be the raw outputs list, got {md['outputs']!r}. "
        "test_backdoor compares parse_main_output(...) == metadata['outputs']; types must match."
    )
    assert isinstance(sample.target, str), (
        f"Sample.target must be a str (Inspect requirement), got {type(sample.target).__name__}. "
        "Use target=str(input_output['outputs'])."
    )
