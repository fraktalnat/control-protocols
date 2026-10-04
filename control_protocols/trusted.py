from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

from control_protocols.backdoor import generate_code
from control_protocols.config import DEFAULT_N, TRUSTED_MODEL
from control_protocols.dataset import apps_dataset, hf_backdoor_log_path
from control_protocols.parsing import unescape_code
from control_protocols.prompts import EDIT_OUT_BACKDOOR_PROMPT


@solver
def edit_with_trusted() -> Solver:
    """T-driven editor: returns possibly-cleaned code in metadata['edited code']."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = EDIT_OUT_BACKDOOR_PROMPT.format(
            question=state.metadata["question"],
            code=state.metadata["code"],
        )
        # Replace the entire conversation with the editor prompt — we want
        # T's edit, not a continuation of any prior dialogue.
        state.messages = [ChatMessageUser(content=prompt)]
        await generate(state)
        content = state.output.message.content
        try:
            edited = unescape_code(content.split("```python")[1].split("```")[0])
        except (AttributeError, IndexError):
            edited = state.metadata["code"]
        state.metadata["edited code"] = edited
        return state

    return solve


@task
def trusted_solve_task(n: int = DEFAULT_N) -> Task:
    return Task(dataset=apps_dataset()[:n], plan=[generate_code()], model=TRUSTED_MODEL)


@task
def trusted_edit_task(backdoor_log_path: str | None = None) -> Task:
    if backdoor_log_path is None:
        backdoor_log_path = hf_backdoor_log_path()
    log = read_eval_log(backdoor_log_path)
    print(f"Log path: {backdoor_log_path}")
    print(type(log))
    print(f"Samples: {len(log.samples)}")

    dataset = []
    for sample in log.samples:
        if (
            sample.metadata.get("question")
            and sample.metadata.get("backdoored code")
            and sample.metadata.get("backdoor successful")
        ):
            dataset.append(
                Sample(
                    input=EDIT_OUT_BACKDOOR_PROMPT.format(
                        question=sample.metadata.get("question"),
                        code=sample.metadata.get("backdoored_code"),
                    ),
                    target="edited",
                    metadata={
                        "question": sample.metadata.get("question"),
                        "code": sample.metadata.get("backdoored code"),
                        "inputs": sample.metadata.get("inputs"),
                        "outputs": sample.metadata.get("outputs"),
                        "backdoor": sample.metadata.get("backdoor"),
                        "backdoor_successful": True,
                    },
                )
            )
    return Task(dataset=dataset, plan=[edit_with_trusted()], model=TRUSTED_MODEL)
