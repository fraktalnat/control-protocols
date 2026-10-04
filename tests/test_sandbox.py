from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import Generate, TaskState, solver

from control_protocols import sandbox
from control_protocols.parsing import parse_main_output


def test_test_code(tmp_path):
    """Verify test_code's sandbox round-trip on a known-good code snippet.

    Common breakage:
      - extra repr-wrapping (main("['2','3']") instead of main(['2','3']))
      - forgetting to call unescape_code (HTML entities break the runner)
      - wrong python binary path for the docker/local toggle
    """
    @solver
    def _runner():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # Returns sum of two stdin lines as an int.
            code = (
                "def main(input):\n"
                "    return int(input[0]) + int(input[1])\n"
            )
            res = await sandbox.test_code(state, code, ["2", "3"], False)
            state.metadata["res_success"] = res.success
            state.metadata["res_stdout"] = res.stdout
            state.metadata["res_stderr"] = res.stderr
            state.metadata["parsed"] = parse_main_output(res)
            return state
        return solve

    task = Task(dataset=[Sample(input="x", metadata={})], plan=[_runner()])
    log = eval(task, model="mockllm/model", sandbox="local",
               log_dir=str(tmp_path), display="none")[0]
    md = log.samples[0].metadata
    assert md["res_success"], (
        f"test_code's subprocess failed.\nstderr:\n{md['res_stderr']}\n"
        "Likely causes: (1) wrong python binary path; (2) the runner script "
        "has a syntax error; (3) the code wasn't unescape'd before writing."
    )
    assert md["parsed"] == 5, (
        f"test_code returned parsed={md['parsed']!r} from main(['2','3']), expected 5.\n"
        f"raw stdout: {md['res_stdout']!r}\n"
        "Most likely the runner is calling main('...') (a single string) instead of "
        "main(['2','3']) — an extra layer of repr() in the runner script."
    )
