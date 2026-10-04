from typing import Union

from inspect_ai.solver import TaskState
from inspect_ai.util import ExecResult, sandbox

from control_protocols.config import EXEC_TIMEOUT_SECONDS, docker_exec, local_exec
from control_protocols.parsing import unescape_code


async def test_code(
    state: TaskState,
    code: str,
    input: Union[str, list[str], list[list[str]]],
    USING_DOCKER: bool = False,
) -> ExecResult[str]:
    """Run `main(input)` from `code` using the sandbox; return ExecResult.

    The model's `main` is expected to take a list of stdin lines and return a
    Python value. We print `repr(result)` so `parsing.parse_main_output` can
    recover it.
    """

    code = unescape_code(code)
    await sandbox().exec(["mkdir", "-p", "/tmp/sandbox"])
    await sandbox().write_file("/tmp/sandbox/code.py", code)
    py = docker_exec if USING_DOCKER else local_exec

    runner_script = f"""
import sys
sys.path.insert(0, "/tmp/sandbox")
from code import main
result = main({repr(input)})
print(repr(result))
"""
    await sandbox().write_file("/tmp/sandbox/runner.py", runner_script)
    try:
        return await sandbox().exec(
            [py, "/tmp/sandbox/runner.py"],
            timeout=EXEC_TIMEOUT_SECONDS,
            timeout_retry=False,
        )
    except TimeoutError:
        # Report a timeout as a failed run so callers treat it like any other crash.
        return ExecResult(
            success=False,
            returncode=124,
            stdout="",
            stderr=f"TimeoutError: main() did not finish within {EXEC_TIMEOUT_SECONDS}s",
        )
