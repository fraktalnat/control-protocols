from __future__ import annotations

import ast
import html
from typing import Any, Optional


def unescape_code(code: str) -> str:
    """Reverse the HTML/escape mangling that occasionally hits LLM-generated code."""
    code = code.replace("\r\n", "\n").replace("\r", "\n")

    prev = None
    while prev != code:
        prev = code
        code = html.unescape(code)
        code = (
            code.replace("&apos;", "'")
            .replace("&#39;", "'")
            .replace("&#x27;", "'")
            .replace("&quot;", '"')
            .replace("&#34;", '"')
            .replace("&#x22;", '"')
        )
    if "\n" not in code and "\\n" in code:
        code = code.encode("utf-8").decode("unicode_escape")

    return code


def between(text: str, start: str, end: str) -> Optional[str]:
    if start not in text:
        return None
    after = text.split(start, 1)[1]
    if end not in after:
        return None
    return after.split(end, 1)[0]


def parse_trigger(raw: str) -> list[str]:
    raw = raw.strip()
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        parsed = raw.split("\n")
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)]


def parse_main_output(res) -> Any:
    if not res.success:
        return ("__ERROR__", res.returncode, res.stderr)
    try:
        return ast.literal_eval(res.stdout.strip())
    except (ValueError, SyntaxError):
        return ("__UNPARSEABLE__", res.stdout)
