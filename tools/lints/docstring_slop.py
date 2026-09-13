"""A docstring states the contract; it does not narrate, decorate, or repeat.

Ruff's ``D``/``DOC`` rules check a docstring's shape. This rule checks what is
inside it — the eight patterns that made 900+ docstrings in ``app/`` read like
pasted PR descriptions:

  DS1  longer than 6 lines (function) / 12 (class) / 15 (module)
  DS2  backticks — Google style is plain text; nothing here renders markup
  DS3  RST/Sphinx markup: ``x``, :param, :returns:, :raises:, .. note::, >>>
  DS4  a ``test_*`` docstring longer than one line — the test name is the doc
  DS5  a summary that only restates the function name
  DS6  an Args entry that only restates the argument name
  DS7  a type inside an Args entry — the signature already carries it
  DS8  an Examples section — prose code rots; the call sites are the examples

Docstrings that are runtime data are skipped, never rewritten: ``@tool`` /
``@custom_tool`` bodies are the model-facing tool description, ``@with_doc``
injects them, ``@router.*`` / ``@app.*`` handlers feed OpenAPI, and
``BaseModel`` / ``BaseSettings`` / ``BaseTool`` class docstrings become schema
descriptions.

No allowlist and no ``noqa`` — the cleanup that introduced this rule brought
``app/`` and ``tests/`` to zero, and a finding is fixed by shortening.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from functools import cache
from pathlib import Path
import re

from _common import Violation

Documented = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef

RULE = "docstring-content"
WHY = (
    "a docstring that narrates, decorates or restates the signature costs every "
    "reader time and tells them nothing the code does not"
)
DOC = "tools/lints/README.md#docstring-content"

#: Runs on ``tests/`` too — DS4 lives there, and a backtick is a backtick.
INCLUDES_TESTS = True

FUNCTION_MAX_LINES = 6
CLASS_MAX_LINES = 12
MODULE_MAX_LINES = 15

#: Decorators whose docstring is consumed at runtime as a tool description;
#: bare (``@tool``) or dotted (``@composio.tools.custom_tool(...)``).
_RUNTIME_DECORATORS = frozenset({"tool", "custom_tool", "with_doc"})
#: Route registrations feed OpenAPI. Only the dotted form (``@router.patch``)
#: counts — a bare ``@patch`` is ``unittest.mock.patch``.
_ROUTE_METHODS = frozenset({"get", "post", "put", "patch", "delete", "api_route", "websocket"})
#: Base classes whose class docstring becomes a schema description. Subclasses
#: of these defined anywhere in the app (CamelModel, MongoDocument, …) count too.
_RUNTIME_BASES = frozenset({"BaseModel", "BaseSettings", "BaseTool", "BaseToolkit"})
_APP_ROOT = Path(__file__).resolve().parents[2] / "apps" / "api" / "app"

_RST = re.compile(r"``|:param\b|:returns?:|:raises?:|:type\b|:rtype:|\.\. \w+::|^\s*>>>", re.M)
_EXAMPLES = re.compile(r"^\s*Examples?\s*:\s*$", re.M)
_ARGS_SECTION = re.compile(r"^\s*(?:Args|Arguments)\s*:\s*\n((?:[ \t]+\S.*\n?)+)", re.M)
_ARGS_ENTRY = re.compile(r"\s*(\*{0,2}\w+)\s*(\([^)]*\))?\s*:\s*(.*)")

#: Words that carry no information on their own; a description made only of
#: these plus the name's own tokens is a restatement.
_FILLER = frozenset(
    (
        "the",
        "a",
        "an",
        "of",
        "to",
        "for",
        "and",
        "or",
        "in",
        "on",
        "is",
        "this",
        "that",
        "with",
        "from",
        "if",
        "by",
        "its",
        "it",
        "given",
        "value",
        "values",
        "object",
        "instance",
        "string",
        "list",
        "dict",
        "id",
        "name",
        "return",
        "returns",
        "get",
        "gets",
        "set",
        "sets",
        "check",
        "checks",
        "whether",
        "current",
        "specified",
        "provided",
        "new",
        "all",
        "data",
        "info",
        "information",
    )
)


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w and w not in _FILLER}


def _last_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _is_runtime_decorator(decorator: ast.expr) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Attribute):
        return target.attr in _RUNTIME_DECORATORS or target.attr in _ROUTE_METHODS
    return isinstance(target, ast.Name) and target.id in _RUNTIME_DECORATORS


def _class_bases(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: {_last_name(base) for base in node.bases}
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }


@cache
def _app_class_bases() -> dict[str, set[str]]:
    """Return class name -> base names across the app, so a single-file run sees indirect bases."""
    bases: dict[str, set[str]] = {}
    for path in sorted(_APP_ROOT.rglob("*.py")) if _APP_ROOT.is_dir() else ():
        for name, names in _class_bases(path).items():
            bases.setdefault(name, set()).update(names)
    return bases


def _runtime_classes(files: list[Path]) -> frozenset[str]:
    """Return every class name that reaches a runtime base, directly or through another class.

    Matched by bare name, so two unrelated classes sharing a name are both exempt.
    """
    bases = {name: set(names) for name, names in _app_class_bases().items()}
    for path in files:
        for name, names in _class_bases(path).items():
            bases.setdefault(name, set()).update(names)
    runtime = set(_RUNTIME_BASES)
    grew = True
    while grew:
        reached = {name for name, names in bases.items() if names & runtime}
        grew = not reached <= runtime
        runtime |= reached
    return frozenset(runtime)


def _is_runtime_docstring(node: Documented, runtime_classes: frozenset[str]) -> bool:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return any(_is_runtime_decorator(d) for d in node.decorator_list)
    if isinstance(node, ast.ClassDef):
        return any(_last_name(b) in runtime_classes for b in node.bases)
    return False


def _args_entries(doc: str) -> Iterator[tuple[str, str | None, str]]:
    section = _ARGS_SECTION.search(doc + "\n")
    if not section:
        return
    for line in section.group(1).splitlines():
        entry = _ARGS_ENTRY.match(line)
        if entry:
            yield entry.group(1).lstrip("*"), entry.group(2), entry.group(3)


Finding = tuple[str, str, str]


def _shape_findings(doc: str, node: Documented, name: str, is_test_file: bool) -> list[Finding]:
    n_lines = doc.count("\n") + 1
    if isinstance(node, ast.Module):
        cap = MODULE_MAX_LINES
    elif isinstance(node, ast.ClassDef):
        cap = CLASS_MAX_LINES
    else:
        cap = FUNCTION_MAX_LINES
    is_test_function = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and name.startswith(
        "test_"
    )
    checks: list[tuple[bool, Finding]] = [
        (
            n_lines > cap,
            (
                "DS1",
                f"docstring of `{name}` is {n_lines} lines (max {cap})",
                "keep the summary and the one non-obvious constraint; the why belongs in the PR",
            ),
        ),
        (
            "`" in doc,
            (
                "DS2",
                f"backticks in the docstring of `{name}`",
                "plain text — nothing renders markup here",
            ),
        ),
        (
            bool(_RST.search(doc)),
            ("DS3", f"RST/Sphinx markup in the docstring of `{name}`", "Google style, plain text"),
        ),
        (
            is_test_file and is_test_function and n_lines > 1,
            (
                "DS4",
                f"test docstring of `{name}` is {n_lines} lines",
                "one line or none — the test name is the doc",
            ),
        ),
        (
            bool(_EXAMPLES.search(doc)),
            (
                "DS8",
                f"Examples section in the docstring of `{name}`",
                "delete it — the call sites are the examples",
            ),
        ),
    ]
    return [finding for failed, finding in checks if failed]


def _restatement_findings(doc: str, name: str) -> list[Finding]:
    findings: list[Finding] = []
    summary_words = _words(doc.split("\n", 1)[0])
    if summary_words and summary_words <= _words(name):
        findings.append(
            (
                "DS5",
                f"summary of `{name}` only restates its name",
                "delete it, or say what the name does not",
            )
        )
    for arg, typ, desc in _args_entries(doc):
        desc_words = _words(desc)
        if desc_words and desc_words <= _words(arg):
            findings.append(
                ("DS6", f"Args entry `{arg}` in `{name}` only restates its name", "drop the entry")
            )
        if typ:
            findings.append(
                (
                    "DS7",
                    f"type written in Args entry `{arg}` of `{name}`",
                    "the signature carries the type",
                )
            )
    return findings


def _check_node(
    path: Path, node: Documented, is_test_file: bool, runtime_classes: frozenset[str]
) -> list[Violation]:
    doc = ast.get_docstring(node, clean=True)
    if not doc or _is_runtime_docstring(node, runtime_classes):
        return []
    line = 1 if isinstance(node, ast.Module) else node.body[0].lineno
    name = getattr(node, "name", path.stem)
    findings = [*_shape_findings(doc, node, name, is_test_file), *_restatement_findings(doc, name)]
    return [
        Violation(path=path, line=line, detail=f"{code}: {detail}", fix=fix)
        for code, detail, fix in findings
    ]


def check(files: list[Path]) -> list[Violation]:
    """Return docstring-content violations across ``files``."""
    runtime_classes = _runtime_classes(files)
    violations: list[Violation] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        is_test_file = "tests" in path.parts or path.name.startswith("test_")
        for node in ast.walk(tree):
            if isinstance(node, Documented):
                violations.extend(_check_node(path, node, is_test_file, runtime_classes))
    return violations
