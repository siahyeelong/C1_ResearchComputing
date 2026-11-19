#!/usr/bin/env python3
"""Check the submission format and referenced files, without running student code."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path, PurePosixPath
import sys

try:
    from jsonschema import Draft202012Validator
    from pypdf import PdfReader
except ImportError:
    raise SystemExit(
        "Install check dependencies: python -m pip install -r requirements-submission.txt"
    )

SCHEMA_PATH = Path(__file__).with_name("results.schema.json")


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"{value} is not a JSON number")

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid_constant,
    )


def draft_schema(schema):
    result = copy.deepcopy(schema)
    if "properties" in result:
        result["properties"] = {
            k: draft_schema(v) for k, v in result["properties"].items()
        }
    if "items" in result:
        result["items"] = draft_schema(result["items"])
    if result.pop("x-answer", False):
        return {"anyOf": [result, {"type": "null"}]}
    return result


def label(parts):
    return ".".join(map(str, parts)) or "$"


def walk(data, schema, parts=()):
    yield parts, data, schema
    if isinstance(data, dict):
        for key, child in schema.get("properties", {}).items():
            if key in data:
                yield from walk(data[key], child, (*parts, key))
    elif isinstance(data, list) and "items" in schema:
        for index, child in enumerate(data):
            yield from walk(child, schema["items"], (*parts, index))


def validate(data, *, root=None, draft=False):
    """Return format errors. A non-None root also enables artifact existence checks."""
    schema = load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(draft_schema(schema) if draft else schema)
    errors = [
        f"{label(error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(data), key=lambda e: label(e.absolute_path)
        )
    ]
    root = Path(root).resolve() if root is not None else None
    for parts, value, node in walk(data, schema):
        location = label(parts)
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"{location}: use a finite number")
        if node.get("format") != "relative-path" or not isinstance(value, str):
            continue
        path = PurePosixPath(value)
        if (
            not value.strip()
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or ":" in value
            or value.startswith("~")
            or any(ord(c) < 32 for c in value)
        ):
            errors.append(f"{location}: use a relative path within the project")
            continue
        if node.get("x-extensions") and path.suffix.lower() not in node["x-extensions"]:
            errors.append(f"{location}: expected one of {node['x-extensions']}")
        if root is not None and not draft:
            try:
                resolved = (root / value).resolve()
                if not resolved.is_relative_to(root):
                    errors.append(f"{location}: path resolves outside the project")
                elif not resolved.is_file():
                    errors.append(f"{location}: file not found: {value}")
                elif resolved.stat().st_size == 0:
                    errors.append(f"{location}: file is empty: {value}")
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"{location}: cannot inspect path ({exc})")
    return errors


def validate_file(path, *, root, draft=False):
    path, root = Path(path), Path(root).resolve()
    try:
        if not path.resolve().is_relative_to(root):
            return None, [f"{path}: results file resolves outside the project"]
        data = load_json(path)
    except (OSError, ValueError) as exc:
        return None, [f"{path}: {exc}"]
    errors = validate(data, root=root, draft=draft)
    if not draft and isinstance(data, dict):
        expected = root / "results/results.json"
        if path.resolve() != expected.resolve():
            errors.append(
                "results: save this file as results/results.json in your own project"
            )
        readme = root / "README.md"
        try:
            if (
                not readme.resolve().is_relative_to(root)
                or not readme.is_file()
                or not readme.stat().st_size
            ):
                errors.append(
                    "README.md: a non-empty file within the project is required"
                )
        except (OSError, RuntimeError):
            errors.append("README.md: cannot read file")
        # Only open the report after validating its path and file existence.
        if isinstance(data.get("report_path"), str) and not any(
            e.startswith("report_path:") for e in errors
        ):
            try:
                reader = PdfReader(root / data["report_path"])
                if reader.is_encrypted:
                    errors.append("report_path: submit an unencrypted PDF")
                elif not reader.pages:
                    errors.append("report_path: PDF has no pages")
            except Exception as exc:
                errors.append(f"report_path: cannot read PDF ({exc})")
    return data, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        nargs="?",
        help="Results file (default: results/results.json under the project root).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: current directory).",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Allow null answers and skip file-existence checks.",
    )
    args = parser.parse_args()
    paths = [args.results] if args.results else [args.root / "results/results.json"]
    failed = False
    for path in paths:
        _, errors = validate_file(path, root=args.root, draft=args.draft)
        if errors:
            failed = True
            print(f"FAIL: {path}", file=sys.stderr)
            for error in errors[:30]:
                print(f"- {error}", file=sys.stderr)
            if len(errors) > 30:
                print(f"... and {len(errors) - 30} further errors", file=sys.stderr)
        else:
            print(
                f"PASS: {path}"
                + (
                    " (draft; files not checked)"
                    if args.draft
                    else " (format, report PDF and files)"
                )
            )
    if not failed:
        print("Numerical correctness and report content have not been assessed.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
