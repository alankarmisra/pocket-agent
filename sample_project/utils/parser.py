"""Parse a JSON string into a list of flat dicts."""

import json


def parse_json(raw: str) -> list[dict]:
    """
    Accept a JSON string that is either:
    - a list of objects   → returned as-is
    - a single object     → wrapped in a list
    - a list of scalars   → each scalar wrapped as {"value": scalar}

    Raises ValueError on malformed input.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if isinstance(data, list):
        return [_flatten(item) if isinstance(item, dict) else {"value": item}
                for item in data]
    if isinstance(data, dict):
        return [_flatten(data)]

    raise ValueError(f"Expected a JSON object or array, got {type(data).__name__}")


def _flatten(obj: dict, prefix: str = "", sep: str = ".") -> dict:
    """Recursively flatten nested dicts: {"a": {"b": 1}} → {"a.b": 1}."""
    result = {}
    for key, value in obj.items():
        full_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, full_key, sep))
        else:
            result[full_key] = value
    return result
