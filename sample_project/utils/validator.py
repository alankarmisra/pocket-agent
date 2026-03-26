"""Validate a list of records against a simple JSON schema."""

import json


def validate_schema(records: list[dict], schema_raw: str) -> None:
    """
    Validate each record against *schema_raw* (a JSON object mapping
    field names to expected types: "string", "number", "boolean").

    Raises TypeError on the first violation found.
    """
    schema: dict[str, str] = json.loads(schema_raw)
    type_map = {"string": str, "number": (int, float), "boolean": bool}

    for i, record in enumerate(records):
        for field, expected_type_name in schema.items():
            if field not in record:
                raise TypeError(
                    f"Record {i}: missing required field '{field}'"
                )
            expected = type_map.get(expected_type_name)
            if expected and not isinstance(record[field], expected):
                raise TypeError(
                    f"Record {i}: field '{field}' expected {expected_type_name}, "
                    f"got {type(record[field]).__name__}"
                )
