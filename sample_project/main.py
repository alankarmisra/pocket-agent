"""Entry point for the JSON-to-CSV converter tool."""

from utils.parser import parse_json
from utils.formatter import to_csv
from utils.validator import validate_schema
import sys


def main(input_path: str, output_path: str, schema_path: str | None = None) -> None:
    with open(input_path) as f:
        raw = f.read()

    records = parse_json(raw)

    if schema_path:
        with open(schema_path) as f:
            schema = f.read()
        validate_schema(records, schema)

    csv_text = to_csv(records)

    with open(output_path, "w") as f:
        f.write(csv_text)

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
