"""Format a list of flat dicts as CSV text."""

import csv
import io


def to_csv(records: list[dict], delimiter: str = ",") -> str:
    """
    Convert *records* (list of flat dicts) to a CSV string.

    All keys across all records are used as headers.
    Missing values are rendered as empty strings.
    """
    if not records:
        return ""

    # Collect all headers preserving first-seen order
    headers: list[str] = []
    seen: set[str] = set()
    for rec in records:
        for key in rec:
            if key not in seen:
                headers.append(key)
                seen.add(key)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers,
                            delimiter=delimiter, extrasaction="ignore")
    writer.writeheader()
    for rec in records:
        writer.writerow({h: rec.get(h, "") for h in headers})

    return buf.getvalue()
