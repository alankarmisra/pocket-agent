"""Unit tests for utils/formatter.py."""

from utils.formatter import to_csv


def test_empty_input():
    assert to_csv([]) == ""


def test_single_record():
    result = to_csv([{"name": "Alice", "age": 30}])
    lines = result.strip().splitlines()
    assert lines[0] == "name,age"
    assert lines[1] == "Alice,30"


def test_missing_fields_become_empty():
    records = [{"a": 1, "b": 2}, {"a": 3}]
    result = to_csv(records)
    lines = result.strip().splitlines()
    assert lines[0] == "a,b"
    assert lines[2] == "3,"


def test_header_order_follows_first_seen():
    records = [{"z": 1, "a": 2}]
    result = to_csv(records)
    assert result.startswith("z,a")
