import pytest
from utils.formatter import to_csv


class TestToCsv:
    def test_basic_conversion(self):
        records = [
            {"name": "Alice", "age": "30", "city": "NYC"},
            {"name": "Bob", "age": "25", "city": "LA"},
        ]
        result = to_csv(records)
        assert result == "name,age,city\r\nAlice,30,NYC\r\nBob,25,LA\r\n"

    def test_empty_records(self):
        assert to_csv([]) == ""

    def test_missing_values(self):
        records = [
            {"name": "Alice", "city": "NYC"},
            {"name": "Bob"},
            {"city": "Chicago"},
        ]
        result = to_csv(records)
        lines = result.strip().split("\r\n")
        assert lines[0] == "name,city"
        assert lines[1] == "Alice,NYC"
        assert lines[2] == "Bob,"
        assert lines[3] == ",Chicago"

    def test_different_delimiter(self):
        records = [
            {"a": "1", "b": "2"},
            {"a": "3", "b": "4"},
        ]
        result = to_csv(records, delimiter=";")
        assert result == "a;b\r\n1;2\r\n3;4\r\n"

    def test_key_order_preservation(self):
        records = [
            {"z": 1, "a": 2},
            {"a": 3, "z": 4},
        ]
        result = to_csv(records)
        lines = result.strip().split("\r\n")
        assert lines[0] == "z,a"

    def test_extra_keys_ignored(self):
        records = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25, "extra": "ignored"},
        ]
        result = to_csv(records, fields=["name", "age"])
        lines = result.strip().split("\r\n")
        assert "extra" not in lines[0]

    def test_unicode_content(self):
        records = [
            {"name": "Café", "city": "Zürich"},
        ]
        result = to_csv(records)
        assert "Café" in result
        assert "Zürich" in result
