import pytest
from utils.formatter import to_csv


class TestToCsv:
    def test_happy_path_basic(self):
        records = [
            {"name": "Alice", "age": "30", "city": "NYC"},
            {"name": "Bob", "age": "25", "city": "LA"},
        ]
        result = to_csv(records)
        lines = result.split("\r\n")
        assert lines[0] == "name,age,city"
        assert lines[1] == "Alice,30,NYC"
        assert lines[2] == "Bob,25,LA"

    def test_happy_path_custom_delimiter(self):
        records = [{"name": "Alice", "value": "100"}]
        result = to_csv(records, delimiter=";")
        assert "name;value" in result
        assert "Alice;100" in result

    def test_edge_case_empty_records(self):
        assert to_csv([]) == ""

    def test_edge_case_empty_list_of_records(self):
        records = []
        assert to_csv(records) == ""

    def test_edge_case_missing_keys_in_records(self):
        records = [
            {"name": "Alice", "age": "30"},
            {"name": "Bob", "city": "LA"},
            {"age": "25", "city": "Chicago"},
        ]
        result = to_csv(records)
        lines = result.split("\r\n")
        assert lines[0] == "name,age,city"
        assert lines[1] == "Alice,30,"
        assert lines[2] == "Bob,,LA"
        assert lines[3] == ",25,Chicago"

    def test_edge_case_extra_keys_ignored(self):
        records = [
            {"name": "Alice", "age": "30", "extra_field": "should be ignored"},
        ]
        result = to_csv(records)
        lines = result.split("\r\n")
        assert lines[0] == "name,age,extra_field"
        assert lines[1] == "Alice,30,should be ignored"

    def test_edge_case_different_key_ordering_preserves_first_seen(self):
        records = [
            {"b": "2", "a": "1"},
            {"c": "3", "a": "4"},
        ]
        result = to_csv(records)
        lines = result.split("\r\n")
        assert lines[0] == "b,a,c"
        assert lines[1] == "2,1,"
        assert lines[2] == ",4,3"

    def test_edge_case_unicode_and_special_characters(self):
        records = [
            {"name": "José", "quote": "Hello, \"world\""},
            {"name": "中", "quote": "Line1\nLine2"},
        ]
        result = to_csv(records)
        lines = result.split("\r\n")
        assert lines[0] == "name,quote"
        assert 'José,"Hello, ""world"""' in lines[1]
        assert '中,"Line1\nLine2"' in lines[2]
