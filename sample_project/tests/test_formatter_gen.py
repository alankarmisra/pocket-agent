import pytest
from utils.formatter import to_csv


class TestToCsv:
    def test_happy_path(self):
        records = [
            {"name": "Alice", "age": "30", "city": "NYC"},
            {"name": "Bob", "age": "25", "city": "LA"},
        ]
        result = to_csv(records)
        assert "name,age,city\r\n" in result
        assert "Alice,30,NYC\r\n" in result
        assert "Bob,25,LA\r\n" in result

    def test_empty_list(self):
        assert to_csv([]) == ""

    def test_single_record(self):
        records = [{"id": 1, "value": "test"}]
        result = to_csv(records)
        assert "id,value\r\n" in result
        assert "1,test\r\n" in result

    def test_missing_values_filled_with_empty_string(self):
        records = [
            {"a": "1", "b": "2"},
            {"a": "3"},  # missing 'b'
        ]
        result = to_csv(records)
        assert "a,b\r\n" in result
        assert "1,2\r\n" in result
        assert "3,\r\n" in result

    def test_custom_delimiter(self):
        records = [{"x": "1", "y": "2"}]
        result = to_csv(records, delimiter=";")
        assert "x;y\r\n" in result
        assert "1;2\r\n" in result

    def test_headers_ordered_by_first_seeing(self):
        records = [
            {"z": 1, "a": 2},
            {"b": 3, "a": 4},  # 'b' appears first in second record but 'a' was seen first
        ]
        result = to_csv(records)
        lines = result.strip().split("\r\n")
        headers = lines[0].split(",")
        assert headers == ["z", "a", "b"]

    def test_non_string_values_converted(self):
        records = [{"num": 42, "flt": 3.14, "bool": True, "none_val": None}]
        result = to_csv(records)
        assert "num,flt,bool,none_val\r\n" in result
        assert "42,3.14,True,\r\n" in result

    def test_records_with_different_keys(self):
        records = [
            {"only_a": "first"},
            {"only_b": "second"},
            {"a": "third", "b": "fourth"},
        ]
        result = to_csv(records)
        lines = result.strip().split("\r\n")
        assert lines[0] == "only_a,only_b,a,b"
        assert lines[1] == "first,,,"
        assert lines[2] == ",second,,"
        assert lines[3] == ",,third,fourth"

    def test_records_preserves_header_order_for_subsequent_records(self):
        records = [
            {"first": "1", "second": "2"},
            {"third": "3", "first": "4"},  # 'third' appears before 'first' in second record
        ]
        result = to_csv(records)
        lines = result.strip().split("\r\n")
        assert lines[0] == "first,second,third"
        assert lines[1] == "1,2,"
        assert lines[2] == "4,,3"
