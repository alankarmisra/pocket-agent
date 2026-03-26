"""Unit tests for utils/parser.py."""

import pytest
from utils.parser import parse_json, _flatten


class TestParseJson:
    def test_list_of_dicts(self):
        result = parse_json('[{"a": 1}, {"a": 2}]')
        assert result == [{"a": 1}, {"a": 2}]

    def test_single_dict_wrapped(self):
        result = parse_json('{"x": 10}')
        assert result == [{"x": 10}]

    def test_list_of_scalars(self):
        result = parse_json('[1, 2, 3]')
        assert result == [{"value": 1}, {"value": 2}, {"value": 3}]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_json("{not valid}")

    def test_unexpected_type_raises(self):
        with pytest.raises(ValueError, match="Expected"):
            parse_json('"just a string"')


class TestFlatten:
    def test_nested_dict(self):
        assert _flatten({"a": {"b": 1}}) == {"a.b": 1}

    def test_deeply_nested(self):
        assert _flatten({"a": {"b": {"c": 42}}}) == {"a.b.c": 42}

    def test_flat_dict_unchanged(self):
        assert _flatten({"x": 1, "y": 2}) == {"x": 1, "y": 2}
