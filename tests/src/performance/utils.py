
import pytest
from src.performance.utils import detect_model_encoding, get_encoding_from_model_name, count_tokens, terminal_http_code, split_list_into_variable_parts

class MockEncoding:
    def encode(self, string):
        return list(string)

@pytest.fixture
def mock_encoding():
    return MockEncoding()

@pytest.mark.parametrize("model_name,expected", [
    ("gpt-4", ("gpt-4", "cl100k_base")),
    ("gpt-3.5-turbo-", ("gpt-3.5-turbo-", "cl100k_base")),
    ("unknown-model", ("gpt-4", "cl100k_base")),  # Assuming fuzzy matching defaults to "gpt-4"
])
def test_detect_model_encoding(model_name, expected):
    assert detect_model_encoding(model_name) == expected

def test_get_encoding_from_model_name_valid(mock_encoding):
    encoding = get_encoding_from_model_name("gpt-4")
    assert isinstance(encoding, MockEncoding)

def test_get_encoding_from_model_name_none(mock_encoding):
    encoding = get_encoding_from_model_name(None)
    assert isinstance(encoding, MockEncoding)

# Tests for count_tokens
def test_count_tokens_valid(mock_encoding):
    assert count_tokens(mock_encoding, "test string") == 11

def test_count_tokens_empty(mock_encoding):
    assert count_tokens(mock_encoding, "") == 0

# Tests for terminal_http_code
@pytest.mark.parametrize("status_code,expected", [
    (400, True),
    (499, True),
    (500, False),
    (200, False),
])
def test_terminal_http_code(status_code, expected):
    class MockException:
        def __init__(self, status):
            self.status = status
    assert terminal_http_code(MockException(status_code)) == expected

@pytest.mark.parametrize("input_list,expected_output", [
    ([], ([],)),
    ([1], ([1],)),
    ([1, 2], ([1], [2])),
    ([1, 2, 3], ([1], [2], [3])),
    ([1, 2, 3, 4], ([1], [2], [3], [4])),
    ([1, 2, 3, 4, 5], ([1, 2], [3], [4], [5])),
    (list(range(1, 21)), (list(range(1, 6)), list(range(6, 11)), list(range(11, 16)), list(range(16, 21)))),
])
def test_split_list_into_variable_parts(input_list, expected_output):
    assert split_list_into_variable_parts(input_list) == expected_output