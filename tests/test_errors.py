"""Error handling and edge case tests for django-yoloquery."""

import pytest
from datetime import date
from django.test import TestCase

from django_yoloquery import (
    DummyLLM,
    IntentCompiler,
    FilterSpec,
    IntentSpec,
    AIQueryModelMismatchError,
    AIQueryOperatorError,
    AIQueryValueError,
    AIQueryLLMError,
    _coerce_value,
)
from django.db import models
from tests.testapp.models import Author, Book, Publisher


@pytest.mark.django_db
class TestErrorHandling(TestCase):
    """Test error handling scenarios."""

    def setUp(self):
        # Create minimal test data
        self.author = Author.objects.create(name="Test Author", country="USA", birth_date=date(1980, 1, 1))

        self.publisher = Publisher.objects.create(name="Test Publisher", country="USA")

        self.book = Book.objects.create(
            title="Test Book", isbn="1234567890123", publisher=self.publisher, is_published=True
        )
        self.book.authors.add(self.author)

    def test_invalid_field_error(self):
        """Test handling of invalid field references."""
        dummy_llm = DummyLLM(
            {
                "books with invalid field": {
                    "status": "ok",
                    "filters": [{"path": "nonexistent_field", "op": "exact", "value": "test"}],
                }
            }
        )

        qs = Book.objects.ai_query("books with invalid field", llm=dummy_llm)

        assert qs.count() == 0
        assert hasattr(qs, "ai_error")
        assert isinstance(qs.ai_error, AIQueryModelMismatchError)
        assert "nonexistent_field" in str(qs.ai_error)

    def test_invalid_relationship_path_error(self):
        """Test handling of invalid relationship paths."""
        dummy_llm = DummyLLM(
            {
                "books with invalid relationship": {
                    "status": "ok",
                    "filters": [{"path": "invalid_relation__name", "op": "exact", "value": "test"}],
                }
            }
        )

        qs = Book.objects.ai_query("books with invalid relationship", llm=dummy_llm)

        assert qs.count() == 0
        assert hasattr(qs, "ai_error")
        assert isinstance(qs.ai_error, AIQueryModelMismatchError)

    def test_invalid_operator_error(self):
        """Test handling of invalid operators for field types."""
        # Try to use 'gt' on a CharField (should not be allowed by default policy)
        compiler = IntentCompiler(Author)
        intent = IntentSpec(status="ok", filters=[FilterSpec(path="name", op="regex", value="test")])

        with pytest.raises(AIQueryOperatorError):
            compiler.compile(intent)

    def test_llm_error_status(self):
        """Test handling when LLM returns error status."""
        dummy_llm = DummyLLM(
            {"ambiguous query": {"status": "error", "message": "Query is too ambiguous to parse correctly"}}
        )

        qs = Author.objects.ai_query("ambiguous query", llm=dummy_llm)

        assert qs.count() == 0
        assert hasattr(qs, "ai_error")
        assert isinstance(qs.ai_error, AIQueryLLMError)
        assert "too ambiguous" in str(qs.ai_error)

    def test_llm_error_with_raise_errors(self):
        """Test that errors are raised when raise_errors=True."""
        dummy_llm = DummyLLM({"error query": {"status": "error", "message": "Cannot process this query"}})

        with pytest.raises(AIQueryLLMError):
            Author.objects.ai_query("error query", llm=dummy_llm, raise_errors=True)

    def test_invalid_filter_value_error(self):
        """Test handling of invalid filter values."""
        dummy_llm = DummyLLM(
            {
                "invalid range": {
                    "status": "ok",
                    "filters": [{"path": "birth_date", "op": "range", "value": "not_a_list"}],
                }
            }
        )

        qs = Author.objects.ai_query("invalid range", llm=dummy_llm)

        assert qs.count() == 0
        assert hasattr(qs, "ai_error")
        assert isinstance(qs.ai_error, AIQueryValueError)

    def test_malformed_json_response(self):
        """Test handling of malformed JSON responses."""
        # This would typically be handled by the LLM client
        # but we can test IntentSpec parsing
        with pytest.raises(Exception):
            IntentSpec.from_json_dict("not a dict")

    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        dummy_llm = DummyLLM({"": {"status": "ok", "filters": []}})

        qs = Author.objects.ai_query("", llm=dummy_llm)
        results = list(qs)

        # Empty query should return all objects
        assert len(results) == 1
        assert results[0] == self.author

    def test_none_value_handling(self):
        """Test handling of None values in filters."""
        dummy_llm = DummyLLM(
            {
                "authors with no birth date": {
                    "status": "ok",
                    "filters": [{"path": "birth_date", "op": "isnull", "value": True}],
                }
            }
        )

        # Create author with no birth date
        Author.objects.create(name="No Date Author", country="USA")

        qs = Author.objects.ai_query("authors with no birth date", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 1
        assert results[0].name == "No Date Author"
        assert results[0].birth_date is None


class TestValueCoercionErrors:
    """Test error cases in value coercion."""

    def test_invalid_integer_coercion(self):
        """Test error when coercing invalid values to integer."""
        field = models.IntegerField()

        with pytest.raises(AIQueryValueError):
            _coerce_value(field, "not_a_number")

    def test_invalid_boolean_coercion(self):
        """Test error when coercing invalid values to boolean."""
        field = models.BooleanField()

        with pytest.raises(AIQueryValueError):
            _coerce_value(field, "maybe")

    def test_invalid_date_coercion(self):
        """Test error when coercing invalid values to date."""
        field = models.DateField()

        with pytest.raises(AIQueryValueError):
            _coerce_value(field, "not-a-date")

        with pytest.raises(AIQueryValueError):
            _coerce_value(field, "2023-13-01")  # Invalid month

    def test_invalid_datetime_coercion(self):
        """Test error when coercing invalid values to datetime."""
        field = models.DateTimeField()

        with pytest.raises(AIQueryValueError):
            _coerce_value(field, "not-a-datetime")


@pytest.mark.django_db
class TestEdgeCases(TestCase):
    """Test edge cases and unusual scenarios."""

    def setUp(self):
        self.author = Author.objects.create(
            name="Edge Case Author",
            country="",  # Empty string
            email="",  # Empty email
        )

    def test_empty_string_values(self):
        """Test handling of empty string values."""
        dummy_llm = DummyLLM(
            {
                "authors with empty country": {
                    "status": "ok",
                    "filters": [{"path": "country", "op": "exact", "value": ""}],
                }
            }
        )

        qs = Author.objects.ai_query("authors with empty country", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 1
        assert results[0].country == ""

    def test_very_long_query_text(self):
        """Test handling of very long query text."""
        long_query = "find authors " + "very " * 1000 + "specific"

        dummy_llm = DummyLLM({long_query: {"status": "ok", "filters": []}})

        qs = Author.objects.ai_query(long_query, llm=dummy_llm)
        results = list(qs)

        # Should still work
        assert len(results) == 1

    def test_unicode_in_query(self):
        """Test handling of unicode characters in queries."""
        unicode_query = "find authors named José or François"

        dummy_llm = DummyLLM(
            {unicode_query: {"status": "ok", "filters": [{"path": "name", "op": "icontains", "value": "José"}]}}
        )

        # Create author with unicode name
        Author.objects.create(name="José García", country="Spain")

        qs = Author.objects.ai_query(unicode_query, llm=dummy_llm)
        results = list(qs)

        assert len(results) == 1
        assert "José" in results[0].name

    def test_special_characters_in_values(self):
        """Test handling of special characters in filter values."""
        dummy_llm = DummyLLM(
            {
                "books with special chars": {
                    "status": "ok",
                    "filters": [{"path": "title", "op": "icontains", "value": "O'Reilly & Sons"}],
                }
            }
        )

        # This should not crash
        qs = Book.objects.ai_query("books with special chars", llm=dummy_llm)
        list(qs)  # Force evaluation

    def test_large_limit_value(self):
        """Test handling of very large limit values."""
        dummy_llm = DummyLLM({"all authors": {"status": "ok", "filters": [], "limit": 999999}})

        qs = Author.objects.ai_query("all authors", llm=dummy_llm)
        results = list(qs)

        # Should work but be limited by actual data
        assert len(results) == 1

    def test_zero_limit_value(self):
        """Test handling of zero limit value."""
        dummy_llm = DummyLLM({"no authors": {"status": "ok", "filters": [], "limit": 0}})

        qs = Author.objects.ai_query("no authors", llm=dummy_llm)
        results = list(qs)

        # Limit of 0 should return empty result
        assert len(results) == 0

    def test_negative_limit_value(self):
        """Test handling of negative limit value."""
        dummy_llm = DummyLLM({"negative limit": {"status": "ok", "filters": [], "limit": -5}})

        qs = Author.objects.ai_query("negative limit", llm=dummy_llm)
        results = list(qs)

        # Negative limit should be ignored (no limit applied)
        assert len(results) == 1

    def test_multiple_ordering_fields(self):
        """Test handling of multiple ordering fields."""
        # Create multiple authors for meaningful ordering
        Author.objects.create(name="Alpha", country="USA", birth_date=date(1990, 1, 1))
        Author.objects.create(name="Beta", country="USA", birth_date=date(1980, 1, 1))

        dummy_llm = DummyLLM(
            {"authors ordered by country then name": {"status": "ok", "filters": [], "order_by": ["country", "name"]}}
        )

        qs = Author.objects.ai_query("authors ordered by country then name", llm=dummy_llm)
        results = list(qs)

        assert len(results) >= 2

    def test_invalid_ordering_field(self):
        """Test handling of invalid ordering field."""
        dummy_llm = DummyLLM(
            {"authors with invalid order": {"status": "ok", "filters": [], "order_by": ["nonexistent_field"]}}
        )

        # This should return a QuerySet with an error attached
        qs = Author.objects.ai_query("authors with invalid order", llm=dummy_llm)

        # Should have error attached instead of raising
        assert qs.count() == 0
        assert hasattr(qs, "ai_error")
        assert isinstance(qs.ai_error, AIQueryModelMismatchError)
        assert "Invalid ordering field" in str(qs.ai_error)


@pytest.mark.django_db
class TestComplexErrorScenarios(TestCase):
    """Test complex error scenarios that might occur in real usage."""

    def test_circular_relationship_in_schema(self):
        """Test that circular relationships in schema don't cause infinite loops."""
        # Our test models have circular relationships (Author -> Book -> Author)
        # This should not cause issues in schema generation
        from django_yoloquery import build_schema_for_model

        # This should complete without infinite recursion
        schema = build_schema_for_model(Author, depth=3, include_reverse=True)
        assert "Author" in schema["models"]
        assert "Book" in schema["models"]

    def test_deeply_nested_relationship_query(self):
        """Test deeply nested relationship queries."""
        # Create test data with relationships
        publisher = Publisher.objects.create(name="Test Pub", country="USA")
        author = Author.objects.create(name="Test Author", country="USA")
        book = Book.objects.create(title="Test Book", isbn="1234567890123", publisher=publisher, is_published=True)
        book.authors.add(author)

        dummy_llm = DummyLLM(
            {
                "authors of books by publishers from USA": {
                    "status": "ok",
                    "filters": [{"path": "books__publisher__country", "op": "iexact", "value": "USA"}],
                }
            }
        )

        qs = Author.objects.ai_query("authors of books by publishers from USA", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 1
        assert results[0].name == "Test Author"

    def test_mixed_valid_and_invalid_filters(self):
        """Test queries with mix of valid and invalid filters."""
        dummy_llm = DummyLLM(
            {
                "mixed filters": {
                    "status": "ok",
                    "logic": "and",
                    "filters": [
                        {"path": "name", "op": "iexact", "value": "Test"},  # Valid
                        {"path": "invalid_field", "op": "exact", "value": "test"},  # Invalid
                    ],
                }
            }
        )

        qs = Author.objects.ai_query("mixed filters", llm=dummy_llm)

        # Should fail on the first invalid filter
        assert qs.count() == 0
        assert hasattr(qs, "ai_error")
        assert isinstance(qs.ai_error, AIQueryModelMismatchError)

    def test_filter_with_none_path(self):
        """Test handling of filter with None path."""
        dummy_llm = DummyLLM(
            {"none path": {"status": "ok", "filters": [{"path": None, "op": "exact", "value": "test"}]}}
        )

        qs = Author.objects.ai_query("none path", llm=dummy_llm)

        # Should handle gracefully
        assert qs.count() == 0
        assert hasattr(qs, "ai_error")

    def test_filter_with_empty_path(self):
        """Test handling of filter with empty path."""
        dummy_llm = DummyLLM(
            {"empty path": {"status": "ok", "filters": [{"path": "", "op": "exact", "value": "test"}]}}
        )

        qs = Author.objects.ai_query("empty path", llm=dummy_llm)

        # Should handle gracefully
        assert qs.count() == 0
        assert hasattr(qs, "ai_error")
