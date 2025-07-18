"""Core unit tests for django-yoloquery."""

import pytest
from datetime import date, datetime
from django.test import TestCase
from django.db import models

from django_yoloquery import (
    FilterSpec,
    IntentSpec,
    YoloQuerySet,
    DummyLLM,
    build_schema_for_model,
    _coerce_value,
    AIQueryModelMismatchError,
    AIQueryLLMError,
    IntentCompiler,
)
from tests.testapp.models import Author, Book, Publisher, Review, ManualModel


class TestFilterSpec:
    """Test FilterSpec data class."""

    def test_filter_spec_creation(self):
        spec = FilterSpec(path="name", op="iexact", value="John")
        assert spec.path == "name"
        assert spec.op == "iexact"
        assert spec.value == "John"

    def test_filter_spec_defaults(self):
        spec = FilterSpec(path="name")
        assert spec.path == "name"
        assert spec.op == "iexact"
        assert spec.value is None


class TestIntentSpec:
    """Test IntentSpec data class and JSON parsing."""

    def test_intent_spec_creation(self):
        filters = [FilterSpec(path="name", op="iexact", value="John")]
        spec = IntentSpec(status="ok", message="", logic="and", filters=filters, order_by=["name"], limit=10)
        assert spec.status == "ok"
        assert len(spec.filters) == 1
        assert spec.order_by == ["name"]
        assert spec.limit == 10

    def test_intent_spec_from_json_dict(self):
        data = {
            "status": "ok",
            "filters": [
                {"path": "name", "op": "iexact", "value": "John"},
                {"path": "country", "op": "in", "value": ["USA", "UK"]},
            ],
            "order_by": ["name", "-created_at"],
            "limit": 5,
        }
        spec = IntentSpec.from_json_dict(data)
        assert spec.status == "ok"
        assert len(spec.filters) == 2
        assert spec.filters[0].path == "name"
        assert spec.filters[1].value == ["USA", "UK"]
        assert spec.order_by == ["name", "-created_at"]
        assert spec.limit == 5

    def test_intent_spec_from_json_dict_minimal(self):
        data = {"status": "error", "message": "Cannot parse query"}
        spec = IntentSpec.from_json_dict(data)
        assert spec.status == "error"
        assert spec.message == "Cannot parse query"
        assert len(spec.filters) == 0

    def test_intent_spec_from_json_dict_empty(self):
        spec = IntentSpec.from_json_dict({})
        assert spec.status == "ok"
        assert len(spec.filters) == 0


class TestValueCoercion:
    """Test _coerce_value function."""

    def test_coerce_char_field(self):
        field = models.CharField(max_length=100)
        assert _coerce_value(field, "test") == "test"
        assert _coerce_value(field, 123) == "123"
        assert _coerce_value(field, ["a", "b"]) == ["a", "b"]

    def test_coerce_integer_field(self):
        field = models.IntegerField()
        assert _coerce_value(field, "123") == 123
        assert _coerce_value(field, 123) == 123
        assert _coerce_value(field, ["1", "2"]) == [1, 2]

    def test_coerce_boolean_field(self):
        field = models.BooleanField()
        assert _coerce_value(field, True) is True
        assert _coerce_value(field, "true") is True
        assert _coerce_value(field, "false") is False
        assert _coerce_value(field, "1") is True
        assert _coerce_value(field, "0") is False

    def test_coerce_date_field(self):
        field = models.DateField()
        result = _coerce_value(field, "2023-01-15")
        assert result == date(2023, 1, 15)

        # Test timestamp
        result = _coerce_value(field, 1642204800)  # 2022-01-15
        assert isinstance(result, date)

    def test_coerce_datetime_field(self):
        field = models.DateTimeField()
        result = _coerce_value(field, "2023-01-15T10:30:00")
        assert isinstance(result, datetime)
        assert result.year == 2023

    def test_coerce_none_value(self):
        field = models.CharField(max_length=100)
        assert _coerce_value(field, None) is None


@pytest.mark.django_db
class TestSchemaGeneration(TestCase):
    """Test schema generation functionality."""

    def test_build_schema_for_author(self):
        schema = build_schema_for_model(Author, depth=1, include_reverse=True)

        assert schema["root_model"] == "Author"
        assert schema["app_label"] == "testapp"
        assert "models" in schema
        assert "Author" in schema["models"]

        author_model = schema["models"]["Author"]
        assert "fields" in author_model
        assert "relationships" in author_model

        # Check fields
        fields = author_model["fields"]
        assert "name" in fields
        assert fields["name"]["type"] == "CharField"
        assert "country" in fields
        assert "birth_date" in fields
        assert "email" in fields
        assert fields["email"]["type"] == "EmailField"
        assert "is_active" in fields
        assert fields["is_active"]["type"] == "BooleanField"

    def test_build_schema_with_relationships(self):
        schema = build_schema_for_model(Book, depth=2, include_reverse=True)

        book_model = schema["models"]["Book"]
        relationships = book_model["relationships"]

        # Check forward relationships
        assert "authors" in relationships
        assert relationships["authors"]["type"] == "M2M"
        assert relationships["authors"]["to"] == "Author"

        assert "publisher" in relationships
        assert relationships["publisher"]["type"] == "FK"
        assert relationships["publisher"]["to"] == "Publisher"

        # Should include related models
        assert "Author" in schema["models"]
        assert "Publisher" in schema["models"]

    def test_build_schema_with_reverse_relationships(self):
        schema = build_schema_for_model(Author, depth=2, include_reverse=True)

        author_model = schema["models"]["Author"]
        relationships = author_model["relationships"]

        # Should have reverse relationship to books
        assert "books" in relationships
        assert relationships["books"]["type"] == "REV"

    def test_build_schema_without_reverse_relationships(self):
        schema = build_schema_for_model(Author, depth=1, include_reverse=False)

        author_model = schema["models"]["Author"]
        relationships = author_model["relationships"]

        # Should not have reverse relationships
        assert "books" not in relationships

    def test_build_schema_depth_zero(self):
        schema = build_schema_for_model(Book, depth=0, include_reverse=False)

        # Should only have the root model
        assert len(schema["models"]) == 1
        assert "Book" in schema["models"]
        assert "Author" not in schema["models"]
        assert "Publisher" not in schema["models"]


@pytest.mark.django_db
class TestIntentCompiler(TestCase):
    """Test IntentCompiler functionality."""

    def setUp(self):
        self.compiler = IntentCompiler(Author)

    def test_compile_simple_filter(self):
        intent = IntentSpec(status="ok", filters=[FilterSpec(path="name", op="iexact", value="John")])
        q = self.compiler.compile(intent)
        assert str(q) == "(AND: ('name__iexact', 'John'))"

    def test_compile_multiple_filters_and(self):
        intent = IntentSpec(
            status="ok",
            logic="and",
            filters=[
                FilterSpec(path="name", op="iexact", value="John"),
                FilterSpec(path="country", op="iexact", value="USA"),
            ],
        )
        q = self.compiler.compile(intent)
        # Should combine with AND
        assert "name__iexact" in str(q)
        assert "country__iexact" in str(q)
        assert "AND" in str(q)

    def test_compile_multiple_filters_or(self):
        intent = IntentSpec(
            status="ok",
            logic="or",
            filters=[
                FilterSpec(path="name", op="iexact", value="John"),
                FilterSpec(path="name", op="iexact", value="Jane"),
            ],
        )
        q = self.compiler.compile(intent)
        # Should combine with OR
        assert "OR" in str(q)

    def test_compile_error_status(self):
        intent = IntentSpec(status="error", message="Cannot parse")
        with pytest.raises(AIQueryLLMError):
            self.compiler.compile(intent)

    def test_compile_invalid_field(self):
        intent = IntentSpec(status="ok", filters=[FilterSpec(path="nonexistent_field", op="iexact", value="test")])
        with pytest.raises(AIQueryModelMismatchError):
            self.compiler.compile(intent)

    def test_compile_relationship_path(self):
        compiler = IntentCompiler(Book)
        intent = IntentSpec(status="ok", filters=[FilterSpec(path="authors__name", op="iexact", value="John")])
        q = compiler.compile(intent)
        assert "authors__name__iexact" in str(q)

    def test_compile_in_operator(self):
        intent = IntentSpec(status="ok", filters=[FilterSpec(path="country", op="in", value=["USA", "UK"])])
        q = self.compiler.compile(intent)
        assert "country__in" in str(q)

    def test_compile_isnull_operator(self):
        intent = IntentSpec(status="ok", filters=[FilterSpec(path="birth_date", op="isnull", value=True)])
        q = self.compiler.compile(intent)
        assert "birth_date__isnull" in str(q)


@pytest.mark.django_db
class TestYoloQuerySet(TestCase):
    """Test YoloQuerySet functionality."""

    def setUp(self):
        # Create test data
        self.author1 = Author.objects.create(
            name="John Doe", country="USA", birth_date=date(1980, 1, 1), is_active=True
        )
        self.author2 = Author.objects.create(
            name="Jane Smith", country="UK", birth_date=date(1985, 5, 15), is_active=True
        )
        self.author3 = Author.objects.create(name="Bob Johnson", country="USA", is_active=False)

    def test_ai_query_with_dummy_llm(self):
        dummy_llm = DummyLLM(
            {"authors named John": {"status": "ok", "filters": [{"path": "name", "op": "icontains", "value": "John"}]}}
        )

        qs = Author.objects.ai_query("authors named John", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 2  # John Doe and Bob Johnson
        names = [author.name for author in results]
        assert "John Doe" in names
        assert "Bob Johnson" in names

    def test_ai_query_exact_match(self):
        dummy_llm = DummyLLM(
            {
                "author named John Doe": {
                    "status": "ok",
                    "filters": [{"path": "name", "op": "iexact", "value": "John Doe"}],
                }
            }
        )

        qs = Author.objects.ai_query("author named John Doe", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 1
        assert results[0].name == "John Doe"

    def test_ai_query_with_ordering(self):
        dummy_llm = DummyLLM({"all authors by name": {"status": "ok", "filters": [], "order_by": ["name"]}})

        qs = Author.objects.ai_query("all authors by name", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 3
        assert results[0].name == "Bob Johnson"
        assert results[1].name == "Jane Smith"
        assert results[2].name == "John Doe"

    def test_ai_query_with_limit(self):
        dummy_llm = DummyLLM({"first 2 authors": {"status": "ok", "filters": [], "limit": 2}})

        qs = Author.objects.ai_query("first 2 authors", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 2

    def test_ai_query_error_handling(self):
        dummy_llm = DummyLLM({"invalid query": {"status": "error", "message": "Cannot parse this query"}})

        qs = Author.objects.ai_query("invalid query", llm=dummy_llm)

        # Should return empty queryset with error attached
        assert qs.count() == 0
        assert hasattr(qs, "ai_error")
        assert isinstance(qs.ai_error, AIQueryLLMError)
        assert "Cannot parse this query" in str(qs.ai_error)

    def test_ai_query_raise_errors(self):
        dummy_llm = DummyLLM({"invalid query": {"status": "error", "message": "Cannot parse this query"}})

        with pytest.raises(AIQueryLLMError):
            Author.objects.ai_query("invalid query", llm=dummy_llm, raise_errors=True)


@pytest.mark.django_db
class TestYoloManager(TestCase):
    """Test YoloManager functionality."""

    def test_yolo_manager_installation(self):
        # Test that YoloManager is installed on models via auto-install
        assert hasattr(Author, "objects")
        assert hasattr(Author.objects, "ai_query")

        # Test that original manager is preserved
        assert hasattr(Author, "_orig_objects")

        # Test yolo alias
        assert hasattr(Author, "yolo")
        assert hasattr(Author.yolo, "ai_query")

    def test_manual_manager_installation(self):
        # ManualModel should have YoloManager manually installed
        assert hasattr(ManualModel, "objects")
        assert hasattr(ManualModel.objects, "ai_query")

    def test_manager_delegates_to_queryset(self):
        dummy_llm = DummyLLM({"test query": {"status": "ok", "filters": []}})

        # Should be able to call ai_query on manager
        qs = Author.objects.ai_query("test query", llm=dummy_llm)
        assert isinstance(qs, YoloQuerySet)


@pytest.mark.django_db
class TestComplexQueries(TestCase):
    """Test complex query scenarios."""

    def setUp(self):
        # Create more complex test data
        self.publisher = Publisher.objects.create(name="Test Publisher", country="USA", founded_year=1990)

        self.author1 = Author.objects.create(name="John Doe", country="USA", birth_date=date(1980, 1, 1))

        self.author2 = Author.objects.create(name="Jane Smith", country="UK", birth_date=date(1985, 5, 15))

        self.book = Book.objects.create(
            title="Test Book",
            isbn="1234567890123",
            publisher=self.publisher,
            published_date=date(2023, 1, 1),
            page_count=300,
            is_published=True,
        )
        self.book.authors.add(self.author1, self.author2)

        self.review = Review.objects.create(
            book=self.book, reviewer_name="Test Reviewer", rating=5, comment="Great book!"
        )

    def test_relationship_query(self):
        dummy_llm = DummyLLM(
            {
                "books by John Doe": {
                    "status": "ok",
                    "filters": [{"path": "authors__name", "op": "iexact", "value": "John Doe"}],
                }
            }
        )

        qs = Book.objects.ai_query("books by John Doe", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 1
        assert results[0].title == "Test Book"

    def test_reverse_relationship_query(self):
        dummy_llm = DummyLLM(
            {
                "authors with published books": {
                    "status": "ok",
                    "filters": [{"path": "books__is_published", "op": "exact", "value": True}],
                }
            }
        )

        qs = Author.objects.ai_query("authors with published books", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 2  # Both authors have the published book

    def test_nested_relationship_query(self):
        dummy_llm = DummyLLM(
            {
                "books by American authors": {
                    "status": "ok",
                    "filters": [{"path": "authors__country", "op": "iexact", "value": "USA"}],
                }
            }
        )

        qs = Book.objects.ai_query("books by American authors", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 1
        assert results[0].title == "Test Book"
