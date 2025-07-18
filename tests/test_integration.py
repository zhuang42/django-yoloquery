"""Integration tests for django-yoloquery."""

import pytest
from datetime import date
from django.test import TestCase

from django_yoloquery import DummyLLM, build_schema_for_model
from tests.testapp.models import Author, Book, Publisher, Review


@pytest.mark.django_db
class TestIntegrationScenarios(TestCase):
    """Test realistic usage scenarios."""

    def setUp(self):
        # Create comprehensive test data
        self.setup_test_data()

    def setup_test_data(self):
        # Publishers
        self.penguin = Publisher.objects.create(name="Penguin Random House", country="USA", founded_year=1927)

        self.oxford = Publisher.objects.create(name="Oxford University Press", country="UK", founded_year=1586)

        # Authors
        self.author_american = Author.objects.create(
            name="John Steinbeck", country="USA", birth_date=date(1902, 2, 27), email="john@example.com", is_active=True
        )

        self.author_british = Author.objects.create(
            name="Jane Austen", country="UK", birth_date=date(1775, 12, 16), is_active=False  # Historical author
        )

        self.author_contemporary = Author.objects.create(
            name="Margaret Atwood",
            country="Canada",
            birth_date=date(1939, 11, 18),
            email="margaret@example.com",
            is_active=True,
        )

        # Books
        self.grapes_of_wrath = Book.objects.create(
            title="The Grapes of Wrath",
            isbn="9780143039433",
            publisher=self.penguin,
            published_date=date(1939, 4, 14),
            page_count=464,
            is_published=True,
            price=12.99,
        )
        self.grapes_of_wrath.authors.add(self.author_american)

        self.pride_prejudice = Book.objects.create(
            title="Pride and Prejudice",
            isbn="9780141439518",
            publisher=self.oxford,
            published_date=date(1813, 1, 28),
            page_count=432,
            is_published=True,
            price=9.99,
        )
        self.pride_prejudice.authors.add(self.author_british)

        self.handmaids_tale = Book.objects.create(
            title="The Handmaid's Tale",
            isbn="9780385490818",
            publisher=self.penguin,
            published_date=date(1985, 8, 1),
            page_count=311,
            is_published=True,
            price=14.99,
        )
        self.handmaids_tale.authors.add(self.author_contemporary)

        self.unpublished_book = Book.objects.create(
            title="Unpublished Work", isbn="9999999999999", publisher=self.penguin, is_published=False
        )
        self.unpublished_book.authors.add(self.author_contemporary)

        # Reviews
        Review.objects.create(
            book=self.grapes_of_wrath,
            reviewer_name="Literary Critic",
            rating=5,
            comment="A masterpiece of American literature",
        )

        Review.objects.create(
            book=self.pride_prejudice,
            reviewer_name="Classic Reader",
            rating=4,
            comment="Timeless romance and social commentary",
        )

        Review.objects.create(
            book=self.handmaids_tale,
            reviewer_name="Modern Reader",
            rating=5,
            comment="Chilling and relevant dystopian fiction",
        )

    def test_find_authors_by_name(self):
        """Test finding authors by name."""
        dummy_llm = DummyLLM(
            {
                "find authors named John": {
                    "status": "ok",
                    "filters": [{"path": "name", "op": "icontains", "value": "John"}],
                }
            }
        )

        qs = Author.objects.ai_query("find authors named John", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 1
        assert results[0].name == "John Steinbeck"

    def test_find_authors_by_country(self):
        """Test finding authors by country."""
        dummy_llm = DummyLLM(
            {"American authors": {"status": "ok", "filters": [{"path": "country", "op": "iexact", "value": "USA"}]}}
        )

        qs = Author.objects.ai_query("American authors", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 1
        assert results[0].country == "USA"

    def test_find_active_authors(self):
        """Test finding active authors."""
        dummy_llm = DummyLLM(
            {"active authors": {"status": "ok", "filters": [{"path": "is_active", "op": "exact", "value": True}]}}
        )

        qs = Author.objects.ai_query("active authors", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 2  # John and Margaret
        active_names = {author.name for author in results}
        assert "John Steinbeck" in active_names
        assert "Margaret Atwood" in active_names

    def test_find_books_by_publisher(self):
        """Test finding books by publisher."""
        dummy_llm = DummyLLM(
            {
                "books published by Penguin": {
                    "status": "ok",
                    "filters": [{"path": "publisher__name", "op": "icontains", "value": "Penguin"}],
                }
            }
        )

        qs = Book.objects.ai_query("books published by Penguin", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 3  # Grapes of Wrath, Handmaid's Tale, Unpublished Work
        penguin_books = {book.title for book in results}
        assert "The Grapes of Wrath" in penguin_books
        assert "The Handmaid's Tale" in penguin_books
        assert "Unpublished Work" in penguin_books

    def test_find_published_books(self):
        """Test finding only published books."""
        dummy_llm = DummyLLM(
            {"published books": {"status": "ok", "filters": [{"path": "is_published", "op": "exact", "value": True}]}}
        )

        qs = Book.objects.ai_query("published books", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 3
        for book in results:
            assert book.is_published is True

    def test_find_books_after_date(self):
        """Test finding books published after a certain date."""
        dummy_llm = DummyLLM(
            {
                "books published after 1900": {
                    "status": "ok",
                    "filters": [{"path": "published_date", "op": "gt", "value": "1900-01-01"}],
                }
            }
        )

        qs = Book.objects.ai_query("books published after 1900", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 2  # Grapes of Wrath and Handmaid's Tale
        modern_books = {book.title for book in results}
        assert "The Grapes of Wrath" in modern_books
        assert "The Handmaid's Tale" in modern_books

    def test_find_books_by_american_authors(self):
        """Test finding books by American authors using relationship."""
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
        assert results[0].title == "The Grapes of Wrath"

    def test_find_authors_with_published_books(self):
        """Test finding authors who have published books."""
        dummy_llm = DummyLLM(
            {
                "authors with published books": {
                    "status": "ok",
                    "filters": [{"path": "books__is_published", "op": "exact", "value": True}],
                }
            }
        )

        qs = Author.objects.ai_query("authors with published books", llm=dummy_llm)
        results = list(qs.distinct())  # Use distinct to avoid duplicates

        assert len(results) == 3  # All authors have published books

    def test_find_books_with_high_ratings(self):
        """Test finding books with high ratings using reviews."""
        dummy_llm = DummyLLM(
            {
                "books with 5 star ratings": {
                    "status": "ok",
                    "filters": [{"path": "reviews__rating", "op": "exact", "value": 5}],
                }
            }
        )

        qs = Book.objects.ai_query("books with 5 star ratings", llm=dummy_llm)
        results = list(qs.distinct())

        assert len(results) == 2  # Grapes of Wrath and Handmaid's Tale
        five_star_books = {book.title for book in results}
        assert "The Grapes of Wrath" in five_star_books
        assert "The Handmaid's Tale" in five_star_books

    def test_complex_query_with_multiple_filters(self):
        """Test complex query with multiple filters."""
        dummy_llm = DummyLLM(
            {
                "published books by active authors from North America": {
                    "status": "ok",
                    "logic": "and",
                    "filters": [
                        {"path": "is_published", "op": "exact", "value": True},
                        {"path": "authors__is_active", "op": "exact", "value": True},
                        {"path": "authors__country", "op": "in", "value": ["USA", "Canada"]},
                    ],
                }
            }
        )

        qs = Book.objects.ai_query("published books by active authors from North America", llm=dummy_llm)
        results = list(qs.distinct())

        assert len(results) == 2  # Grapes of Wrath and Handmaid's Tale
        north_american_books = {book.title for book in results}
        assert "The Grapes of Wrath" in north_american_books
        assert "The Handmaid's Tale" in north_american_books

    def test_query_with_ordering(self):
        """Test query with custom ordering."""
        dummy_llm = DummyLLM(
            {
                "all books ordered by publication date": {
                    "status": "ok",
                    "filters": [{"path": "is_published", "op": "exact", "value": True}],
                    "order_by": ["published_date"],
                }
            }
        )

        qs = Book.objects.ai_query("all books ordered by publication date", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 3
        # Should be ordered chronologically
        assert results[0].title == "Pride and Prejudice"  # 1813
        assert results[1].title == "The Grapes of Wrath"  # 1939
        assert results[2].title == "The Handmaid's Tale"  # 1985

    def test_query_with_limit(self):
        """Test query with limit."""
        dummy_llm = DummyLLM({"first 2 books": {"status": "ok", "filters": [], "order_by": ["title"], "limit": 2}})

        qs = Book.objects.ai_query("first 2 books", llm=dummy_llm)
        results = list(qs)

        assert len(results) == 2

    def test_or_logic_query(self):
        """Test query with OR logic."""
        dummy_llm = DummyLLM(
            {
                "books by John or Jane": {
                    "status": "ok",
                    "logic": "or",
                    "filters": [
                        {"path": "authors__name", "op": "icontains", "value": "John"},
                        {"path": "authors__name", "op": "icontains", "value": "Jane"},
                    ],
                }
            }
        )

        qs = Book.objects.ai_query("books by John or Jane", llm=dummy_llm)
        results = list(qs.distinct())

        assert len(results) == 2
        author_books = {book.title for book in results}
        assert "The Grapes of Wrath" in author_books  # by John
        assert "Pride and Prejudice" in author_books  # by Jane

    def test_error_handling_with_invalid_field(self):
        """Test error handling when LLM returns invalid field."""
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
        assert "nonexistent_field" in str(qs.ai_error)

    def test_error_handling_with_llm_error(self):
        """Test error handling when LLM returns error status."""
        dummy_llm = DummyLLM({"ambiguous query": {"status": "error", "message": "Query is too ambiguous to parse"}})

        qs = Book.objects.ai_query("ambiguous query", llm=dummy_llm)

        assert qs.count() == 0
        assert hasattr(qs, "ai_error")
        assert "too ambiguous" in str(qs.ai_error)


@pytest.mark.django_db
class TestSchemaGeneration(TestCase):
    """Test schema generation with complex models."""

    def test_complete_schema_generation(self):
        """Test generating schema for complex model relationships."""
        schema = build_schema_for_model(Book, depth=2, include_reverse=True)

        # Should include all related models
        assert "Book" in schema["models"]
        assert "Author" in schema["models"]
        assert "Publisher" in schema["models"]
        assert "Review" in schema["models"]

        # Check Book model structure
        book_model = schema["models"]["Book"]
        assert "fields" in book_model
        assert "relationships" in book_model

        # Check Book fields
        book_fields = book_model["fields"]
        expected_fields = ["title", "isbn", "published_date", "page_count", "is_published", "price"]
        for field in expected_fields:
            assert field in book_fields

        # Check Book relationships
        book_rels = book_model["relationships"]
        assert "authors" in book_rels
        assert book_rels["authors"]["type"] == "M2M"
        assert book_rels["authors"]["to"] == "Author"

        assert "publisher" in book_rels
        assert book_rels["publisher"]["type"] == "FK"
        assert book_rels["publisher"]["to"] == "Publisher"

        # Check reverse relationships
        assert "reviews" in book_rels
        assert book_rels["reviews"]["type"] == "REV"
        assert book_rels["reviews"]["to"] == "Review"

        # Check Author model has reverse relationship to books
        author_model = schema["models"]["Author"]
        author_rels = author_model["relationships"]
        assert "books" in author_rels
        assert author_rels["books"]["type"] == "REV"

    def test_schema_depth_limiting(self):
        """Test that schema depth limiting works correctly."""
        # Depth 0 should only include the root model
        schema_depth_0 = build_schema_for_model(Book, depth=0)
        assert len(schema_depth_0["models"]) == 1
        assert "Book" in schema_depth_0["models"]

        # Depth 1 should include immediate relationships
        schema_depth_1 = build_schema_for_model(Book, depth=1)
        models_depth_1 = set(schema_depth_1["models"].keys())
        expected_depth_1 = {"Book", "Author", "Publisher", "Review"}
        assert models_depth_1 == expected_depth_1

    def test_schema_reverse_relationship_control(self):
        """Test controlling reverse relationship inclusion."""
        # With reverse relationships
        schema_with_reverse = build_schema_for_model(Author, depth=1, include_reverse=True)
        author_model = schema_with_reverse["models"]["Author"]
        assert "books" in author_model["relationships"]

        # Without reverse relationships
        schema_without_reverse = build_schema_for_model(Author, depth=1, include_reverse=False)
        author_model = schema_without_reverse["models"]["Author"]
        assert "books" not in author_model["relationships"]


@pytest.mark.django_db
class TestRealWorldScenarios(TestCase):
    """Test scenarios that might occur in real-world usage."""

    def setUp(self):
        self.setup_realistic_data()

    def setup_realistic_data(self):
        """Create realistic test data."""
        # Create publishers
        self.publishers = []
        for name, country, year in [
            ("Penguin Random House", "USA", 2013),
            ("HarperCollins", "USA", 1989),
            ("Macmillan", "UK", 1843),
            ("Simon & Schuster", "USA", 1924),
        ]:
            pub = Publisher.objects.create(name=name, country=country, founded_year=year)
            self.publishers.append(pub)

        # Create authors
        self.authors = []
        author_data = [
            ("Stephen King", "USA", date(1947, 9, 21), True),
            ("J.K. Rowling", "UK", date(1965, 7, 31), True),
            ("George Orwell", "UK", date(1903, 6, 25), False),
            ("Toni Morrison", "USA", date(1931, 2, 18), False),
            ("Haruki Murakami", "Japan", date(1949, 1, 12), True),
        ]
        for name, country, birth_date, is_active in author_data:
            author = Author.objects.create(
                name=name,
                country=country,
                birth_date=birth_date,
                is_active=is_active,
                email=f"{name.lower().replace(' ', '.')}@example.com",
            )
            self.authors.append(author)

        # Create books with realistic data
        book_data = [
            ("The Shining", "9780307743657", 0, [0], date(1977, 1, 28), 447, True, 15.99),
            ("Harry Potter and the Sorcerer's Stone", "9780439708180", 1, [1], date(1997, 6, 26), 309, True, 8.99),
            ("1984", "9780451524935", 2, [2], date(1949, 6, 8), 328, True, 13.99),
            ("Beloved", "9781400033416", 0, [3], date(1987, 9, 1), 324, True, 14.99),
            ("Norwegian Wood", "9780375704024", 0, [4], date(1987, 9, 4), 296, True, 16.99),
            ("Unpublished Novel", "9999999999999", 0, [0], None, None, False, None),
        ]

        self.books = []
        for title, isbn, pub_idx, author_indices, pub_date, pages, published, price in book_data:
            book = Book.objects.create(
                title=title,
                isbn=isbn,
                publisher=self.publishers[pub_idx],
                published_date=pub_date,
                page_count=pages,
                is_published=published,
                price=price,
            )
            for author_idx in author_indices:
                book.authors.add(self.authors[author_idx])
            self.books.append(book)

        # Create reviews
        review_data = [
            (0, "Horror Fan", 5, "Absolutely terrifying!"),
            (1, "Fantasy Reader", 5, "Magical and wonderful!"),
            (1, "Young Adult", 4, "Great for kids and adults"),
            (2, "Political Reader", 5, "Chilling and prophetic"),
            (3, "Literature Student", 4, "Powerful and moving"),
            (4, "International Reader", 5, "Beautiful and melancholic"),
        ]

        for book_idx, reviewer, rating, comment in review_data:
            Review.objects.create(book=self.books[book_idx], reviewer_name=reviewer, rating=rating, comment=comment)

    def test_natural_language_author_search(self):
        """Test natural language author searches."""
        test_cases = [
            ("authors from Japan", [4]),  # Murakami
            ("British authors", [1, 2]),  # Rowling, Orwell
            ("active American authors", [0]),  # King (Morrison is not active)
            ("authors born after 1950", [1]),  # Rowling only (Murakami born 1949)
        ]

        for query, expected_indices in test_cases:
            dummy_llm = self._create_author_search_llm(query, expected_indices)
            qs = Author.objects.ai_query(query, llm=dummy_llm)
            results = list(qs)

            expected_authors = [self.authors[i] for i in expected_indices]
            assert len(results) == len(expected_authors)

            result_names = {author.name for author in results}
            expected_names = {author.name for author in expected_authors}
            assert result_names == expected_names

    def _create_author_search_llm(self, query, expected_indices):
        """Helper to create DummyLLM for author searches."""
        filters = []

        if "Japan" in query:
            filters.append({"path": "country", "op": "iexact", "value": "Japan"})
        elif "British" in query:
            filters.append({"path": "country", "op": "iexact", "value": "UK"})
        elif "American" in query:
            filters.append({"path": "country", "op": "iexact", "value": "USA"})

        if "active" in query:
            filters.append({"path": "is_active", "op": "exact", "value": True})

        if "born after 1950" in query:
            filters.append({"path": "birth_date", "op": "gt", "value": "1950-01-01"})

        return DummyLLM({query: {"status": "ok", "logic": "and", "filters": filters}})

    def test_natural_language_book_search(self):
        """Test natural language book searches."""
        # Test horror books
        dummy_llm = DummyLLM(
            {"horror books": {"status": "ok", "filters": [{"path": "title", "op": "icontains", "value": "Shining"}]}}
        )
        qs = Book.objects.ai_query("horror books", llm=dummy_llm)
        results = list(qs)
        assert len(results) == 1  # Should find "The Shining"
        assert "Shining" in results[0].title

        # Test books published before 1990
        dummy_llm = DummyLLM(
            {
                "books published before 1990": {
                    "status": "ok",
                    "filters": [{"path": "published_date", "op": "lt", "value": "1990-01-01"}],
                }
            }
        )
        qs = Book.objects.ai_query("books published before 1990", llm=dummy_llm)
        results = list(qs)
        assert len(results) >= 3  # Should find books from 1977, 1949, 1987

        # Test books with many pages
        dummy_llm = DummyLLM(
            {
                "books with more than 400 pages": {
                    "status": "ok",
                    "filters": [{"path": "page_count", "op": "gt", "value": 400}],
                }
            }
        )
        qs = Book.objects.ai_query("books with more than 400 pages", llm=dummy_llm)
        results = list(qs)
        assert len(results) >= 1  # Should find "The Shining" (447 pages)

    def test_complex_relationship_queries(self):
        """Test complex queries involving relationships."""
        queries = {
            "books by living authors": [{"path": "authors__is_active", "op": "exact", "value": True}],
            "books published by American publishers": [{"path": "publisher__country", "op": "iexact", "value": "USA"}],
            "highly rated books": [{"path": "reviews__rating", "op": "gte", "value": 5}],
            "books by authors born in the 20th century": [
                {"path": "authors__birth_date", "op": "gte", "value": "1901-01-01"}
            ],
        }

        for query, filters in queries.items():
            dummy_llm = DummyLLM({query: {"status": "ok", "filters": filters}})

            qs = Book.objects.ai_query(query, llm=dummy_llm)
            results = list(qs.distinct())  # Use distinct to avoid duplicates from joins
            assert len(results) > 0
