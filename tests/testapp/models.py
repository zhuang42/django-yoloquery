"""Test models for django-yoloquery."""

from django.db import models
from django_yoloquery import YoloManager


class Author(models.Model):
    name = models.CharField(max_length=200)
    country = models.CharField(max_length=100, blank=True)
    birth_date = models.DateField(null=True, blank=True)
    email = models.EmailField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "testapp"

    def __str__(self):
        return self.name


class Publisher(models.Model):
    name = models.CharField(max_length=200)
    country = models.CharField(max_length=100, blank=True)
    founded_year = models.IntegerField(null=True, blank=True)

    class Meta:
        app_label = "testapp"

    def __str__(self):
        return self.name


class Book(models.Model):
    title = models.CharField(max_length=200)
    isbn = models.CharField(max_length=13, unique=True)
    authors = models.ManyToManyField(Author, related_name="books")
    publisher = models.ForeignKey(Publisher, on_delete=models.CASCADE, related_name="books")
    published_date = models.DateField(null=True, blank=True)
    page_count = models.IntegerField(null=True, blank=True)
    is_published = models.BooleanField(default=False)
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)

    class Meta:
        app_label = "testapp"

    def __str__(self):
        return self.title


class Review(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name="reviews")
    reviewer_name = models.CharField(max_length=100)
    rating = models.IntegerField()  # 1-5 stars
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "testapp"

    def __str__(self):
        return f"{self.reviewer_name} - {self.book.title}"


# Test manual manager installation
class ManualModel(models.Model):
    name = models.CharField(max_length=100)

    # Manually attach YoloManager
    objects = YoloManager()

    class Meta:
        app_label = "testapp"
