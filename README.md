# Django YOLOQuery

**LLM‑Based Queries for the Django ORM**

⚠️ **This is just a Proof-of-Concept**

[![PyPI version](https://badge.fury.io/py/django-yoloquery.svg)](https://badge.fury.io/py/django-yoloquery)
[![Python versions](https://img.shields.io/pypi/pyversions/django-yoloquery.svg)](https://pypi.org/project/django-yoloquery/)
[![Django versions](https://img.shields.io/badge/django-3.2%2B-blue.svg)](https://www.djangoproject.com/)

---

## Why?

I want to see how badly LLMs can be integrated into our system

---

## TL;DR

```python
authors = Author.objects.ai_query("get all authors named John")
books   = Book.objects.ai_query("books published after 2020 by American authors")
recent  = Author.objects.ai_query("authors with books published in the last 5 years")
```

---


## Install

```bash
pip install django-yoloquery
```

---

## Configure

Add to **`INSTALLED_APPS`** and set a few knobs in `settings.py`.

```python
INSTALLED_APPS = [
    # ... your apps ...
    'django_yoloquery',
]

# Required: OpenAI API key (or set env OPENAI_API_KEY)
YOLOQUERY_OPENAI_API_KEY = "sk-..."

# Optional (shown with defaults)
YOLOQUERY_SCHEMA_DEPTH = 1            # how far to traverse relations
YOLOQUERY_INCLUDE_REVERSE = True      # include reverse relations (related_name / _set)
YOLOQUERY_LLM_MODEL = "gpt-4o-mini"   # pick your OpenAI model
YOLOQUERY_AUTO_INSTALL = ["myapp.*"]  # auto‑YOLO patch models
```

> 🔑 If `YOLOQUERY_OPENAI_API_KEY` is not set, YOLOQuery falls back to the `OPENAI_API_KEY` environment variable.

---

## Manual Wiring (if you don’t want auto‑patch)

```python
from django.db import models
from django_yoloquery import YoloManager

class Author(models.Model):
    name       = models.CharField(max_length=200)
    country    = models.CharField(max_length=100)
    birth_date = models.DateField()

    objects = YoloManager()  # manual install

class Book(models.Model):
    title          = models.CharField(max_length=200)
    authors        = models.ManyToManyField(Author, related_name="books")
    published_date = models.DateField()
    isbn           = models.CharField(max_length=13)
```

---

## Auto‑Installation Patterns

YOLOQuery can **monkey‑patch all the things** at startup. Patterns are case‑sensitive app labels; model name matching is case‑insensitive.

```python
YOLOQUERY_AUTO_INSTALL = [
    "myapp.*",        # all models in myapp
    "blog.Post",      # just that one model
    "*",              # YOLO EVERYTHING (seriously?)
]
```

When auto‑installed:

* The model’s default manager (`.objects`) is replaced with a YOLO manager.
* The original manager is saved as `._orig_objects`.
* A `.yolo` alias also points at the YOLO manager because branding.

---

## What the LLM Sees

YOLOQuery builds a JSON description of your model + related models (depth configurable, reverse rels optional). Example schema snippet for `Author`:

```json
{
  "root_model": "Author",
  "app_label": "myapp",
  "models": {
    "Author": {
      "fields": {
        "name":    {"type": "CharField", "null": false},
        "country": {"type": "CharField", "null": false},
        "birth_date": {"type": "DateField", "null": false}
      },
      "relationships": {
        "books": {"type": "REV", "to": "Book"}
      }
    },
    "Book": {
      "fields": {
        "title": {"type": "CharField", "null": false},
        "published_date": {"type": "DateField", "null": false}
      },
      "relationships": {
        "authors": {"type": "M2M", "to": "Author"}
      }
    }
  }
}
```

---

## Structured JSON Contract

The model must return JSON like:

```json
{
  "status": "ok",
  "logic": "and",             // or "or"
  "filters": [
    {"path": "name", "op": "iexact", "value": "John"},
    {"path": "books__published_date", "op": "gt", "value": "2020-01-01"}
  ],
  "order_by": ["-published_date"],  // optional
  "limit": 10                        // optional
}
```

If the request can’t be translated (missing info / nonsense / hallucination danger), the model must respond:

```json
{"status": "error", "message": "Missing value for field 'name'"}
```

YOLOQuery will turn that into an `AIQueryLLMError` (attached to an empty QuerySet unless `raise_errors=True`).

---

## Supported Query Concepts

| You Say                         | LLM Might Output                       | Django Filters                      |                                 |
| ------------------------------- | -------------------------------------- | ----------------------------------- | ------------------------------- |
| "authors named john"            | `name iexact John`                     | `name__iexact="John"`               |                                 |
| "authors from usa or canada"    | logic=or + country in \[USA, Canada]   | \`Q(country\_\_iexact="USA")        | Q(country\_\_iexact="Canada")\` |
| "books published after 2020"    | published\_date gt 2020-01-01          | `published_date__gt=date(2020,1,1)` |                                 |
| "authors with books after 2020" | books\_\_published\_date gt 2020-01-01 | join across reverse M2M             |                                 |
| "top 10 newest books"           | order\_by=-published\_date, limit=10   | `.order_by('-published_date')[:10]` |                                 |

---

## Error Handling

YOLOQuery errs on the side of *not doing something dumb*:

```python
qs = Author.objects.ai_query("authors where")  # incomplete!
if hasattr(qs, 'ai_error'):
    print("LLM said nope:", qs.ai_error)
```

Raise immediately:

```python
Author.objects.ai_query("authors where", raise_errors=True)
# AIQueryLLMError: LLM could not translate query.
```

---

## Testing (No API Calls)

```python
from django_yoloquery import DummyLLM

DUMMY_RESPONSES = {
    "authors named John": {
        "status": "ok",
        "filters": [{"path": "name", "op": "iexact", "value": "John"}]
    },
    "name is": {
        "status": "error", "message": "Missing value for name"
    },
}

dummy = DummyLLM(DUMMY_RESPONSES)
qs = Author.objects.ai_query("authors named John", llm=dummy)
assert list(qs.values_list("name", flat=True)) == ["John"]
```

---

## FAQ (Frequently Asked Questions No One Asked Yet)

**Q: Does this belong in production?**
A: The library literally has YOLO in the name.

**Q: Will it leak my schema to OpenAI?**
A: Yes, that’s how it works. Use stub schemas or anonymized field names if that’s a problem.

**Q: Can I use Anthropic / local models?**
A: PRs welcome. Right now it’s OpenAI only; we rely on JSON Schema format.

**Q: What if the model hallucinates?**
A: We validate every field + op; hallucinations become errors instead of DB hits. (We try, anyway.)

**Q: Does it understand dates like "last quarter"?**
A: Depends on the model; we just validate output.

---

## Requirements

* Python 3.8+
* Django 3.2+
* `openai>=1.0.0`
* OpenAI API key

---

## License

MIT – see [LICENSE](LICENSE).
