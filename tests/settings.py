"""Django settings for tests."""


SECRET_KEY = "test-secret-key-for-django-yoloquery-tests"

DEBUG = True

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django_yoloquery",
    "tests.testapp",
]

USE_TZ = True

# YOLOQuery test settings
YOLOQUERY_SCHEMA_DEPTH = 2
YOLOQUERY_INCLUDE_REVERSE = True
YOLOQUERY_LLM_MODEL = "gpt-4o-mini"
YOLOQUERY_AUTO_INSTALL = ["testapp.*"]

# Don't require OpenAI key for tests
YOLOQUERY_OPENAI_API_KEY = None
