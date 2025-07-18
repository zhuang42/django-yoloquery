"""Test configuration for django-yoloquery."""

import os
import django
from django.conf import settings

# Ensure Django is configured before importing models
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.settings")

if not settings.configured:
    django.setup()


# Now force auto-install after Django is configured
def pytest_configure(config):
    """Configure pytest and ensure auto-install runs."""
    from django_yoloquery import auto_install_yolo_managers

    auto_install_yolo_managers()
