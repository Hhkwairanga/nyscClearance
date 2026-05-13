"""Lightweight scheduled tasks.

This project does not ship with Celery/APScheduler by default. The functions
here are meant to be called by an external scheduler (cron, systemd timer,
GitHub Actions, etc.) via a management command.
"""

from django.utils import timezone

from .services.holidays import sync_national_holidays


def sync_nigeria_holidays_for_year(year: int | None = None) -> int:
    year = year or timezone.localdate().year
    return sync_national_holidays(year=year, country_code='NG')

