from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

import requests
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone

from ..models import NationalHoliday, PublicHoliday


NAGER_BASE_URL = "https://date.nager.at/api/v3"


@dataclass(frozen=True)
class HolidayResult:
    is_holiday: bool
    title: Optional[str] = None


def is_holiday_for_org(org_user, day: date, country_code: str = "NG") -> HolidayResult:
    """Return whether `day` is a holiday for an org.

    Holiday sources:
    - Manual org holidays: `PublicHoliday` (user-scoped)
    - National holidays: `NationalHoliday` (country-scoped)
    """

    _ensure_year_synced(day.year, country_code=country_code)

    if PublicHoliday.objects.filter(user=org_user, start_date__lte=day, end_date__gte=day).exists():
        return HolidayResult(True, "Organisation holiday")
    nh = NationalHoliday.objects.filter(country_code=country_code, date=day).first()
    if nh:
        return HolidayResult(True, nh.name)
    return HolidayResult(False, None)


def working_days(org_user, start_date: date, end_date: date, country_code: str = "NG") -> list[date]:
    """Return list of working dates excluding weekends and holidays."""

    _ensure_year_synced(start_date.year, country_code=country_code)
    _ensure_year_synced(end_date.year, country_code=country_code)

    manual = PublicHoliday.objects.filter(user=org_user, start_date__lte=end_date, end_date__gte=start_date)
    national = NationalHoliday.objects.filter(country_code=country_code, date__gte=start_date, date__lte=end_date)
    national_dates = set(national.values_list("date", flat=True))

    def manual_hit(d: date) -> bool:
        return manual.filter(start_date__lte=d, end_date__gte=d).exists()

    days: list[date] = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5 and d not in national_dates and not manual_hit(d):
            days.append(d)
        d = d.fromordinal(d.toordinal() + 1)
    return days


def fetch_nager_public_holidays(year: int, country_code: str = "NG", timeout_s: int = 20) -> list[dict]:
    url = f"{NAGER_BASE_URL}/PublicHolidays/{year}/{country_code}"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return []
    return data


def _ensure_year_synced(year: int, country_code: str = "NG") -> None:
    """Lazy sync: if national holidays for the year are missing, sync once per day.

    This provides automatic behavior without requiring Celery.
    """

    # Already have holidays for this year
    if NationalHoliday.objects.filter(country_code=country_code, date__year=year).exists():
        return

    key = f"national_holidays_sync:{country_code}:{year}"
    if cache.get(key):
        return
    # Prevent repeated fetch attempts for 24h on failure
    cache.set(key, True, timeout=24 * 3600)
    try:
        sync_national_holidays(year=year, country_code=country_code)
    except Exception:
        # Ignore failures; next day attempt again
        return


@transaction.atomic
def sync_national_holidays(year: int, country_code: str = "NG") -> int:
    """Sync a year's public holidays for `country_code` from Nager.Date.

    Returns number of records upserted.
    """

    rows = fetch_nager_public_holidays(year=year, country_code=country_code)
    upserted = 0
    for row in rows:
        day = row.get("date")
        name = row.get("name")
        local = row.get("localName") or ""
        if not day or not name:
            continue
        try:
            d = date.fromisoformat(day)
        except Exception:
            continue
        NationalHoliday.objects.update_or_create(
            country_code=country_code,
            date=d,
            name=name,
            defaults={
                "local_name": local,
                "raw": row,
            },
        )
        upserted += 1
    return upserted


def ensure_national_holidays(year: int, country_code: str = "NG") -> None:
    """Public helper to ensure a year's national holidays exist (lazy sync)."""

    _ensure_year_synced(year, country_code=country_code)
