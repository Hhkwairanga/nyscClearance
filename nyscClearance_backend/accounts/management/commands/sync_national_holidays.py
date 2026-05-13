from django.core.management.base import BaseCommand
from django.utils import timezone

from accounts.services.holidays import sync_national_holidays


class Command(BaseCommand):
    help = "Sync national public holidays from Nager.Date (default: Nigeria)."

    def add_arguments(self, parser):
        parser.add_argument('--year', type=int, default=timezone.localdate().year)
        parser.add_argument('--country', type=str, default='NG')

    def handle(self, *args, **options):
        year = int(options['year'])
        country = str(options['country']).upper().strip() or 'NG'
        count = sync_national_holidays(year=year, country_code=country)
        self.stdout.write(self.style.SUCCESS(f"Synced {count} holidays for {country} {year}"))

