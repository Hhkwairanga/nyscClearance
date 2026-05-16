from django.contrib.sessions.models import Session
from django.core.management.base import BaseCommand
from django.db import transaction

from accounts.models import SystemSetting


class Command(BaseCommand):
    help = "Invalidate all active sessions and signed access tokens."

    def handle(self, *args, **options):
        with transaction.atomic():
            settings = SystemSetting.current()
            settings.auth_token_version = int(settings.auth_token_version or 1) + 1
            settings.save(update_fields=['auth_token_version', 'updated_at'])
            deleted_count, _ = Session.objects.all().delete()

        self.stdout.write(
            self.style.SUCCESS(
                f"Forced logout complete. Deleted {deleted_count} session(s); token version is now {settings.auth_token_version}."
            )
        )
