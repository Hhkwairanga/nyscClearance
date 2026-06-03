from datetime import time

from django.db import migrations


def backfill_profile_defaults(apps, schema_editor):
    OrganizationProfile = apps.get_model('accounts', 'OrganizationProfile')
    for profile in OrganizationProfile.objects.all().iterator():
        changed = []
        if profile.late_time is None:
            profile.late_time = time(8, 30)
            changed.append('late_time')
        if profile.closing_time is None:
            profile.closing_time = time(16, 0)
            changed.append('closing_time')
        if profile.max_days_late is None:
            profile.max_days_late = 5
            changed.append('max_days_late')
        if profile.max_days_absent is None:
            profile.max_days_absent = 3
            changed.append('max_days_absent')
        if changed:
            profile.save(update_fields=changed)


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0038_profile_attendance_defaults'),
    ]

    operations = [
        migrations.RunPython(backfill_profile_defaults, migrations.RunPython.noop),
    ]

