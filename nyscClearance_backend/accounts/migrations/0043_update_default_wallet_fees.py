from decimal import Decimal

from django.db import migrations, models


def update_default_wallet_fees(apps, schema_editor):
    SystemSetting = apps.get_model('accounts', 'SystemSetting')
    setting, _created = SystemSetting.objects.get_or_create(pk=1)
    setting.welcome_bonus = Decimal('10000.00')
    setting.clearance_fee = Decimal('1000.00')
    setting.save(update_fields=['welcome_bonus', 'clearance_fee', 'updated_at'])


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0042_unique_enterprise_subscription_subdomain'),
    ]

    operations = [
        migrations.AlterField(
            model_name='systemsetting',
            name='clearance_fee',
            field=models.DecimalField(decimal_places=2, default=Decimal('1000.00'), max_digits=12),
        ),
        migrations.RunPython(update_default_wallet_fees, migrations.RunPython.noop),
    ]
