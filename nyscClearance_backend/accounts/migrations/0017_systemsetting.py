from django.db import migrations, models
import decimal


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0016_alter_walletaccount_balance_alter_walletaccount_id_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="SystemSetting",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("welcome_bonus", models.DecimalField(decimal_places=2, default=decimal.Decimal("10000.00"), max_digits=12)),
                ("discount_enabled", models.BooleanField(default=False)),
                ("discount_percent", models.DecimalField(decimal_places=2, default=decimal.Decimal("0.00"), max_digits=5)),
                ("notify_enabled", models.BooleanField(default=False)),
                ("notify_title", models.CharField(blank=True, max_length=200)),
                ("notify_message", models.TextField(blank=True)),
                ("notify_start", models.DateTimeField(blank=True, null=True)),
                ("notify_end", models.DateTimeField(blank=True, null=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "System Setting",
                "verbose_name_plural": "System Settings",
            },
        ),
    ]

