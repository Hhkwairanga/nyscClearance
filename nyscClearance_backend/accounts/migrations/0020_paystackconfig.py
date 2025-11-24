from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0019_systemsetting_add_clearance_fee"),
    ]

    operations = [
        migrations.CreateModel(
            name="PaystackConfig",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("public_key", models.CharField(max_length=200)),
                ("secret_key", models.CharField(max_length=200)),
                ("webhook_secret", models.CharField(blank=True, max_length=200)),
                ("is_active", models.BooleanField(default=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "Paystack Config",
                "verbose_name_plural": "Paystack Config",
            },
        ),
    ]

