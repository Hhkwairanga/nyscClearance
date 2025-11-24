from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0020_paystackconfig"),
    ]

    operations = [
        migrations.CreateModel(
            name="ClearanceOverride",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("year_month", models.CharField(max_length=6)),
                ("reason", models.CharField(blank=True, max_length=255)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "corper",
                    models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="clearance_overrides", to="accounts.corpmember"),
                ),
                (
                    "created_by",
                    models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="granted_clearance_overrides", to="accounts.organizationuser"),
                ),
            ],
            options={
                "ordering": ("-created_at",),
                "unique_together": {("corper", "year_month")},
                "verbose_name": "Clearance Override",
                "verbose_name_plural": "Clearance Overrides",
            },
        ),
    ]

