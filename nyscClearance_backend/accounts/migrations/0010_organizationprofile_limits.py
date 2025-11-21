from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0009_alter_publicholiday_unique_together_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="organizationprofile",
            name="max_days_late",
            field=models.PositiveSmallIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="organizationprofile",
            name="max_days_absent",
            field=models.PositiveSmallIntegerField(blank=True, null=True),
        ),
    ]

