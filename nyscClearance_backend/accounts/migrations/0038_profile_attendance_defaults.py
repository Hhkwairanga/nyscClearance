from datetime import time

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0037_organizationuser_force_password_change'),
    ]

    operations = [
        migrations.AlterField(
            model_name='organizationprofile',
            name='late_time',
            field=models.TimeField(blank=True, default=time(8, 30), null=True),
        ),
        migrations.AlterField(
            model_name='organizationprofile',
            name='closing_time',
            field=models.TimeField(blank=True, default=time(16, 0), null=True),
        ),
        migrations.AlterField(
            model_name='organizationprofile',
            name='max_days_late',
            field=models.PositiveSmallIntegerField(blank=True, default=5, null=True),
        ),
        migrations.AlterField(
            model_name='organizationprofile',
            name='max_days_absent',
            field=models.PositiveSmallIntegerField(blank=True, default=3, null=True),
        ),
    ]

