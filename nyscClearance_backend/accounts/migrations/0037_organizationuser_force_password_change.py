from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0036_units_org_wide_departments_no_office_link'),
    ]

    operations = [
        migrations.AddField(
            model_name='organizationuser',
            name='force_password_change',
            field=models.BooleanField(default=False),
        ),
    ]

