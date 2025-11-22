from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0012_organizationuser_phone_number'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='organizationuser',
            name='number_of_corpers',
        ),
    ]

