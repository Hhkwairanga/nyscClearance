from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0013_remove_organizationuser_number_of_corpers'),
    ]

    operations = [
        migrations.AddField(
            model_name='organizationprofile',
            name='signatory_name',
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.AddField(
            model_name='organizationprofile',
            name='signature',
            field=models.ImageField(blank=True, null=True, upload_to='org_signatures/'),
        ),
    ]
