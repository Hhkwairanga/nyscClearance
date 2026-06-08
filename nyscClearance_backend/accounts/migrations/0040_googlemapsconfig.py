from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0039_backfill_profile_attendance_defaults'),
    ]

    operations = [
        migrations.CreateModel(
            name='GoogleMapsConfig',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default='Default Google Maps Config', max_length=120)),
                ('browser_api_key', models.CharField(blank=True, max_length=255)),
                ('server_api_key', models.CharField(blank=True, max_length=255)),
                ('map_id', models.CharField(blank=True, max_length=120)),
                ('is_active', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Google Maps Config',
                'verbose_name_plural': 'Google Maps Config',
            },
        ),
    ]

