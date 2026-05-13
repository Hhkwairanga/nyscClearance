from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0022_tempfaceencoding'),
    ]

    operations = [
        migrations.CreateModel(
            name='NationalHoliday',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('country_code', models.CharField(db_index=True, default='NG', max_length=2)),
                ('date', models.DateField(db_index=True)),
                ('name', models.CharField(max_length=255)),
                ('local_name', models.CharField(blank=True, default='', max_length=255)),
                ('raw', models.JSONField(blank=True, default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ('date', 'name'),
                'unique_together': {('country_code', 'date', 'name')},
            },
        ),
    ]

