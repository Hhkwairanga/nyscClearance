from django.db import migrations, models
import django.db.models.deletion
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0010_organizationprofile_limits'),
    ]

    operations = [
        migrations.CreateModel(
            name='AttendanceLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('state', models.CharField(blank=True, max_length=64)),
                ('code', models.CharField(blank=True, max_length=32)),
                ('date', models.DateField()),
                ('time_in', models.TimeField(blank=True, null=True)),
                ('time_out', models.TimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('account', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='attendance_logs', to=settings.AUTH_USER_MODEL)),
                ('org', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='org_attendance_logs', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ('-date', '-created_at'),
                'unique_together': {('account', 'date')},
            },
        ),
    ]

