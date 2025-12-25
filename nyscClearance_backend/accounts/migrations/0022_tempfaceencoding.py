from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0021_clearanceoverride'),
    ]

    operations = [
        migrations.CreateModel(
            name='TempFaceEncoding',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('session_id', models.CharField(db_index=True, max_length=64)),
                ('idx', models.PositiveIntegerField(default=0)),
                ('vector', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('corper', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='temp_face_encodings', to='accounts.corpmember')),
            ],
            options={
                'ordering': ('created_at', 'id'),
            },
        ),
        migrations.AddIndex(
            model_name='tempfaceencoding',
            index=models.Index(fields=['corper', 'session_id'], name='accounts_te_corper__eb1f35_idx'),
        ),
    ]

