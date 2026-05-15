from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0024_rename_accounts_te_corper__eb1f35_idx_accounts_te_corper__a77caf_idx'),
    ]

    operations = [
        migrations.AddField(
            model_name='corpmember',
            name='cds_day',
            field=models.PositiveSmallIntegerField(blank=True, help_text='CDS weekday: 0=Mon..4=Fri', null=True),
        ),
    ]

