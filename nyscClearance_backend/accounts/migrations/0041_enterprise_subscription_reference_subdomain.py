from django.db import migrations, models
import uuid


def backfill_subscription_references(apps, schema_editor):
    Subscription = apps.get_model('accounts', 'OrganizationSubscription')
    for subscription in Subscription.objects.filter(reference=''):
        subscription.reference = f"SUB-ADMIN-{uuid.uuid4().hex[:18]}".upper()
        subscription.save(update_fields=['reference'])


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0040_googlemapsconfig'),
    ]

    operations = [
        migrations.AlterField(
            model_name='organizationsubscription',
            name='plan_code',
            field=models.CharField(blank=True, max_length=20),
        ),
        migrations.AlterField(
            model_name='organizationsubscription',
            name='plan_name',
            field=models.CharField(blank=True, max_length=80),
        ),
        migrations.AddField(
            model_name='organizationsubscription',
            name='reference',
            field=models.CharField(blank=True, default='', max_length=64),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='organizationsubscription',
            name='subdomain',
            field=models.SlugField(blank=True, default='', help_text='Enterprise organization subdomain, for example "acme" for acme.nyscclearance.com', max_length=63),
            preserve_default=False,
        ),
        migrations.RunPython(backfill_subscription_references, reverse_code=migrations.RunPython.noop),
        migrations.AlterField(
            model_name='organizationsubscription',
            name='reference',
            field=models.CharField(blank=True, max_length=64, unique=True),
        ),
    ]
