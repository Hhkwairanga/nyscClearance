from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0041_enterprise_subscription_reference_subdomain'),
    ]

    operations = [
        migrations.AddConstraint(
            model_name='organizationsubscription',
            constraint=models.UniqueConstraint(
                condition=~models.Q(('subdomain', '')),
                fields=('subdomain',),
                name='unique_enterprise_subscription_subdomain',
            ),
        ),
    ]
