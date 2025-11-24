from django.db import migrations, models
import decimal


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0018_alter_systemsetting_id"),
    ]

    operations = [
        migrations.AddField(
            model_name="systemsetting",
            name="clearance_fee",
            field=models.DecimalField(decimal_places=2, default=decimal.Decimal("300.00"), max_digits=12),
        ),
    ]

