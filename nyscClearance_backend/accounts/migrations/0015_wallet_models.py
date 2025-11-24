from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0014_organizationprofile_signatory'),
    ]

    operations = [
        migrations.CreateModel(
            name='WalletAccount',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('balance', models.DecimalField(decimal_places=2, default='0.00', max_digits=12)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='wallet', to='accounts.organizationuser')),
            ],
        ),
        migrations.CreateModel(
            name='WalletTransaction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type', models.CharField(choices=[('CREDIT', 'Credit'), ('DEBIT', 'Debit')], max_length=6)),
                ('amount', models.DecimalField(decimal_places=2, max_digits=12)),
                ('vat_amount', models.DecimalField(decimal_places=2, default='0.00', max_digits=12)),
                ('total_amount', models.DecimalField(decimal_places=2, max_digits=12)),
                ('description', models.CharField(blank=True, max_length=255)),
                ('reference', models.CharField(blank=True, max_length=64)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('account', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='transactions', to='accounts.walletaccount')),
            ],
            options={'ordering': ('-created_at', '-id')},
        ),
    ]

