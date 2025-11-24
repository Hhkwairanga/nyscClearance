from .models import PaystackConfig


def get_paystack_keys():
    cfg = PaystackConfig.objects.filter(is_active=True).order_by('-updated_at').first()
    if not cfg or not cfg.public_key or not cfg.secret_key:
        raise ValueError('Paystack keys missing or inactive')
    return {
        'public_key': cfg.public_key,
        'secret_key': cfg.secret_key,
        'webhook_secret': cfg.webhook_secret or '',
    }

