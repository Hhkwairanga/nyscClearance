from typing import Optional
from django.conf import settings
from django.core import signing


SIGNER_SALT = 'accounts.email.verification'


def generate_email_token(user_id: int) -> str:
    return signing.dumps({'uid': user_id}, salt=SIGNER_SALT)


def validate_email_token(token: str, max_age_seconds: int = 60 * 60 * 24) -> Optional[int]:
    try:
        data = signing.loads(token, salt=SIGNER_SALT, max_age=max_age_seconds)
        return int(data['uid'])
    except Exception:
        return None
