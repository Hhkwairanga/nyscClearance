from django.core import signing
from django.contrib.auth import get_user_model
from rest_framework.authentication import BaseAuthentication, get_authorization_header
from rest_framework import exceptions


JWT_SALT = 'accounts.jwt'


def make_access_token(user_id: int) -> str:
    """Create a signed access token carrying the user id.

    Uses Django's signing for HMAC-based tamper-proof tokens.
    Expiry enforced during verification via max_age in loads().
    """
    return signing.dumps({'uid': int(user_id)}, salt=JWT_SALT)


class SignedTokenAuthentication(BaseAuthentication):
    """Authenticate via Authorization: Bearer <token> using Django signing.

    Token validity window is enforced at verification time (default 24h).
    """

    keyword = b'bearer'
    max_age_seconds = 60 * 60 * 24  # 24 hours

    def authenticate(self, request):
        auth = get_authorization_header(request).split()
        if not auth or auth[0].lower() != self.keyword:
            return None
        if len(auth) != 2:
            raise exceptions.AuthenticationFailed('Invalid authorization header')
        try:
            token = auth[1].decode('utf-8')
        except Exception:
            raise exceptions.AuthenticationFailed('Invalid authorization header')

        try:
            data = signing.loads(token, salt=JWT_SALT, max_age=self.max_age_seconds)
        except signing.SignatureExpired:
            raise exceptions.AuthenticationFailed('Token expired')
        except Exception:
            raise exceptions.AuthenticationFailed('Invalid token')

        uid = data.get('uid')
        if not uid:
            raise exceptions.AuthenticationFailed('Invalid token payload')
        User = get_user_model()
        try:
            user = User.objects.get(id=uid)
        except User.DoesNotExist:
            raise exceptions.AuthenticationFailed('User not found')
        return (user, None)

