from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse
from accounts.views import verify_clearance
from django.conf import settings
from django.conf.urls.static import static


def api_root(_request):
    return JsonResponse({
        'name': 'NYSC Clearance API',
        'status': 'ok',
        'version': getattr(settings, 'DEPLOYMENT_VERSION', ''),
        'docs': '/api/auth/config/',
    })


urlpatterns = [
    path('', api_root, name='api-root'),
    path('admin/', admin.site.urls),
    path('api/auth/', include('accounts.urls')),
    path('verify/', verify_clearance, name='verify-clearance'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
