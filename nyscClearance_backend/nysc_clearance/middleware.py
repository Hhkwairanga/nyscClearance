class NoCacheApiAuthErrorMiddleware:
    """Prevent browsers/proxies from caching API auth and error responses."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        path = getattr(request, 'path', '') or ''
        if path.startswith('/api/') and (response.status_code >= 400 or path.startswith('/api/auth/')):
            response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response['Pragma'] = 'no-cache'
            response['Expires'] = '0'
        return response
