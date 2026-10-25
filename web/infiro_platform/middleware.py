class PrivateAssetMiddleware:
    """The generated app HTML must pass through the authenticated shell view."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.rstrip("/") == "/app/index.html":
            from django.shortcuts import redirect

            return redirect("/app")
        return self.get_response(request)


class PrivateResponseMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if request.path.startswith("/app"):
            response["Cache-Control"] = "private, no-store"
            response["Vary"] = "Cookie"
            response["Referrer-Policy"] = "same-origin"
            nonce = " 'nonce-" + request.csp_nonce + "'" if hasattr(request, "csp_nonce") else ""
            response["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'"
                + nonce
                + "; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            )
        return response
