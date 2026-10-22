from django.conf import settings
from django.http import FileResponse, HttpResponse, JsonResponse
from django.views.decorators.http import require_safe

from .frontend import resolve_built_path


def _unavailable():
    return HttpResponse("Frontend build unavailable. Run npm run build in web/.", status=503, content_type="text/plain; charset=utf-8", headers={"Cache-Control": "no-store", "Retry-After": "30"})


@require_safe
def page(request, path=""):
    if not (settings.FRONTEND_DIST / "index.html").is_file():
        return _unavailable()
    target = resolve_built_path(settings.FRONTEND_DIST, "/" + path)
    # WhiteNoise owns static assets; missing files never become an HTML fallback.
    if target is None or target.suffix != ".html":
        return not_found(request)
    return FileResponse(target.open("rb"), content_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"})


def not_found(request, exception=None):
    target = resolve_built_path(settings.FRONTEND_DIST, "/404.html")
    if target:
        return FileResponse(target.open("rb"), status=404, content_type="text/html; charset=utf-8", headers={"Cache-Control": "no-store"})
    return HttpResponse("Page not found.", status=404, content_type="text/plain; charset=utf-8")


@require_safe
def health(request):
    ready = (settings.FRONTEND_DIST / "index.html").is_file() and (settings.FRONTEND_DIST / "_astro").is_dir()
    return JsonResponse({"status": "ok" if ready else "frontend_unavailable"}, status=200 if ready else 503, headers={"Cache-Control": "no-store"})
