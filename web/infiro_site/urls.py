from django.urls import include, path, re_path

from . import views

urlpatterns = [
    path("app", include("infiro_platform.urls")),
    path("healthz", views.health, name="health"),
    path("", views.page, name="home"),
    re_path(r"^(?P<path>.+)$", views.page, name="frontend-page"),
]
handler404 = "infiro_site.views.not_found"
