from django.urls import path, re_path

from . import views

urlpatterns = [
    path("healthz", views.health, name="health"),
    path("", views.page, name="home"),
    re_path(r"^(?P<path>.+)$", views.page, name="frontend-page"),
]
handler404 = "infiro_site.views.not_found"
