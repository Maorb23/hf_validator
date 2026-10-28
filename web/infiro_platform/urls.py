from django.urls import path, re_path

from . import views

urlpatterns = [
    path("/login", views.auth),
    path("/register", views.auth, {"action": "register"}),
    path("/logout", views.auth, {"action": "logout"}),
    path("/api/bootstrap", views.bootstrap),
    path("/api/connections", views.create_connection),
    path("/api/connections/<uuid:identifier>", views.connection_detail),
    path("/api/recommendations", views.create_recommendation),
    path("/api/models", views.select_direct_model),
    path("/api/recommendations/<uuid:identifier>", views.recommendation_detail),
    path("/api/datasets", views.create_dataset),
    path("/api/datasets/<uuid:identifier>", views.dataset_detail),
    path("/api/prepared", views.create_prepared),
    path("/api/prepared/<uuid:identifier>", views.prepared_detail),
    path("/api/prepared/<uuid:identifier>/request", views.edit_prepared_request),
    path("/api/jobs/<uuid:identifier>", views.job_detail),
    path("/api/jobs/<uuid:identifier>/trials", views.trial_detail),
    path("/api/jobs/<uuid:identifier>/diagnostics", views.study_diagnostics),
    path("/api/jobs/<uuid:identifier>/retry", views.retry_job),
    path("/api/artifacts/<uuid:identifier>", views.artifact_download),
    path("", views.shell),
    path("/", views.shell),
    re_path(
        r"^/(?P<path>overview|recommendations|datasets|studies|connections|configure|approval)(?:/[^/]+)?/?$",
        views.shell,
    ),
]
