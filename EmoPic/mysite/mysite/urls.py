"""
URL configuration for mysite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static
from .views import serve_image

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.dashboard_view, name="dashboard"),
    path("service/", views.service_view, name="service1"),
    path("service2/", views.service_select_view, name="service2"),
    path("service3/", views.service_result_view, name="service3"),
    path('screenshot/', views.screenshot_view, name='screenshot'),
    path('predict', views.predict_emotion, name='predict_emotion'),
    path('run_main/', views.run_main, name='run_main'),
    path('serve_image/<str:filename>/', serve_image, name='serve_image'),
    path("users/", include('users.urls')),
    path("upload_image/", views.upload_image, name='upload_image')
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)