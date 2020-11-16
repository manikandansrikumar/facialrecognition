"""FaceRecognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from Faceapp.views import FaceRecognitionView, SavePhotoView
from webcam_face_recognize import RealTimeFaceRecognition

urlpatterns = [
    path('admin/', admin.site.urls),
    path('detect/face/',FaceRecognitionView.as_view()),
    path('save/photo/',SavePhotoView.as_view()),
    path('video/',RealTimeFaceRecognition.as_view())
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
