from django.urls import path,include
from test_app import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('',views.hello,name='test'),
    path('test/', views.test, name='test'),
    path('check/', views.check, name='check'),
    # path('upload/', views.upload, name='upload'),
    # path('forms/',views.form,name='forms'),

] +static(settings.MEDIA_URL, document_root = settings.MEDIA_DIR)