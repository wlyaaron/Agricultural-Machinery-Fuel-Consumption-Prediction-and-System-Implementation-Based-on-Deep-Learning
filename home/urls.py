from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    path('navigation/', views.navigation, name='navigation'),
    path('teaching/', views.teaching, name='teaching'),
    path('map/', views.map, name='map'),
    path('diabetes/', views.diabetes, name='diabetes'),
    path('about/', views.about, name='about'),
    path('predict/', views.predict, name='predict'),
    path('predict1/', views.predict1, name='predict1'),
    path('ct/', views.ct, name='ct'),
    path('history/', views.history, name='history'),
    path('history/show_data/', views.show_data_info, name='show_data_info'),
    path('coor/', views.coor, name='coor'),
    path('history/delete_data/', views.delete_data, name='delete_data'),
    path('history/detail_data/', views.detail_data, name='detail_data'),
    path('get_uploads/', views.get_uploads_with_times, name='get_uploads_with_times'),
    path('get_map_data/', views.get_map_data, name='get_map_data'),
    path('get_daily_upload_counts/', views.get_daily_upload_counts, name='get_daily_upload_counts'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
