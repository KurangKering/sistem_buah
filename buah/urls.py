from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='buah/index'),
    path('data_master/', views.data_master, name='buah/data_master'),
    path('tambah_data_master/', views.tambah_data_master, name='buah/tambah_data_master'),
    path('proses_tambah_data_master/', views.proses_tambah_data_master, name='buah/proses_tambah_data_master'),
    path('hapus_data/', views.hapus_data, name='buah/hapus_data'),
    path('detail_gambar/', views.detail_gambar, name='buah/detail_gambar'),
    path('download_csv/', views.download_csv, name='buah/download_csv'),
    path('data_anfis/', views.data_anfis, name='buah/data_anfis'),
    path('tambah_model/', views.tambah_model, name='buah/tambah_model'),
    path('proses_pelatihan/', views.proses_pelatihan, name='buah/proses_pelatihan'),
    path('hapus_anfis/', views.hapus_anfis, name='buah/hapus_anfis'),
    path('lihat_anfis/<int:model_id>/', views.lihat_anfis, name='buah/lihat_anfis'),
    path('pengujian/', views.pengujian, name='buah/pengujian'),
    path('proses_pengujian/', views.proses_pengujian, name='buah/proses_pengujian'),
]
