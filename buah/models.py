from django.db import models
from django_pandas.managers import DataFrameManager
# Create your models here.
import json

class Dataset(models.Model):
    filename = models.ImageField(upload_to="uploads")
    form_factor = models.FloatField(null=True, blank=True)
    aspect_ratio = models.FloatField(null=True, blank=True)
    rect = models.FloatField(null=True, blank=True)
    narrow_factor = models.FloatField(null=True, blank=True)
    prd = models.FloatField(null=True, blank=True)
    plw = models.FloatField(null=True, blank=True)
    mean_h = models.FloatField(null=True, blank=True)
    mean_s = models.FloatField(null=True, blank=True)
    mean_v = models.FloatField(null=True, blank=True)

    MALACCENSIS = 0
    SUBINTEGRA = 1

    KELAS_CHOICES = [
        (MALACCENSIS, 'MALACCENSIS'),
        (SUBINTEGRA, 'SUBINTEGRA'),
    ]
    kelas = models.IntegerField(null=True, blank=True, choices=KELAS_CHOICES)


class Model(models.Model):
    filename = models.FileField(upload_to="models")
    datalatih_ids = models.TextField(null=True, blank=True)
    datauji_ids = models.TextField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    epoch = models.IntegerField(null=True, blank=True, default=1)

    @property
    def jumlah_data_latih(self):
        json_datalatih_ids = json.loads(self.datalatih_ids)

        return  len(json_datalatih_ids)

    @property
    def jumlah_data_uji(self):
        json_datauji_ids = json.loads(self.datauji_ids)

        return  len(json_datauji_ids)




