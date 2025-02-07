from django.conf import settings
from django.db import models
from django.utils import timezone


class CVmodel(models.Model):
    weight_file = models.FileField(upload_to='model_weights')
    # title = models.CharField(max_length=32)

class Book(models.Model):
    image = models.ImageField(upload_to='images',blank=True, null=True)
    cam_image = models.ImageField(upload_to='images',blank=True, null=True)
        