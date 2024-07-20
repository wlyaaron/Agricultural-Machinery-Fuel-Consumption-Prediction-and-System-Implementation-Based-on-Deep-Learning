from django.db import models
import json

class Coordinate(models.Model):
    lng = models.FloatField()
    lat = models.FloatField()
    time = models.CharField(max_length=100)
    speed = models.FloatField()
    direction = models.FloatField()
    height = models.FloatField()
    did = models.CharField(max_length=100)
    flag = models.CharField(max_length=100)
    class Meta:
        managed = True
        db_table = 'home_coordinate'

class Upload(models.Model):
    created_at = models.DateTimeField(auto_now_add = True)#上传时间
    coordinates = models.JSONField()#经纬度json数据
    upload_count = models.PositiveIntegerField(default = 1)#上传次数
    file_name = models.CharField(max_length=100)
    time = models.JSONField()
    speed = models.JSONField()
    direction = models.JSONField()
    height = models.JSONField()
    did = models.JSONField()
    flag = models.JSONField()
    start_coor = models.JSONField()

    def __str__(self):
        return f"Upload {self.upload_count} at {self.created_at}"
    

class Filename(models.Model):
    name = models.CharField(max_length=100)
    class Meta:
        managed = True
        db_table = 'home_filename'


