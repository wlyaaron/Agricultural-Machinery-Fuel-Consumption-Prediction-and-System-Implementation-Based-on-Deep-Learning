from django.forms import ModelForm
from .models import Coordinate

class CoordinateForm(ModelForm):
    class Meta:
        model = Coordinate
        fields = [
            'lng',
            'lat',
            'time',
            'speed', 
            'direction',
            'height',
            'did',
            'flag',
        ]