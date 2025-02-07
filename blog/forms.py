from django import forms
from .models import Book, CVmodel

class ImageForm(forms.ModelForm):
    class Meta:
        model = Book
        fields = ['image']

class WeightsForm(forms.ModelForm):
    class Meta:
        model = CVmodel
        fields = ["weight_file"]#, "title"]
