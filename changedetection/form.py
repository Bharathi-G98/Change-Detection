from django import forms
from pagedown.widgets import PagedownWidget
from .models import Image

class ImageForm(forms.ModelForm):
    #cover= forms.CharField(widget = PagedownWidget())
    class Meta:
        model = Image
        fields = ('title','pc','video1','video2')

        widgets = {
            'title':forms.TextInput(attrs= {'class': 'title'}),
            'video1': forms.FileInput(attrs={'class':'video1'}),
            'video2': forms.FileInput(attrs={'class':'video2'}),
        }