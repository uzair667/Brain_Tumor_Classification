from django import forms

class user(forms.Form):
    name = forms.CharField(max_length=128)
    email = forms.EmailField(max_length=255)
    password = forms.CharField(max_length=128)
    text = forms.CharField(widget=forms.Textarea)

    
class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()