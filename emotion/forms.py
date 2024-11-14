from django import forms

class EmotionForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea, label='Enter Text')
