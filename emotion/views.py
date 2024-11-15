from django.shortcuts import render
from .forms import EmotionForm
import joblib
import numpy as np

with open('model/text_emotion.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

emotion_labels = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise',
                  6: 'neutral', 7: 'disgust', 8: 'shame', 9: 'worry', 10: 'fun',
                  11: 'relief', 12: 'hate', 13: 'enthusiasm', 14: 'boredom'}

emotion_colors = {
    'sadness': '#1f77b4',
    'joy': '#ff7f0e',
    'love': '#2ca02c',
    'anger': '#d62728',
    'fear': '#9467bd',
    'surprise': '#8c564b',
    'neutral': '#7f7f7f',
    'disgust': '#bcbd22',
    'shame': '#17becf',
    'worry': '#e377c2',
    'fun': '#ffbb78',
    'relief': '#98df8a',
    'hate': '#c49c94',
    'enthusiasm': '#f7b6d2',
    'boredom': '#c5b0d5'
}

def predict_emotion(request):
    if request.method == 'POST':
        form = EmotionForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            prediction = model.predict([text])[0]
            probabilities = model.predict_proba([text])[0]
            label = emotion_labels[prediction]
            confidence = round(probabilities[prediction] * 100, 2)
            prob_dict = {emotion_labels[i]: round(prob, 2) for i, prob in enumerate(probabilities)}
            return render(request, 'emotion_form.html', {'form': form, 'label': label, 'probabilities': prob_dict, 'confidence': confidence, 'color': emotion_colors[label]})
    else:
        form = EmotionForm()
    return render(request, 'emotion_form.html', {'form': form})
