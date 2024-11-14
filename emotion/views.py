from django.shortcuts import render
from .forms import EmotionForm
import joblib

# Load the trained model
with open('model/text_emotion.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

emotion_labels = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise',
                6: 'neutral', 7: 'disgust', 8: 'shame', 9: 'worry', 10: 'fun',
                11: 'relief', 12: 'hate', 13: 'enthusiasm', 14: 'boredom'}


def predict_emotion(request):
    if request.method == 'POST':
        form = EmotionForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            prediction = model.predict([text])[0]
            label = emotion_labels[prediction]
            return render(request, 'emotion_form.html', {'form': form, 'label': label})
    else:
        form = EmotionForm()
    return render(request, 'emotion_form.html', {'form': form})
