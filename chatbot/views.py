from django.shortcuts import render
from emotion.forms import EmotionForm
from .eliza import match_response
import joblib
import random

# Load the model
with open('model/text_emotion.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

emotion_labels = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise',
                  6: 'neutral', 7: 'disgust', 8: 'shame', 9: 'worry', 10: 'fun',
                  11: 'relief', 12: 'hate', 13: 'enthusiasm', 14: 'boredom'}

# Initial ELIZA prompts
initial_eliza_prompts = [
    "Hello! How can I assist you today?",
    "Hi there! How are you feeling today?",
    "Hello! What's on your mind?",
]

def chat_view(request):
    if request.method == 'POST':
        form = EmotionForm(request.POST)
        if form.is_valid():
            user_input = form.cleaned_data['text']
            chat_history = request.session.get('chat_history', [])
            emotion_detected = request.session.get('emotion_detected', None)

            response_text, emotion_detected = match_response(user_input, emotion_detected)

            chat_history.append(f"You: {user_input}")
            chat_history.append(f"Eliza: {response_text}")

            request.session['chat_history'] = chat_history
            request.session['emotion_detected'] = emotion_detected

            return render(request, 'chatbot.html', {
                'form': form,
                'chat_history': chat_history
            })
    else:
        form = EmotionForm()
        request.session['chat_history'] = []
        request.session['emotion_detected'] = None

        initial_prompt = random.choice(initial_eliza_prompts)
        chat_history = [f"Eliza: {initial_prompt}"]
        request.session['chat_history'] = chat_history

    return render(request, 'chatbot.html', {'form': form, 'chat_history': chat_history})
