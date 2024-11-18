import re
import random
import joblib

# Load the model
with open('model/text_emotion.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

emotion_labels = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise',
                  6: 'neutral', 7: 'disgust', 8: 'shame', 9: 'worry', 10: 'fun',
                  11: 'relief', 12: 'hate', 13: 'enthusiasm', 14: 'boredom'}

# Base responses for generic patterns
base_response_patterns = {
    r"do you like (.*)": [
        "Why do you ask if I like {}?",
        "What makes you think I like {}?",
        "Do you like {}?"
    ],
    r"i don't like (.*)": [
        "Why don’t you like {}?",
        "What is it about {} that you don't like?",
        "Have you always disliked {}?"
    ],
    r"i feel (.*)": [
        "Why do you feel {}?",
        "What makes you feel {}?",
        "How long have you felt {}?"
    ],
    r"i am (.*)": [
        "Why are you {}?",
        "What makes you {}?",
        "How long have you been {}?"
    ],
    r"i'm (.*)": [
        "Why are you {}?",
        "What makes you {}?",
        "How long have you been {}?"
    ],
    r"(.*) friend (.*)": [
        "Tell me more about your friends.",
        "How do your friends affect your feelings?",
        "Do you have a close friend?"
    ],
    r"yes": [
        "You seem quite sure.",
        "Okay, tell me more."
    ],
    r"no": [
        "Why not?",
        "Are you sure?"
    ],
    r"hello|hi|hey": [
        "Hello! How can I assist you today?",
        "Hi there! What's on your mind?",
        "Hey! How are you feeling today?"
    ],
    r"thank you|thanks": [
        "You're welcome!",
        "No problem! Anything else you want to talk about?"
    ],
    r"bye|goodbye": [
        "Goodbye! Have a great day!",
        "See you later! Take care!"
    ],
    # Add more patterns as needed
}

# Emotion-specific responses
emotion_response_patterns = {
    "sadness": {
        r"(.*) sad (.*)": [
            "I'm sorry to hear that. What's making you feel sad about {}?",
            "What happened that made you feel sad about {}?",
            "Can you tell me more about why you're sad about {}?"
        ],
        r"i feel (.*)": [
            "Why do you feel {}?",
            "What makes you feel {}?",
            "How long have you felt {}?"
        ],
        r"(.*) unhappy (.*)": [
            "What’s making you feel unhappy about {}?",
            "Why do you feel unhappy about {}?",
            "Can you share more about your unhappiness with {}?"
        ],
        # Add more patterns for sadness
    },
    "joy": {
        r"(.*) happy (.*)": [
            "I'm glad to hear that! What made you happy about {}?",
            "That’s great! What’s bringing you joy about {}?",
            "Why do you feel happy about {}?"
        ],
        r"i feel (.*)": [
            "Why do you feel {}?",
            "What makes you feel {}?",
            "How long have you felt {}?"
        ],
        r"(.*) joyful (.*)": [
            "What’s making you feel joyful about {}?",
            "Why do you feel joyful about {}?",
            "Can you share more about your joy with {}?"
        ],
        # Add more patterns for joy
    },
    # Define patterns for other emotions similarly
}


# Function to get a response based on a pattern match
def get_pattern_response(patterns, user_input):
    for pattern, responses in patterns.items():
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            return random.choice(responses).format(*match.groups())
    return None


# Function to get responses based on emotion
def get_emotion_response(emotion, user_input):
    if emotion in emotion_response_patterns:
        response = get_pattern_response(emotion_response_patterns[emotion], user_input)
        if response:
            return response
    return get_pattern_response(base_response_patterns,
                                user_input) or "I'm not sure how to respond to that. Can you tell me more?"


# Function to get base response
def get_base_response(user_input):
    return get_pattern_response(base_response_patterns,
                                user_input) or "I'm not sure how to respond to that. Can you tell me more?"


# Function to match user input to responses
def match_response(input_text, emotion_detected):
    if not emotion_detected:
        # Detect emotion based on initial input
        prediction = model.predict([input_text])[0]
        emotion = emotion_labels[prediction]
        return get_base_response(input_text), emotion
    else:
        # Continue responding based on detected emotion
        return get_emotion_response(emotion_detected, input_text), emotion_detected
