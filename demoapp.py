import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

pipe_rf = joblib.load(open("model/text_emotion.pkl", "rb"))



emotion_dict = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise',
                6: 'neutral', 7: 'disgust', 8: 'shame', 9: 'worry', 10: 'fun',
                11: 'relief', 12: 'hate', 13: 'enthusiasm', 14: 'boredom'}


def predict_emotion(docx):
    result = pipe_rf.predict([docx])
    return emotion_dict[result[0]]


def get_prediction_prob(docx):
    result = pipe_rf.predict_proba([docx])
    return result


def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotion In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Enter Your Text Here")
        submit_text = st.form_submit_button(label='Submit')
        if submit_text:
            col1, col2 = st.columns(2)
            prediction = predict_emotion(raw_text)
            probability = get_prediction_prob(raw_text)
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                st.write(f"Emotion: {prediction}")
                st.write(f"Probability: {np.max(probability):.2f}")

            with col2:
                st.success("Prediction Probabilities")
                proba_df = pd.DataFrame(probability, columns=pipe_rf.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probabilities"]
                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x="emotions:N",
                    y="probabilities:Q",
                    color="emotions:N"
                )
                st.altair_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
