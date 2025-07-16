import streamlit as st
import requests
from transformers import pipeline

# Load models only once
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return sentiment, ner

sentiment_model, ner_model = load_models()

# Translation via LibreTranslate
def libre_translate(text, target_lang):
    try:
        url = "https://libretranslate.com/translate"
        payload = {
            "q": text,
            "source": "en",
            "target": target_lang,
            "format": "text"
        }
        response = requests.post(url, data=payload)
        return response.json().get('translatedText', '❌ Translation failed.')
    except:
        return "❌ API Error"

# UI setup
st.set_page_config(page_title="NLP Assistant", layout="centered")
st.title("🤖 Mini NLP AI Assistant")
st.markdown("Enter an English sentence to get Sentiment, Named Entities, and Translation 🌍")

# Input
sentence = st.text_area("✍️ Type your sentence in English:", height=100)
language = st.selectbox("🌐 Choose translation language:", {"Hindi": "hi", "Tamil": "ta", "French": "fr"})

if st.button("Analyze"):
    if sentence.strip() == "":
        st.warning("⚠️ Please enter a sentence.")
    else:
        # Sentiment Analysis
        st.subheader("🧠 Sentiment Analysis")
        sentiment = sentiment_model(sentence)[0]
        st.write(f"**Label:** {sentiment['label']}, **Confidence:** {sentiment['score']:.2f}")

        # NER
        st.subheader("📍 Named Entity Recognition")
        entities = ner_model(sentence)
        if entities:
            for e in entities:
                st.write(f"- **{e['entity_group']}**: {e['word']} ({e['score']:.2f})")
        else:
            st.info("No named entities found.")

        # Translation
        st.subheader(f"🌍 Translation to {language}")
        translation = libre_translate(sentence, language)
        st.success(translation)
