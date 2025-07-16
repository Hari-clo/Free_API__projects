import streamlit as st
import requests
from transformers import pipeline

# Load Hugging Face models (cached to avoid reloading)
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return sentiment, ner

sentiment_model, ner_model = load_models()

# Translation using LibreTranslate API (no key)
def translate_text(text, lang_code):
    try:
        response = requests.post("https://libretranslate.com/translate", data={
            "q": text,
            "source": "en",
            "target": lang_code,
            "format": "text"
        })
        return response.json().get("translatedText", "⚠️ Translation failed.")
    except Exception as e:
        return f"⚠️ Error: {e}"

# Streamlit UI
st.set_page_config(page_title="GenAI NLP Assistant", layout="centered")
st.title("🤖 GenAI NLP Assistant")
st.markdown("Perform Sentiment Analysis, Named Entity Recognition, and Translation 🌐")

text_input = st.text_area("📝 Enter an English sentence:")
lang = st.selectbox("🌍 Translate to:", {"Hindi": "hi", "Tamil": "ta", "French": "fr"})

if st.button("Run NLP"):
    if not text_input.strip():
        st.warning("Please enter a sentence.")
    else:
        # Sentiment
        st.subheader("💬 Sentiment Analysis")
        sentiment = sentiment_model(text_input)[0]
        st.write(f"**Label:** {sentiment['label']}")
        st.write(f"**Confidence:** {sentiment['score']:.2f}")

        # NER
        st.subheader("🧠 Named Entity Recognition")
        entities = ner_model(text_input)
        if entities:
            for e in entities:
                st.write(f"- **{e['entity_group']}**: {e['word']} ({e['score']:.2f})")
        else:
            st.info("No named entities found.")

        # Translation
        st.subheader(f"🌐 Translation ({lang})")
        translated = translate_text(text_input, lang)
        st.success(translated)
