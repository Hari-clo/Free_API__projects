import streamlit as st
import requests
from transformers import pipeline

# 🔄 Load NLP models once (cached)
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return sentiment, ner

sentiment_model, ner_model = load_models()

# 🌐 Translate using stable LibreTranslate mirror
def translate_text(text, lang_code):
    try:
        response = requests.post("https://translate.argosopentech.com/translate", data={
            "q": text,
            "source": "en",
            "target": lang_code,
            "format": "text"
        })
        return response.json().get("translatedText", "⚠️ Translation failed.")
    except Exception as e:
        return f"⚠️ Error: {e}"

# 🎨 Streamlit UI setup
st.set_page_config(page_title="GenAI NLP Assistant", layout="centered")
st.title("🤖 GenAI NLP Assistant")
st.markdown("Enter an English sentence to get:")
st.markdown("- 💬 Sentiment Analysis")
st.markdown("- 🧠 Named Entity Recognition")
st.markdown("- 🌍 Translation")

# ✍️ User input
text_input = st.text_area("📝 Enter your sentence:", height=120)
lang = st.selectbox("🌐 Translate to:", {"Hindi": "hi", "Tamil": "ta", "French": "fr"})

if st.button("Run NLP Tasks"):
    if not text_input.strip():
        st.warning("⚠️ Please enter a sentence.")
    else:
        # 👉 Optionally capitalize text for better NER
        cleaned_text = text_input.strip().capitalize()

        # 💬 Sentiment Analysis
        st.subheader("💬 Sentiment Analysis")
        sentiment = sentiment_model(cleaned_text)[0]
        st.write(f"**Label:** {sentiment['label']}")
        st.write(f"**Confidence:** {sentiment['score']:.2f}")

        # 🧠 Named Entity Recognition
        st.subheader("🧠 Named Entity Recognition")
        ner_results = ner_model(cleaned_text)
        if ner_results:
            for ent in ner_results:
                st.write(f"- **{ent['entity_group']}** → {ent['word']} ({ent['score']:.2f})")
        else:
            st.info("No named entities found.")

        # 🌍 Translation
        st.subheader(f"🌍 Translation to {lang}")
        translation = translate_text(text_input, lang)
        st.success(translation)
