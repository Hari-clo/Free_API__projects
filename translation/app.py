import streamlit as st
from transformers import pipeline

# Load Sentiment & NER
@st.cache_resource
def load_sentiment_ner():
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return sentiment, ner

# Load Translation Models
@st.cache_resource
def load_translators():
    tamil_translator = pipeline("translation", model="ai4bharat/indic-trans-translation-en-ta")
    hindi_translator = pipeline("translation", model="ai4bharat/indic-trans-translation-en-hi")
    french_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ROMANCE")
    return {
        "Tamil": tamil_translator,
        "Hindi": hindi_translator,
        "French": french_translator
    }

# Load models
sentiment_model, ner_model = load_sentiment_ner()
translation_models = load_translators()

# UI Setup
st.set_page_config(page_title="GenAI NLP Assistant", layout="centered")
st.title("🤖 GenAI NLP Assistant")
st.markdown("Perform 💬 Sentiment, 🧠 Named Entity Recognition, and 🌍 Translation")

# User Input
text_input = st.text_area("📝 Enter an English sentence:", height=120)
target_lang = st.selectbox("🌐 Translate to:", ["Hindi", "Tamil", "French"])

if st.button("Run NLP Tasks"):
    if not text_input.strip():
        st.warning("Please enter a sentence.")
    else:
        cleaned_text = text_input.strip().capitalize()

        # 💬 Sentiment
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
        st.subheader(f"🌍 Translation to {target_lang}")
        try:
            translator = translation_models[target_lang]
            translated = translator(cleaned_text)
            st.success(translated[0]['translation_text'])
        except Exception as e:
            st.error(f"❌ Translation failed: {e}")
