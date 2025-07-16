import streamlit as st
from transformers import pipeline

# Load all models at once (cached)
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    translators = {
        "Hindi": pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi"),
        "Tamil": pipeline("translation_en_to_ta", model="Helsinki-NLP/opus-mt-en-ta"),
        "French": pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
    }
    return sentiment, ner, translators

sentiment_model, ner_model, translator_models = load_models()

# Streamlit UI
st.set_page_config(page_title="GenAI NLP Assistant", layout="centered")
st.title("🤖 GenAI NLP Assistant")
st.markdown("Perform 💬 Sentiment, 🧠 NER, and 🌍 Translation using Hugging Face models.")

# Input
text_input = st.text_area("📝 Enter your English sentence:", height=120)
target_lang = st.selectbox("🌐 Translate to:", ["Hindi", "Tamil", "French"])

if st.button("Run NLP Tasks"):
    if not text_input.strip():
        st.warning("⚠️ Please enter a sentence.")
    else:
        cleaned_text = text_input.strip().capitalize()

        # Sentiment Analysis
        st.subheader("💬 Sentiment Analysis")
        sentiment = sentiment_model(cleaned_text)[0]
        st.write(f"**Label:** {sentiment['label']}")
        st.write(f"**Confidence:** {sentiment['score']:.2f}")

        # NER
        st.subheader("🧠 Named Entity Recognition")
        ner_result = ner_model(cleaned_text)
        if ner_result:
            for ent in ner_result:
                st.write(f"- **{ent['entity_group']}** → {ent['word']} ({ent['score']:.2f})")
        else:
            st.info("No named entities found.")

        # Translation using Hugging Face
        st.subheader(f"🌍 Translation to {target_lang}")
        translator = translator_models[target_lang]
        translated = translator(cleaned_text)
        st.success(translated[0]['translation_text'])
