import streamlit as st
from transformers import pipeline

# Load Hugging Face Pipelines
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    ner_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    translation_models = {
        "Hindi": pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi"),
        "Tamil": pipeline("translation_en_to_ta", model="Helsinki-NLP/opus-mt-en-ta"),
        "French": pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr"),
    }
    return sentiment_model, ner_model, translation_models

sentiment_model, ner_model, translation_models = load_models()

# Streamlit UI
st.set_page_config(page_title="Mini NLP AI Assistant", layout="centered")
st.title("🧠 Mini NLP AI Assistant")
st.markdown("Enter an English sentence and get sentiment, named entities, and translation 🌍")

# Input Section
sentence = st.text_area("Enter a sentence in English:", height=100)

lang = st.selectbox("Choose target language for translation", ["Hindi", "Tamil", "French"])

if st.button("Analyze"):
    if sentence.strip() == "":
        st.warning("⚠️ Please enter a sentence.")
    else:
        with st.spinner("Analyzing..."):
            # Sentiment
            sentiment = sentiment_model(sentence)[0]
            st.subheader("🔍 Sentiment Analysis")
            st.write(f"**Label**: {sentiment['label']}")
            st.write(f"**Confidence**: {sentiment['score']:.2f}")

            # NER
            st.subheader("🧠 Named Entity Recognition")
            entities = ner_model(sentence)
            if entities:
                for ent in entities:
                    st.write(f"**{ent['entity_group']}** → {ent['word']} (score: {ent['score']:.2f})")
            else:
                st.info("No named entities found.")

            # Translation
            st.subheader("🌐 Translation")
            translated = translation_models[lang](sentence)
            st.write(f"**Translated to {lang}**: {translated[0]['translation_text']}")
