import streamlit as st
from transformers import pipeline

# Load models (no sentencepiece models)
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    ner_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    
    # Using MarianMT models (don't need sentencepiece)
    translation_models = {
        "Hindi": pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi"),
        "French": pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"),
        "Tamil": pipeline("translation", model="Helsinki-NLP/opus-mt-en-ta"),
    }
    
    return sentiment_model, ner_model, translation_models

sentiment_model, ner_model, translation_models = load_models()

# UI
st.set_page_config(page_title="Mini NLP Assistant", layout="centered")
st.title("🤖 Mini NLP AI Assistant")
st.markdown("Enter an English sentence to get Sentiment, Entities, and Translation.")

# Input
sentence = st.text_area("Type your sentence here:", height=100)
language = st.selectbox("Select language for translation", ["Hindi", "Tamil", "French"])

if st.button("Analyze"):
    if not sentence.strip():
        st.warning("Please enter a sentence.")
    else:
        with st.spinner("Analyzing..."):
            # Sentiment
            st.subheader("🧠 Sentiment Analysis")
            sentiment = sentiment_model(sentence)[0]
            st.success(f"{sentiment['label']} ({sentiment['score']:.2f})")

            # NER
            st.subheader("📍 Named Entity Recognition")
            entities = ner_model(sentence)
            if entities:
                for ent in entities:
                    st.write(f"{ent['entity_group']}: {ent['word']} ({ent['score']:.2f})")
            else:
                st.info("No named entities found.")

            # Translation
            st.subheader(f"🌐 Translated to {language}")
            translated = translation_models[language](sentence)
            st.write(translated[0]['translation_text'])
