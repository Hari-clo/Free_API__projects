import streamlit as st
from transformers import pipeline

# Load all models once
@st.cache_resource
def load_pipelines():
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    translate_hi = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")
    translate_fr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
    translate_ta = pipeline("translation_en_to_ta", model="Helsinki-NLP/opus-mt-en-ta")
    return sentiment, ner, {
        "Hindi": translate_hi,
        "French": translate_fr,
        "Tamil": translate_ta
    }

sentiment_pipeline, ner_pipeline, translators = load_pipelines()

# UI
st.set_page_config(page_title="Mini NLP Assistant", layout="centered")
st.title("🧠 Mini NLP AI Assistant")
st.markdown("Analyze a sentence for **Sentiment**, **Named Entities**, and **Translation**.")

sentence = st.text_area("Enter an English sentence:")

lang = st.selectbox("Choose translation language:", ["Hindi", "Tamil", "French"])

if st.button("Run Analysis"):
    if not sentence.strip():
        st.warning("Please enter a sentence.")
    else:
        st.subheader("📊 Sentiment Analysis")
        result = sentiment_pipeline(sentence)[0]
        st.write(f"**Label**: {result['label']}, **Score**: {result['score']:.2f}")

        st.subheader("🧾 Named Entity Recognition")
        entities = ner_pipeline(sentence)
        if entities:
            for e in entities:
                st.write(f"{e['entity_group']}: {e['word']} ({e['score']:.2f})")
        else:
            st.info("No entities found.")

        st.subheader("🌍 Translation")
        translated = translators[lang](sentence)[0]['translation_text']
        st.success(f"**{lang}**: {translated}")
