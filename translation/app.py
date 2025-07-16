import streamlit as st
import requests
from transformers import pipeline

# Load Hugging Face models
@st.cache_resource
def load_models():
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    ner_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return sentiment_pipe, ner_pipe

sentiment_model, ner_model = load_models()

# LibreTranslate API for translation
def translate_libre(text, target_lang):
    try:
        response = requests.post(
            "https://libretranslate.com/translate",
            data={
                "q": text,
                "source": "en",
                "target": target_lang,
                "format": "text"
            }
        )
        result = response.json()
        return result.get("translatedText", "⚠️ Translation failed.")
    except Exception as e:
        return f"⚠️ Error: {e}"

# Streamlit UI
st.set_page_config(page_title="GenAI NLP Assistant", layout="centered")
st.title("🤖 GenAI NLP Assistant")
st.markdown("Enter an English sentence to perform:")
st.markdown("- Sentiment Analysis 💬")
st.markdown("- Named Entity Recognition 🧠")
st.markdown("- Translation 🌍")

sentence = st.text_area("✍️ Input Sentence", height=120)
target_lang = st.selectbox("🌐 Translate To:", {"Hindi": "hi", "Tamil": "ta", "French": "fr"})

if st.button("Generate Results"):
    if not sentence.strip():
        st.warning("Please enter a valid sentence.")
    else:
        # Sentiment
        st.subheader("💬 Sentiment Analysis")
        sentiment_result = sentiment_model(sentence)[0]
        st.write(f"**Label**: {sentiment_result['label']}, **Confidence**: {sentiment_result['score']:.2f}")

        # NER
        st.subheader("🧠 Named Entity Recognition")
        ner_results = ner_model(sentence)
        if ner_results:
            for ent in ner_results:
                st.write(f"• **{ent['entity_group']}** → {ent['word']} ({ent['score']:.2f})")
        else:
            st.info("No named entities detected.")

        # Translation
        st.subheader(f"🌍 Translated to {target_lang}")
        translation = translate_libre(sentence, target_lang)
        st.success(translation)
