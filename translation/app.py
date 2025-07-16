import streamlit as st
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# Load Sentiment & NER pipelines
@st.cache_resource
def load_sentiment_ner():
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return sentiment, ner

# Load NLLB translation model
@st.cache_resource
def load_translation_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = pipeline("translation", model=model, tokenizer=tokenizer)
    return translator

# Language Code Mapping for NLLB
LANG_CODES = {
    "Hindi": "hin_Deva",
    "Tamil": "tam_Taml",
    "French": "fra_Latn"
}

# Load models
sentiment_pipeline, ner_pipeline = load_sentiment_ner()
translator_pipeline = load_translation_model()

# UI Setup
st.set_page_config(page_title="GenAI NLP Assistant", layout="centered")
st.title("🤖 GenAI NLP Assistant")
st.markdown("Analyze your sentence with:")
st.markdown("- 💬 Sentiment Analysis")
st.markdown("- 🧠 Named Entity Recognition")
st.markdown("- 🌍 Translation")

# User Input
text_input = st.text_area("📝 Enter your English sentence:", height=120)
target_lang = st.selectbox("🌐 Translate to:", ["Hindi", "Tamil", "French"])

if st.button("Run NLP Tasks"):
    if not text_input.strip():
        st.warning("⚠️ Please enter a sentence.")
    else:
        cleaned_text = text_input.strip().capitalize()

        # 🔹 Sentiment Analysis
        st.subheader("💬 Sentiment Analysis")
        sentiment = sentiment_pipeline(cleaned_text)[0]
        st.write(f"**Label:** {sentiment['label']}")
        st.write(f"**Confidence:** {sentiment['score']:.2f}")

        # 🔹 Named Entity Recognition
        st.subheader("🧠 Named Entity Recognition")
        ner_results = ner_pipeline(cleaned_text)
        if ner_results:
            for ent in ner_results:
                st.write(f"- **{ent['entity_group']}** → {ent['word']} ({ent['score']:.2f})")
        else:
            st.info("No named entities found.")

        # 🔹 Translation using NLLB
        st.subheader(f"🌍 Translation to {target_lang}")
        lang_code = LANG_CODES[target_lang]
        try:
            translated = translator_pipeline(
                cleaned_text,
                src_lang="eng_Latn",
                tgt_lang=lang_code,
                max_length=200
            )
            st.success(translated[0]['translation_text'])
        except Exception as e:
            st.error(f"❌ Translation failed: {str(e)}")
