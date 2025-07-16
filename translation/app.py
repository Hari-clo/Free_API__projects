import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load sentiment & NER models (unchanged)
@st.cache_resource
def load_sentiment_ner():
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return sentiment, ner

# Load multilingual translator (NLLB model)
@st.cache_resource
def load_translator():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer)

# Language code mapping (NLLB uses ISO 639-3)
LANG_CODES = {
    "Hindi": "hin_Deva",
    "Tamil": "tam_Taml",
    "French": "fra_Latn"
}

sentiment_model, ner_model = load_sentiment_ner()
translation_pipeline = load_translator()

# Streamlit UI
st.set_page_config(page_title="GenAI NLP Assistant", layout="centered")
st.title("🤖 GenAI NLP Assistant")
st.markdown("Perform 💬 Sentiment, 🧠 NER, and 🌍 Translation (without failures)")

text_input = st.text_area("📝 Enter your English sentence:", height=120)
target_lang = st.selectbox("🌐 Translate to:", ["Hindi", "Tamil", "French"])

if st.button("Run NLP Tasks"):
    if not text_input.strip():
        st.warning("⚠️ Please enter a sentence.")
    else:
        cleaned_text = text_input.strip().capitalize()

        # Sentiment
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

        # Translation
        st.subheader(f"🌍 Translation to {target_lang}")
        lang_code = LANG_CODES[target_lang]
        translated = translation_pipeline(cleaned_text, src_lang="eng_Latn", tgt_lang=lang_code)
        st.success(translated[0]['translation_text'])
