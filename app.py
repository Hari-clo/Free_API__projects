import streamlit as st
import requests

st.set_page_config(page_title="DeepAI Text-to-Image", layout="centered")

st.title("🎨 DeepAI Text-to-Image Generator")
st.markdown("Turn words into art using [DeepAI's API](https://deepai.org/).")

# Text input
prompt = st.text_input("🖊️ Enter your prompt:", "a cyberpunk city at night with neon lights")

# API Key
api_key = st.text_input("🔐 Enter your DeepAI API Key:", type="password")

# Button
if st.button("🚀 Generate Image"):
    if not api_key:
        st.warning("Please provide your API key.")
    else:
        with st.spinner("Generating image..."):
            try:
                response = requests.post(
                    "https://api.deepai.org/api/text2img",
                    data={'text': prompt},
                    headers={'api-key': api_key}
                )
                result = response.json()

                if "output_url" in result:
                    st.image(result["output_url"], caption=prompt)
                    st.success("✅ Image generated successfully!")
                    st.markdown(f"[🖼️ View Full Image]({result['output_url']})")
                else:
                    st.error("❌ Failed to generate image.")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

st.markdown("""🔐 **Note:** This app uses [DeepAI](https://deepai.org/) API. Each user must use their own key.""")
