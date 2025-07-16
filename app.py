import streamlit as st
import requests

# ✅ Your API Key
API_KEY = "pub_90a23434a6d64d68b1fe125a8fb5b3d7"

# 🌍 Country codes
COUNTRIES = {
    "India 🇮🇳": "in",
    "United States 🇺🇸": "us",
    "United Kingdom 🇬🇧": "gb",
    "Australia 🇦🇺": "au",
    "Canada 🇨🇦": "ca",
    "Germany 🇩🇪": "de",
    "France 🇫🇷": "fr",
    "Japan 🇯🇵": "jp",
    "Italy 🇮🇹": "it",
    "Russia 🇷🇺": "ru",
    "Brazil 🇧🇷": "br",
    "Spain 🇪🇸": "es",
    "South Korea 🇰🇷": "kr",
    "Singapore 🇸🇬": "sg",
    "Turkey 🇹🇷": "tr",
    "South Africa 🇿🇦": "za",
    "Mexico 🇲🇽": "mx",
    "Saudi Arabia 🇸🇦": "sa",
    "Nigeria 🇳🇬": "ng",
    "UAE 🇦🇪": "ae"
}

# 📂 News categories
CATEGORIES = [
    "top", "business", "entertainment", "environment", "food",
    "health", "politics", "science", "sports", "technology",
    "tourism", "world"
]

# 🔍 Fetching logic
def fetch_news(country_name, category):
    country_code = COUNTRIES[country_name]
    url = f"https://newsdata.io/api/1/news?apikey={API_KEY}&country={country_code}&category={category}&language=en"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "results" not in data or not data["results"]:
            return None, "⚠️ No news found for this category or country."

        return data["results"][:5], None  # return top 5
    except Exception as e:
        return None, f"❌ Error fetching news: {e}"

# 🌟 Streamlit UI
st.set_page_config(page_title="🗞️ Real-Time News App", layout="centered")
st.title("🗞️ Real-Time News App")
st.markdown("Get the latest headlines by country and category — powered by [NewsData.io](https://newsdata.io)")

# Dropdowns
country_choice = st.selectbox("🌍 Select Country", list(COUNTRIES.keys()), index=0)
category_choice = st.selectbox("🗂️ Select News Category", CATEGORIES, index=0)

# Fetch button
if st.button("🚀 Fetch News"):
    with st.spinner("Fetching news..."):
        news_list, error = fetch_news(country_choice, category_choice)

        if error:
            st.warning(error)
        else:
            for article in news_list:
                st.markdown(f"### 📰 {article['title']}")
                if article.get("description"):
                    st.markdown(article["description"])
                st.markdown(f"[🔗 Read more]({article['link']})")
                st.markdown("---")
