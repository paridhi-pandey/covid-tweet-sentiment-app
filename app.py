import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import plotly.express as px

# --- Setup ---
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Load model/vectorizer ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))

# --- Preprocessing ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- Sentiment label to emoji ---
def format_sentiment(sentiment):
    emojis = {
        'Positive': '😊',
        'Negative': '😞',
        'Neutral': '😐'
    }
    return f"{emojis.get(sentiment, '')} {sentiment}"

# --- Streamlit Layout ---
st.sidebar.markdown("## 👤 About")
st.sidebar.markdown("""
**Author:** *Paridhi*  
**Project:** COVID-19 Tweet Sentiment Classifier  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("💡 Use the tabs to analyze single tweets or upload a CSV for batch prediction.")


tab1, tab2 = st.tabs(["🔍 Predict Single Tweet", "📊 Batch Prediction (CSV)"])

# --- Tab 1: Single Tweet ---
with tab1:
    st.title("🧠 Tweet Sentiment Classifier")
    st.markdown("Enter a tweet below or choose a sample to predict its **COVID-19 sentiment**.")

    # --- Sample Tweets ---
sample_tweets = {
    "Sample 1": "The government needs to do more about this pandemic!",
    "Sample 2": "Feeling blessed to work from home during COVID times.",
    "Sample 3": "I'm tired of hearing about COVID-19 all the time.",
}

st.sidebar.markdown("## 🎯 Try Sample Tweets")
for label, tweet in sample_tweets.items():
    if st.sidebar.button(label):
        st.session_state.tweet_input = tweet

# --- Main Input ---
tweet_input = st.text_area("📝 Enter a Tweet", value=st.session_state.get("tweet_input", ""), height=150)



if st.button("🔮 Predict Sentiment"):
        if tweet_input.strip() == "":
            st.warning("Please enter or select a tweet.")
        else:
            clean_text = preprocess(tweet_input)
            vectorized_input = vectorizer.transform([clean_text])
            prediction = model.predict(vectorized_input)[0]
            st.success(f"🎯 Predicted Sentiment: **{format_sentiment(prediction)}**")

# --- Tab 2: Batch CSV Upload ---
with tab2:
    st.header("📤 Upload CSV for Bulk Prediction")
    st.write("Upload a CSV file with a column named `tweet`.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

        if 'tweet' not in df.columns:
            st.error("CSV must contain a 'tweet' column.")
        else:
            df['clean_text'] = df['tweet'].astype(str).apply(preprocess)
            X_vec = vectorizer.transform(df['clean_text'])
            df['predicted_sentiment'] = model.predict(X_vec)
            df['formatted_sentiment'] = df['predicted_sentiment'].apply(format_sentiment)

            # Sentiment filter
            sentiments = df['predicted_sentiment'].unique().tolist()
            selected = st.selectbox("🔎 Filter by Sentiment", ["All"] + sentiments)
            if selected != "All":
                filtered_df = df[df['predicted_sentiment'] == selected]
            else:
                filtered_df = df

            # Color tags for better visual output
            def color_tag(sentiment):
                colors = {
                    'Positive': 'green',
                    'Negative': 'red',
                    'Neutral': 'gray'
                }
                return f"<span style='color:{colors.get(sentiment, 'black')}'>{sentiment}</span>"

            st.markdown("### 📋 Predictions")
            styled_df = filtered_df[['tweet', 'predicted_sentiment']].copy()
            styled_df['predicted_sentiment'] = styled_df['predicted_sentiment'].apply(
                lambda x: color_tag(x))
            st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

            # Sentiment Distribution Pie Chart
            st.markdown("### 📊 Sentiment Distribution")
            sentiment_counts = df['predicted_sentiment'].value_counts()
            st.plotly_chart(px.pie(
                names=sentiment_counts.index,
                values=sentiment_counts.values,
                title="Sentiment Breakdown",
                color_discrete_sequence=px.colors.qualitative.Set3
            ))

            # Download Results
            st.download_button("⬇️ Download Results as CSV",
                               df.to_csv(index=False),
                               file_name="predicted_sentiments.csv",
                               mime="text/csv")

            # Reset Button
            if st.button("🔄 Reset"):
                st.experimental_rerun()

