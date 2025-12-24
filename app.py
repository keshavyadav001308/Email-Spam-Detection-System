import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide"
)

# --------------------------------------------------
# Load NLP tools
# --------------------------------------------------
ps = PorterStemmer()
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# --------------------------------------------------
# Text Preprocessing Function (UNCHANGED LOGIC)
# --------------------------------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# --------------------------------------------------
# Load Model & Vectorizer
# --------------------------------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --------------------------------------------------
# SIDEBAR (MODEL DETAILS)
# --------------------------------------------------
st.sidebar.title("📊 Model Information")

st.sidebar.markdown("""
**Algorithm:** Random Forest (RF)  
**Accuracy:** 0.9868  
**Precision:** 0.9928  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Visual Insights")

show_wc = st.sidebar.checkbox("Show WordClouds")

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
st.markdown(
    "<h1 style='color:#1f77b4;'>📧 Email Spam Detection System</h1>",
    unsafe_allow_html=True
)

st.write("Classify an email as **Spam** or **Not Spam** using Machine Learning.")

# --------------------------------------------------
# Layout Columns
# --------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    input_sms = st.text_area(
        "✉️ Enter Email Content",
        height=200,
        placeholder="Paste your email text here..."
    )

    predict_btn = st.button("🚀 Check Spam")

with col2:
    st.markdown("### 📌 Prediction Result")

    if predict_btn:
        if input_sms.strip() == "":
            st.warning("Please enter email text.")
        else:
            # Preprocess
            transformed_sms = transform_text(input_sms)

            # Vectorize
            vector_input = tfidf.transform([transformed_sms])

            # Predict
            result = model.predict(vector_input)[0]

            # Display result
            if result == 1:
                st.error("🚨 Spam Email Detected!")
                st.markdown(
                    "**Warning:** This email is classified as **SPAM**. Avoid clicking suspicious links."
                )
            else:
                st.success("✅ Not Spam (Ham Email)")
                st.markdown(
                    "**Safe:** This email appears to be **legitimate**."
                )

# --------------------------------------------------
# WORDCLOUD SECTION
# --------------------------------------------------
if show_wc:
    st.markdown("---")
    st.markdown("## ☁️ Top Keywords Analysis")

    wc1, wc2 = st.columns(2)

    with wc1:
        st.markdown("### 🔴 Spam Emails")
        st.image("download (7).png", use_column_width=True)

    with wc2:
        st.markdown("### 🟢 Ham Emails")
        st.image("download (8).png", use_column_width=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>Built using NLP & Machine Learning | Random Forest Classifier</center>",
    unsafe_allow_html=True
)
