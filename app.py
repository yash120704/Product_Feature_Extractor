import streamlit as st
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# Load FLAN-T5 model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_model()

# Extract product features
def extract_features(review_text):
    prompt = f"Extract key product features mentioned in this review:\n\nReview: {review_text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Clean and split keywords
def extract_keywords(text):
    return [word.strip().lower() for word in re.split(',|;|\\n', text) if word.strip()]

# Set Streamlit page config
st.set_page_config(page_title="Product Feature Extractor", layout="centered")

# Inject Gradient Background and Custom Styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #ffe0cc, #e6ccff);
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .title-box {
        font-size: 2.4rem;
        font-weight: bold;
        color: #cc3366;
        background-color: #fff0f5;
        border-radius: 20px;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .footer {
        margin-top: 4rem;
        text-align: center;
        font-size: 0.9rem;
        color: #cc3366;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title-box">📊 Product Feature Extractor</div>', unsafe_allow_html=True)
st.write("Upload a CSV file with product reviews to extract key features automatically.")

# File Upload
uploaded_file = st.file_uploader("📄 Upload your CSV file (must have a review column)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Select review column
    review_col = st.selectbox("🧾 Select the column containing product reviews", df.columns.tolist())

    if review_col:
        with st.spinner("🔍 Extracting features..."):
            df['Extracted Features'] = df[review_col].apply(lambda x: extract_features(str(x)))

        st.success("✅ Feature extraction complete!")
        st.dataframe(df[[review_col, 'Extracted Features']], use_container_width=True)

        st.download_button("📥 Download Results", df.to_csv(index=False), file_name="product_features.csv")

        # Frequency Chart
        all_keywords = []
        for features in df['Extracted Features']:
            all_keywords.extend(extract_keywords(features))

        if all_keywords:
            st.markdown("### 🔢 Most Mentioned Features")
            freq_df = pd.DataFrame(Counter(all_keywords).most_common(10), columns=["Feature", "Frequency"])
            st.bar_chart(freq_df.set_index("Feature"))

# Footer
st.markdown("""
<div class="footer">
    🚀 Made with ❤️ using FLAN-T5, IBM Watsonx, and Streamlit<br>
    – by Yash Kashyap
</div>
""", unsafe_allow_html=True)
