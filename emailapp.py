import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

st.set_page_config(page_title="HR Email Categorization", layout="wide")

# -------------------------------------
# ✅ BUILT-IN DATASET (No file needed)
# -------------------------------------
SAMPLE_DATA = [
    {
        "subject": "Salary increment not received",
        "body": "Hello HR, I haven't received my salary increment for June.",
        "category": "Payroll"
    },
    {
        "subject": "Laptop not working",
        "body": "My office laptop is not turning on, please help urgently.",
        "category": "IT Support"
    },
    {
        "subject": "Request for sick leave",
        "body": "I am not well and need 2 days of sick leave.",
        "category": "Leave Request"
    },
    {
        "subject": "Question about health insurance benefits",
        "body": "Can you explain what health benefits are available?",
        "category": "Employee Benefits"
    },
    {
        "subject": "Complaint about manager",
        "body": "I want to report workplace harassment from my team lead.",
        "category": "Grievance"
    },
]

# Convert to DataFrame
df = pd.DataFrame(SAMPLE_DATA)
df["text"] = df["subject"] + " " + df["body"]

# -------------------------------------
# Text Cleaning Function
# -------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    return " ".join(tokens)

df["cleaned"] = df["text"].apply(clean_text)

# -------------------------------------
# AUTO MODEL TRAINING
# -------------------------------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df["cleaned"])
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# -------------------------------------
# STREAMLIT UI
# -------------------------------------
st.title("📧 HR AUTOMATIC EMAIL CATEGORIZATION SYSTEM")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📮 Classify Email", "📁 Batch Prediction"])


# ------------------ TAB 1 - DASHBOARD ------------------
with tab1:
    st.header("📊 Data Dashboard & Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category Distribution")
        cat_count = df["category"].value_counts()
        fig, ax = plt.subplots(figsize=(5,3))
        sns.barplot(x=cat_count.values, y=cat_count.index, palette="viridis")
        st.pyplot(fig)

    with col2:
        st.subheader("Word Cloud")
        text = " ".join(df["cleaned"].tolist())
        wc = WordCloud(width=800, height=300, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(5,3))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    st.subheader("Top Words Per Category")
    for cat in df["category"].unique():
        st.markdown(f"### {cat}")
        words = " ".join(df[df["category"] == cat]["cleaned"]).split()
        common = Counter(words).most_common(5)
        st.write(dict(common))


# ------------------ TAB 2 - SINGLE EMAIL PREDICTION ------------------
with tab2:
    st.header("📮 Classify a Single Email")

    subject_in = st.text_input("Email Subject")
    body_in = st.text_area("Email Body")

    if st.button("Classify Email"):
        raw = subject_in + " " + body_in
        cleaned = clean_text(raw)
        X_infer = tfidf.transform([cleaned])
        pred = model.predict(X_infer)[0]
        st.success(f"Predicted Category: **{pred}**")


# ------------------ TAB 3 - BATCH PREDICTION ------------------
with tab3:
    st.header("📁 Upload File for Batch Prediction")

    uploaded = st.file_uploader("Upload CSV with columns: subject, body", type=["csv"])

    if uploaded:
        data = pd.read_csv(uploaded)

        if "subject" not in data.columns or "body" not in data.columns:
            st.error("CSV must contain 'subject' and 'body' columns!")
        else:
            data["text"] = data["subject"] + " " + data["body"]
            data["cleaned"] = data["text"].apply(clean_text)

            Xb = tfidf.transform(data["cleaned"])
            preds = model.predict(Xb)

            data["predicted_category"] = preds
            st.write(data)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
