import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

st.set_page_config(page_title="HR Email Categorization", layout="wide")

# ---------------------------------------------------------
# ✅ NEW: Larger Built-in Dataset (15 Samples, 3 per class)
# ---------------------------------------------------------

SAMPLE_DATA = [
    # Payroll
    {"subject": "Salary increment not received", "body": "My increment is missing for June", "category": "Payroll"},
    {"subject": "Salary delay", "body": "My salary has not come this month", "category": "Payroll"},
    {"subject": "Bonus query", "body": "I want to know my annual bonus details", "category": "Payroll"},

    # IT Support
    {"subject": "Laptop not working", "body": "Laptop suddenly shut down and won't turn on", "category": "IT Support"},
    {"subject": "VPN not connecting", "body": "VPN blocked, unable to access office network", "category": "IT Support"},
    {"subject": "Email login issue", "body": "Unable to login into office email", "category": "IT Support"},

    # Leave Request
    {"subject": "Sick leave request", "body": "I need 2 days sick leave", "category": "Leave"},
    {"subject": "Casual leave", "body": "Please approve my casual leave for tomorrow", "category": "Leave"},
    {"subject": "Emergency leave", "body": "Urgent family issue, need emergency leave", "category": "Leave"},

    # Benefits
    {"subject": "Insurance details", "body": "Please share medical insurance coverage", "category": "Benefits"},
    {"subject": "PF withdrawal process", "body": "How can I withdraw my PF amount?", "category": "Benefits"},
    {"subject": "Health benefits", "body": "Need details of employee health benefits", "category": "Benefits"},

    # Grievance
    {"subject": "Harassment complaint", "body": "I want to report workplace harassment", "category": "Grievance"},
    {"subject": "Manager misbehaving", "body": "My manager is constantly shouting unnecessarily", "category": "Grievance"},
    {"subject": "Team conflict", "body": "Facing conflict with teammates, need HR support", "category": "Grievance"},
]

df = pd.DataFrame(SAMPLE_DATA)
df["text"] = df["subject"] + " " + df["body"]


# ---------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# ALWAYS Safe Train/Test Split
# ---------------------------------------------------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df["cleaned"])
y = df["category"]

# ❗ No stratify → always safe
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("📧 HR Automatic Email Categorization System (Cloud-Ready)")


tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 Classify Email", "📁 Batch Prediction"])

# ------------------------ TAB 1 -------------------------
with tab1:
    st.header("📊 Data Insights & Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category Distribution")
        counts = df["category"].value_counts()
        fig, ax = plt.subplots(figsize=(5,3))
        sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
        st.pyplot(fig)

    with col2:
        st.subheader("Word Cloud")
        text = " ".join(df["cleaned"])
        wc = WordCloud(width=800, height=300, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(5,3))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    st.subheader("Top Keywords Per Category")
    for cat in df["category"].unique():
        st.markdown(f"### {cat}")
        words = " ".join(df[df["category"] == cat]["cleaned"]).split()
        st.write(dict(Counter(words).most_common(5)))


# ------------------------ TAB 2 -------------------------
with tab2:
    st.header("💬 Classify a Single Email")

    s = st.text_input("Subject")
    b = st.text_area("Body")

    if st.button("Predict Category"):
        raw = s + " " + b
        cleaned = clean_text(raw)
        X_infer = tfidf.transform([cleaned])
        pred = model.predict(X_infer)[0]
        st.success(f"Predicted Category: **{pred}**")


# ------------------------ TAB 3 -------------------------
with tab3:
    st.header("📁 Batch Prediction")

    file = st.file_uploader("Upload CSV (columns: subject, body)", type=["csv"])

    if file:
        data = pd.read_csv(file)
        if "subject" not in data.columns or "body" not in data.columns:
            st.error("CSV must contain 'subject' and 'body'.")
        else:
            data["text"] = data["subject"] + " " + data["body"]
            data["cleaned"] = data["text"].apply(clean_text)
            Xb = tfidf.transform(data["cleaned"])
            data["predicted_category"] = model.predict(Xb)

            st.dataframe(data)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "results.csv", "text/csv")
