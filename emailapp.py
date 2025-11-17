# emailapp.py
import os
import re
import string
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

st.set_page_config(page_title="HR Email Categorization (Your dataset)", layout="wide")
st.title("HR Email Categorization — (Using your dataset)")

# -----------------------
# Config
# -----------------------
DATA_FILE = "hr_support_emails_2025_6.json"   # make sure this file is in repo root
SMALL_CLASS_THRESHOLD_DEFAULT = 3  # classes with < threshold will be mapped to 'Others'

# -----------------------
# Helpers
# -----------------------
def safe_read_json(path):
    if not os.path.exists(path):
        st.error(f"Dataset file not found at `{path}`. Upload it to repo root or use the uploader below.")
        return None
    try:
        df = pd.read_json(path)
        return df
    except Exception as e:
        st.error(f"Failed to read JSON file: {e}")
        return None

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)   # remove email addresses
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)  # remove urls
    text = re.sub(r"\d+", " ", text)  # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if len(t) > 2]
    return " ".join(tokens)

def group_small_classes(df, label_col, threshold):
    counts = df[label_col].value_counts()
    small = counts[counts < threshold].index.tolist()
    if small:
        df[label_col] = df[label_col].apply(lambda x: 'Others' if x in small else x)
    return df

def stratify_safe(y, min_required=2):
    # returns True if all classes have at least min_required samples
    vc = pd.Series(y).value_counts()
    return (vc >= min_required).all()

# -----------------------
# Load dataset (from repo or upload)
# -----------------------
st.sidebar.header("Dataset options")
uploaded = st.sidebar.file_uploader("Upload JSON dataset (optional, will overwrite repo file for session)", type=["json"])
if uploaded:
    try:
        df = pd.read_json(uploaded)
        st.sidebar.success("Uploaded dataset loaded for this session.")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded JSON: {e}")
        st.stop()
else:
    df = safe_read_json(DATA_FILE)
    if df is None:
        st.stop()

st.sidebar.write("Rows:", len(df))

# show sample and required fields
st.sidebar.markdown("**Dataset fields (detected):**")
st.sidebar.write(list(df.columns))

required_fields = ["subject", "body", "category"]
missing = [f for f in required_fields if f not in df.columns]
if missing:
    st.error(f"Dataset missing required fields: {missing}. The app expects `subject`, `body`, and `category`.")
    st.stop()

# combine subject + body
df['subject'] = df['subject'].fillna("")
df['body'] = df['body'].fillna("")
df['text'] = (df['subject'] + " " + df['body']).str.strip()
df = df[df['text'].str.len() > 0].reset_index(drop=True)

# show label counts and let user set small-class threshold
st.sidebar.header("Label cleaning")
label_counts = df['category'].value_counts()
st.sidebar.write(label_counts)
threshold = st.sidebar.number_input("Map classes with < samples -> 'Others' (threshold)", 
                                    min_value=0, max_value=20, value=SMALL_CLASS_THRESHOLD_DEFAULT, step=1)
if threshold > 0:
    df = group_small_classes(df, 'category', threshold)
    st.sidebar.write("After grouping small classes:")
    st.sidebar.write(df['category'].value_counts())

# -----------------------
# Preprocess & EDA
# -----------------------
if st.checkbox("Show dataset sample", value=False):
    st.dataframe(df.sample(min(len(df), 10)))

st.header("Exploratory Data Analysis")

col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Category distribution")
    counts = df['category'].value_counts()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3*len(counts))))
    sns.barplot(y=counts.index, x=counts.values, ax=ax)
    ax.set_xlabel("Count"); ax.set_ylabel("Category")
    st.pyplot(fig)

with col2:
    st.subheader("Word cloud (top words)")
    df['cleaned_text'] = df['text'].apply(clean_text)
    all_text = " ".join(df['cleaned_text'].tolist())
    if len(all_text.strip())==0:
        st.info("Not enough text to generate wordcloud.")
    else:
        wc = WordCloud(width=900, height=300, background_color='white', max_words=150).generate(all_text)
        fig, ax = plt.subplots(figsize=(10,3))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# show top words per top-K categories
st.subheader("Top words per category")
top_k = st.slider("Top how many categories to show?", 1, min(12, len(counts)), min(6, len(counts)))
for cat in counts.index[:top_k]:
    words = " ".join(df[df['category']==cat]['cleaned_text']).split()
    top_words = Counter(words).most_common(8)
    st.markdown(f"**{cat}** — {len(words)} tokens")
    if top_words:
        cols = st.columns(4)
        for i,(w,cnt) in enumerate(top_words):
            cols[i%4].write(f"{w} — {cnt}")

# -----------------------
# Model training (TF-IDF + LogisticRegression)
# -----------------------
st.header("Train model & evaluation")
train_btn = st.button("Train model now")
if train_btn:
    with st.spinner("Training model..."):
        # Build features
        tfidf = TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2), max_features=20000)
        X = tfidf.fit_transform(df['cleaned_text'].tolist())
        y = df['category'].values

        # decide whether to stratify: only stratify when every class has >=2 samples
        stratify_flag = stratify_safe(y, min_required=2)
        if stratify_flag:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42, stratify=y)
            st.write("Using stratified split.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)
            st.warning("Some classes are very small — using non-stratified split to avoid errors.")

        model = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        st.write("Accuracy:", f"{accuracy_score(y_test, preds):.4f}")
        st.write("Weighted F1:", f"{f1_score(y_test, preds, average='weighted'):.4f}")
        st.text("Classification report:")
        st.text(classification_report(y_test, preds, zero_division=0))

        # show confusion matrix
        labels = np.unique(np.concatenate([y_test, preds]))
        cm = confusion_matrix(y_test, preds, labels=labels)
        fig, ax = plt.subplots(figsize=(max(6, 0.4*len(labels)), max(4,0.25*len(labels))))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig)

        # store in session_state so user can predict
        st.session_state['tfidf'] = tfidf
        st.session_state['model'] = model
        st.success("Training complete and model loaded to session.")

# If model is in session, show predict UI
if 'model' in st.session_state and 'tfidf' in st.session_state:
    st.success("Model ready — use the prediction panel below.")

st.header("Prediction")

colA, colB = st.columns([1,2])

with colA:
    st.subheader("Single email")
    subj = st.text_input("Subject")
    body = st.text_area("Body")
    if st.button("Classify this email"):
        if 'model' not in st.session_state:
            st.error("No model loaded — press 'Train model now' first (or upload a ready model).")
        else:
            raw = (subj + " " + body).strip()
            if not raw:
                st.warning("Please enter subject or body")
            else:
                cleaned = clean_text(raw)
                X_infer = st.session_state['tfidf'].transform([cleaned])
                pred = st.session_state['model'].predict(X_infer)[0]
                st.success(f"Predicted Category: **{pred}**")

with colB:
    st.subheader("Batch predict (CSV)")
    batch = st.file_uploader("Upload CSV with columns: subject, body", type=['csv'])
    if batch is not None:
        bdf = pd.read_csv(batch)
        if 'subject' not in bdf.columns or 'body' not in bdf.columns:
            st.error("CSV must contain 'subject' and 'body' columns")
        else:
            bdf['text'] = (bdf['subject'].fillna("") + " " + bdf['body'].fillna("")).str.strip()
            bdf['cleaned_text'] = bdf['text'].apply(clean_text)
            if 'tfidf' in st.session_state and 'model' in st.session_state:
                Xb = st.session_state['tfidf'].transform(bdf['cleaned_text'].tolist())
                bdf['predicted_category'] = st.session_state['model'].predict(Xb)
                st.dataframe(bdf.head(50))
                csv = bdf.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
            else:
                st.error("Train the model first (press 'Train model now') to run batch predictions.")

st.markdown("---")
st.markdown("**Deployment notes:** Make sure `hr_support_emails_2025_6.json` is in the same folder as this `emailapp.py` in your GitHub repo when deploying to Streamlit Cloud.")
