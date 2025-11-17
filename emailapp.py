# emailapp.py
import os
import re
import string
from pathlib import Path
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
import joblib

# -----------------------
# Config / file paths
# -----------------------
DATA_FILENAME = "hr_support_emails_2025_6.json"
MODEL_BUNDLE = "best_email_classifier.joblib"   # single file with {'vectorizer':..., 'model':...}
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / MODEL_BUNDLE

st.set_page_config(page_title="HR Email Categorizer", layout="wide")
st.title("🔖 HR Automatic Email Categorization (Cloud-ready)")

# -----------------------
# Utilities
# -----------------------
@st.cache_data
def load_dataset_from_repo(path: str):
    if os.path.exists(path):
        try:
            df = pd.read_json(path)
            return df
        except Exception as e:
            st.error(f"Failed to read JSON from {path}: {e}")
            return None
    return None

def save_uploaded_file(uploaded, dest_path):
    with open(dest_path, "wb") as f:
        f.write(uploaded.getbuffer())

@st.cache_data
def get_stopwords():
    # lightweight stopwords set (we avoid heavy downloads at import time)
    try:
        import nltk
        from nltk.corpus import stopwords
        stopwords.words('english')
    except Exception:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words("english"))

STOPWORDS = get_stopwords()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)  # emails
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in re.findall(r"\b[a-z]+\b", text) if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

# -----------------------
# Model load / save
# -----------------------
@st.cache_resource
def load_model_bundle(path: str):
    if os.path.exists(path):
        try:
            bundle = joblib.load(path)
            # expect dict with 'vectorizer' and 'model'
            if isinstance(bundle, dict) and 'vectorizer' in bundle and 'model' in bundle:
                return bundle['vectorizer'], bundle['model']
            # fallbacks: if the saved bundle is directly a vectorizer or model, handle gracefully
            st.warning("Model file structure unexpected; attempting to use it directly.")
            return bundle.get('vectorizer', None), bundle.get('model', None)
        except Exception as e:
            st.error(f"Failed to load model bundle: {e}")
            return None, None
    return None, None

def save_model_bundle(path: str, vectorizer, model):
    bundle = {"vectorizer": vectorizer, "model": model}
    joblib.dump(bundle, path)
    st.success(f"Saved model bundle to {path}")

# -----------------------
# UI: Left column - data & training controls
# -----------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Dataset & Model")
    st.markdown("Upload dataset (optional) or use the dataset in repo.")
    uploaded = st.file_uploader("Upload JSON dataset (single file, same schema)", type=["json"], key="upload_dataset")
    if uploaded is not None:
        save_uploaded_file(uploaded, DATA_FILENAME)
        st.success(f"Uploaded and saved dataset as `{DATA_FILENAME}` in repo (temporary file for app runtime).")

    # Load dataset (from repo working directory)
    df = load_dataset_from_repo(DATA_FILENAME)
    if df is None:
        st.info("No dataset found in repo. Upload a dataset above or push the dataset file to your GitHub repo root.")
        st.stop()

    st.write("Dataset loaded. Rows:", len(df))
    if st.checkbox("Show raw data sample"):
        st.dataframe(df.sample(6))

    # Ensure text field exists
    if 'subject' not in df.columns or 'body' not in df.columns:
        st.error("Dataset does not contain 'subject' and 'body' fields. Please provide dataset with these fields.")
        st.stop()

    # create text
    df['subject'] = df['subject'].fillna("")
    df['body'] = df['body'].fillna("")
    df['text'] = (df['subject'] + " " + df['body']).str.strip()
    df = df[df['text'].str.len() > 0].reset_index(drop=True)
    if df.empty:
        st.error("After combining subject+body no text remains. Check dataset.")
        st.stop()

    # quick label cleaning: small classes -> Others (optional)
    st.markdown("**Label cleaning**")
    counts = df['category'].value_counts()
    st.write(counts)
    small_threshold = st.number_input("Small class threshold (map classes with < threshold samples to 'Others')", min_value=0, max_value=50, value=3, step=1)
    if small_threshold > 0:
        small_classes = counts[counts < small_threshold].index.tolist()
        if small_classes:
            df['category'] = df['category'].apply(lambda x: 'Others' if x in small_classes else x)
            st.write("Classes after mapping small classes to 'Others':")
            st.write(df['category'].value_counts())

    st.markdown("---")
    st.markdown("**Model files**")
    vectorizer, model = load_model_bundle(str(MODEL_PATH))
    if vectorizer is None or model is None:
        st.warning("No saved model found. You can train a model below and it will be saved for future runs.")
    else:
        st.success("Saved model bundle loaded from repo.")

    # Train controls
    st.markdown("### Train model (TF-IDF + LogisticRegression)")
    retrain = st.button("Train model now", key="train_btn")
    do_quick_grid = st.checkbox("Use small GridSearch (C values) during training (slower)", value=False)

# -----------------------
# Model training (runs when button clicked)
# -----------------------
def train_and_save(df, quick_grid=False):
    st.info("Starting training... preprocessing text")
    df['cleaned_text'] = df['text'].apply(clean_text)

    X_texts = df['cleaned_text'].values
    y = df['category'].values

    # TF-IDF
    tfidf = TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2), max_features=20000)
    X = tfidf.fit_transform(X_texts)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42, stratify=y)

    if quick_grid:
        st.info("Grid search over small C values for LogisticRegression (this may take a bit).")
        best_model = None
        best_score = -1
        for C in [0.01, 0.1, 1, 5]:
            m = LogisticRegression(C=C, max_iter=2000, solver='liblinear', class_weight='balanced')
            m.fit(X_train, y_train)
            score = f1_score(y_test, m.predict(X_test), average='weighted')
            st.write(f"C={C} -> weighted F1: {score:.4f}")
            if score > best_score:
                best_model = m
                best_score = score
        model = best_model
    else:
        model = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')
        model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    st.success(f"Finished training — Test accuracy: {acc:.4f}, weighted F1: {f1:.4f}")
    st.text("Classification report:")
    st.text(classification_report(y_test, preds, zero_division=0))

    # confusion matrix
    labels = np.unique(np.concatenate([y_test, preds]))
    cm = confusion_matrix(y_test, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Save
    save_model_bundle(str(MODEL_PATH), tfidf, model)
    return tfidf, model

if retrain:
    vectorizer, model = train_and_save(df, quick_grid=do_quick_grid)

# If model is still None, try to load again (in case it was saved)
if vectorizer is None or model is None:
    vectorizer, model = load_model_bundle(str(MODEL_PATH))

# If still none, allow to stop
if vectorizer is None or model is None:
    st.error("No model is available. Please train a model or upload a prepared model bundle (`models/email_classifier_bundle.joblib`).")
    st.stop()

# -----------------------
# Right column: EDA and Prediction UI
# -----------------------
with col2:
    st.header("EDA / Model Inference")

    # EDA - category distribution
    st.subheader("Category Distribution")
    cat_counts = df['category'].value_counts()
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=cat_counts.values, y=cat_counts.index, orient='h', ax=ax)
    ax.set_xlabel("Count"); ax.set_ylabel("Category")
    st.pyplot(fig)

    # Word cloud
    st.subheader("Word Cloud (top words across dataset)")
    df['cleaned_text'] = df.get('cleaned_text', df['text'].apply(clean_text))
    all_text = " ".join(df['cleaned_text'].tolist())
    if len(all_text.strip()) == 0:
        st.info("Not enough text for a word cloud.")
    else:
        wc = WordCloud(width=900, height=300, background_color='white', max_words=150).generate(all_text)
        fig, ax = plt.subplots(figsize=(12,4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    # Top words per category (show top 8 for top N categories)
    st.subheader("Top words per category (sample)")
    top_n_categories = st.slider("Show top how many categories?", min_value=1, max_value=min(8, len(cat_counts)), value=4)
    for cat in cat_counts.index[:top_n_categories]:
        text_cat = " ".join(df[df['category'] == cat]['cleaned_text'].tolist())
        c = Counter(text_cat.split()).most_common(8)
        if not c:
            st.write(f"{cat}: (no tokens)")
            continue
        words, counts = zip(*c)
        fig, ax = plt.subplots(figsize=(6,2))
        sns.barplot(x=list(counts), y=list(words), ax=ax)
        ax.set_xlabel("Count")
        ax.set_title(cat)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Predict a single email")

    subj = st.text_input("Subject", "")
    body = st.text_area("Body", "")
    predict_btn = st.button("Classify email", key="predict_button")
    if predict_btn:
        raw = (subj + " " + body).strip()
        if raw == "":
            st.warning("Enter subject or body text to classify.")
        else:
            cleaned = clean_text(raw)
            X = vectorizer.transform([cleaned])
            pred = model.predict(X)[0]
            st.success(f"Predicted category: **{pred}**")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                top_idx = np.argsort(probs)[::-1][:5]
                st.write("Top probabilities:")
                for i in top_idx:
                    st.write(f"{model.classes_[i]} : {probs[i]:.3f}")

    st.markdown("---")
    st.subheader("Batch predict (upload CSV/JSON with subject/body)")
    batch_upload = st.file_uploader("Upload CSV/JSON (must have 'subject' and 'body' columns)", type=["csv", "json"], key="batch_upload")
    if batch_upload:
        try:
            if batch_upload.name.endswith(".csv"):
                batch_df = pd.read_csv(batch_upload)
            else:
                batch_df = pd.read_json(batch_upload)
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            batch_df = None
        if batch_df is not None:
            if 'subject' not in batch_df.columns or 'body' not in batch_df.columns:
                st.error("Uploaded file must contain 'subject' and 'body' columns.")
            else:
                batch_df['text'] = (batch_df['subject'].fillna("") + " " + batch_df['body'].fillna("")).str.strip()
                batch_df['cleaned_text'] = batch_df['text'].apply(clean_text)
                Xb = vectorizer.transform(batch_df['cleaned_text'].tolist())
                preds = model.predict(Xb)
                batch_df['predicted_category'] = preds
                st.dataframe(batch_df.head(30))
                # allow download
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")

st.markdown("---")
st.markdown(
    """
    **Deployment notes**
    - Push this repo (root) to GitHub with:
      - `emailapp.py`
      - `requirements.txt`
      - `hr_support_emails_2025_6.json` (or upload via UI)
      - `models/email_classifier_bundle.joblib` (optional; app will train and create this)
    - On Streamlit Cloud choose the repo and press 'Deploy'.
    - If you train the model from the web UI, the bundle is saved to the ephemeral disk for that session — to reuse across deployments push `models/email_classifier_bundle.joblib` into your repo.
    """
)
