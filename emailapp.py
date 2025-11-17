# emailapp.py
import os
import re
import string
from collections import Counter, defaultdict
from math import ceil

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils import resample

st.set_page_config(page_title="HR Email Classifier — ML (Fast)", layout="wide")
st.title("HR Email Categorization — ML-only (fast & reliable)")

# -----------------------
# Config
# -----------------------
DATA_FILE = "hr_support_emails_2025_6.json"  # must be in repo root
SMALL_CLASS_THRESHOLD_DEFAULT = 3

# -----------------------
# Helpers
# -----------------------
def safe_read_json(path):
    if os.path.exists(path):
        try:
            df = pd.read_json(path)
            return df
        except Exception as e:
            st.error(f"Failed to read JSON: {e}")
            return None
    return None

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if len(t) > 2]
    return " ".join(tokens)

def upsample_minority_classes(df, label_col, target_size=None, strategy="to_median"):
    """
    Simple oversampling (no external libs). Strategies:
      - "to_median": raise minority classes to median class size.
      - "to_max": raise to the largest class size.
      - "to_fixed": raise to target_size (int) if provided.
    Returns new dataframe.
    """
    counts = df[label_col].value_counts()
    if strategy == "to_median":
        target = int(np.median(counts))
    elif strategy == "to_max":
        target = int(counts.max())
    elif strategy == "to_fixed" and target_size:
        target = int(target_size)
    else:
        target = int(np.median(counts))

    # if target < current smallest, keep it as is (no downsampling here)
    frames = []
    for cls, grp in df.groupby(label_col):
        n = len(grp)
        if n >= target or target <= 0:
            frames.append(grp)
        else:
            # upsample with replacement
            reps = resample(grp, replace=True, n_samples=target, random_state=42)
            frames.append(reps)
    df_bal = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_bal

def stratify_safe(y, min_required=2):
    vc = pd.Series(y).value_counts()
    return (vc >= min_required).all()

def compare_models(X, y, models, cv=3):
    scores = {}
    for name, mdl in models.items():
        try:
            s = cross_val_score(mdl, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
            scores[name] = np.mean(s)
        except Exception as e:
            scores[name] = 0.0
    return scores

# -----------------------
# Load dataset
# -----------------------
st.sidebar.header("Dataset options")
uploaded = st.sidebar.file_uploader("Upload dataset JSON (optional)", type=["json"])
if uploaded:
    try:
        df = pd.read_json(uploaded)
        st.sidebar.success("Uploaded dataset will be used for this session.")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded file: {e}")
        st.stop()
else:
    df = safe_read_json(DATA_FILE)
    if df is None:
        st.error(f"Dataset `{DATA_FILE}` not found in repo root. Upload via sidebar or push the file to repo.")
        st.stop()

st.sidebar.write("Rows:", len(df))
st.sidebar.markdown("Detected fields:")
st.sidebar.write(list(df.columns))

# Validate expected fields
required = ["subject", "body", "category"]
missing = [f for f in required if f not in df.columns]
if missing:
    st.error(f"Dataset missing fields: {missing}. The app requires at least 'subject', 'body', and 'category'.")
    st.stop()

# combine text and clean
df['subject'] = df['subject'].fillna("")
df['body'] = df['body'].fillna("")
df['text'] = (df['subject'] + " " + df['body']).str.strip()
df = df[df['text'].str.len() > 0].reset_index(drop=True)
df['cleaned_text'] = df['text'].apply(clean_text)

# label cleaning UI
st.sidebar.header("Label cleaning")
counts = df['category'].value_counts()
st.sidebar.write(counts)
threshold = st.sidebar.number_input("Map classes with < threshold → Others", min_value=0, max_value=50, value=SMALL_CLASS_THRESHOLD_DEFAULT, step=1)
if threshold > 0:
    small = counts[counts < threshold].index.tolist()
    if small:
        df['category'] = df['category'].apply(lambda x: 'Others' if x in small else x)
        st.sidebar.write("After mapping small classes:")
        st.sidebar.write(df['category'].value_counts())

# EDA
st.header("Exploratory Data Analysis")
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Category distribution")
    ct = df['category'].value_counts()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3*len(ct))))
    sns.barplot(y=ct.index, x=ct.values, ax=ax)
    ax.set_xlabel("Count"); ax.set_ylabel("Category")
    st.pyplot(fig)
with col2:
    st.subheader("Word cloud (top words)")
    text_all = " ".join(df['cleaned_text'].tolist())
    if len(text_all.strip())==0:
        st.info("Not enough text for wordcloud.")
    else:
        wc = WordCloud(width=900, height=300, background_color='white', max_words=150).generate(text_all)
        fig, ax = plt.subplots(figsize=(10,3))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

st.subheader("Top words per top categories")
top_n = st.slider("Show top how many categories?", 1, min(12, len(ct)), min(6, len(ct)))
for cat in ct.index[:top_n]:
    words = " ".join(df[df['category']==cat]['cleaned_text']).split()
    topw = Counter(words).most_common(6)
    st.markdown(f"**{cat}** — {dict(topw)}")

# -----------------------
# Training controls
# -----------------------
st.header("Train models (TF-IDF + multiple ML models)")
balance_strategy = st.selectbox("Balance strategy (oversampling)", ["none", "to_median", "to_max"], index=1)
test_size = st.slider("Test set size (%)", 10, 30, 18)
do_compare = st.checkbox("Compare models (cross-validated)", value=True)
train_btn = st.button("Train & Select Best Model")

if train_btn:
    with st.spinner("Training..."):
        # vectorize
        tfidf = TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2),
                                sublinear_tf=True, max_features=40000)
        X_all = tfidf.fit_transform(df['cleaned_text'].tolist())
        y_all = df['category'].values

        # balancing (simple oversample)
        if balance_strategy != "none":
            df_bal = upsample_minority_classes(df[['cleaned_text','category']].rename(columns={'cleaned_text':'text'}),
                                              label_col='category', strategy=("to_max" if balance_strategy=="to_max" else "to_median"))
            X_all = tfidf.transform(df_bal['text'].tolist()) if False else tfidf.fit_transform(df_bal['text'].tolist())
            y_all = df_bal['category'].values
            st.write("After balancing class counts:")
            st.write(pd.Series(y_all).value_counts())

        # safe stratify
        strat_flag = stratify_safe(y_all, min_required=2)
        if strat_flag:
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size/100.0, random_state=42, stratify=y_all)
            st.write("Using stratified split.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size/100.0, random_state=42)
            st.warning("Some classes are tiny; using non-stratified split.")

        # define candidate models (fast)
        models = {
            "LogisticRegression": LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced'),
            "LinearSVC": LinearSVC(max_iter=5000),
            "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        }

        # optional quick compare
        best_name, best_score, best_model = None, -1.0, None
        if do_compare:
            st.write("Comparing models with 3-fold CV (f1_weighted)...")
            cv_scores = {}
            for name, mdl in models.items():
                try:
                    s = cross_val_score(mdl, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
                    mean_s = float(np.mean(s))
                except Exception as e:
                    mean_s = 0.0
                cv_scores[name] = mean_s
                st.write(f"{name}: CV weighted-F1 = {mean_s:.4f}")
            # pick top model
            best_name = max(cv_scores, key=cv_scores.get)
            st.write("Selected by CV:", best_name, "score", cv_scores[best_name])
            # train best on full train
            best_model = models[best_name]
            best_model.fit(X_train, y_train)
            best_score = cv_scores[best_name]
        else:
            # train logistic by default
            best_name = "LogisticRegression"
            best_model = models[best_name]
            best_model.fit(X_train, y_train)

        # final evaluation
        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1w = f1_score(y_test, preds, average='weighted')
        st.success(f"Final model: {best_name} — Test accuracy: {acc:.4f}, weighted F1: {f1w:.4f}")
        st.text("Classification report:")
        st.text(classification_report(y_test, preds, zero_division=0))

        # confusion matrix
        labels = np.unique(np.concatenate([y_test, preds]))
        cm = confusion_matrix(y_test, preds, labels=labels)
        fig, ax = plt.subplots(figsize=(max(6, 0.4*len(labels)), max(4, 0.25*len(labels))))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig)

        # store model & vectorizer in session
        st.session_state['tfidf'] = tfidf
        st.session_state['model'] = best_model
        st.session_state['model_name'] = best_name
        st.success(f"Model '{best_name}' stored in session for prediction.")

# If model present, show quick stats
if 'model' in st.session_state:
    st.info(f"Loaded model in session: {st.session_state.get('model_name','(unknown)')}")

# -----------------------
# Prediction UI
# -----------------------
st.header("Prediction")

colA, colB = st.columns([1,2])
with colA:
    st.subheader("Single email prediction")
    subj = st.text_input("Subject", key="single_subj")
    body = st.text_area("Body", key="single_body")
    if st.button("Classify single email"):
        if 'model' not in st.session_state or 'tfidf' not in st.session_state:
            st.error("No trained model available. Press 'Train & Select Best Model' first.")
        else:
            raw = (subj + " " + body).strip()
            if raw == "":
                st.warning("Enter subject or body text.")
            else:
                cleaned = clean_text(raw)
                X = st.session_state['tfidf'].transform([cleaned])
                pred = st.session_state['model'].predict(X)[0]
                st.success(f"Predicted category: **{pred}**")

with colB:
    st.subheader("Batch prediction (CSV)")
    uploaded_csv = st.file_uploader("Upload CSV with columns subject, body", type=["csv"])
    if uploaded_csv:
        try:
            bdf = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            bdf = None
        if bdf is not None:
            if 'subject' not in bdf.columns or 'body' not in bdf.columns:
                st.error("CSV must contain 'subject' and 'body' columns.")
            else:
                bdf['text'] = (bdf['subject'].fillna("") + " " + bdf['body'].fillna("")).str.strip()
                bdf['cleaned_text'] = bdf['text'].apply(clean_text)
                if 'model' in st.session_state and 'tfidf' in st.session_state:
                    Xb = st.session_state['tfidf'].transform(bdf['cleaned_text'].tolist())
                    bdf['predicted_category'] = st.session_state['model'].predict(Xb)
                    st.dataframe(bdf.head(50))
                    csv = bdf.to_csv(index=False).encode('utf-8')
                    st.download_button("Download predictions", csv, "predictions.csv", "text/csv")
                else:
                    st.error("Train model first to run batch predictions.")

st.markdown("---")
st.markdown(
    """
    **Notes**
    - This version uses only lightweight ML and TF-IDF for fast, reliable predictions on Streamlit Cloud.
    - I inspected your dataset `hr_support_emails_2025_6.json` to confirm it contains `subject`, `body`, and `category` fields and many HR categories (Finance, IT Support, Leave Management, Project Update, Training & Development, HR Request, Event Coordination, etc.). Use that file in the repo root. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}
    - If you want stronger balancing, we can add SMOTE or try class-weighting / calibrated probabilities. For production, create a persistent model file (trained offline) and upload `models/email_classifier_bundle.joblib` to the repo for instant cold starts.
    """
)
