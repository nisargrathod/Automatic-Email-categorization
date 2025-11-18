# emailapp.py
import os
import re
import string
from collections import Counter, defaultdict
import random

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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils import resample

st.set_page_config(page_title="HR Email Classifier — Auto Synthetic (S3)", layout="wide")
st.title("HR Email Categorization — Auto-synthetic (30 per minority class)")

# -----------------------
# Config
# -----------------------
DATA_FILE = "hr_support_emails_2025_6.json"  # must be in repo root
SYNTHETIC_PER_CLASS = 30  # S3 chosen by you
MINORITY_THRESHOLD = 15   # categories with count < this considered minority (changeable in UI)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------
# Helpers
# -----------------------
def safe_read_json(path):
    if os.path.exists(path):
        try:
            return pd.read_json(path)
        except Exception as e:
            st.error(f"Failed to read JSON: {e}")
            return None
    else:
        st.error(f"Dataset file not found at '{path}'. Upload to repo root or use uploader.")
        return None

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)   # emails
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    toks = [t for t in text.split() if len(t) > 2]
    return " ".join(toks)

# Simple template-based synthetic generator
TEMPLATES = [
    "Hello HR,\n\nI have a question regarding {topic}. Could you please advise the steps and timeline?\n\nThanks,\n{sender}",
    "Hi team,\n\nPlease help with {topic}. I need this resolved urgently.\n\nRegards,\n{sender}",
    "Dear HR,\n\nRequesting information about {topic}. Let me know required documents.\n\nBest,\n{sender}",
    "Hello,\n\nI am writing about {topic}. Please guide me on next steps.\n\nThanks,\n{sender}",
    "Hi,\n\nThere is an issue related to {topic}. Kindly assist.\n\nRegards,\n{sender}"
]

# topic seeds per likely HR category (expandable)
CATEGORY_TOPICS = {
    "Payroll": ["salary not credited", "payslip", "bonus", "salary breakup", "salary deduction"],
    "Finance": ["reimbursement", "expense claim", "invoice", "travel reimbursement"],
    "Leave Management": ["sick leave", "casual leave", "leave approval", "emergency leave"],
    "HR Request": ["relieving letter", "experience certificate", "policy clarification", "onboarding documents"],
    "IT Support": ["vpn access", "email login", "system access", "software installation", "laptop issue"],
    "Security": ["suspicious login", "security alert", "access request", "password reset"],
    "Event Coordination": ["volunteer request", "townhall", "wellness week", "diwali party"],
    "Project Update": ["project status", "milestone update", "timesheet submission"],
    "Training & Development": ["training registration", "course enrollment", "training feedback"],
    "Client Communication": ["client meeting follow-up", "client feedback", "deliverable submission"],
    "Compliance": ["policy compliance", "reimbursement policy", "tax declaration"]
}

# fallback topic generator for unknown categories
GENERIC_TOPICS = ["query about company policy", "request for documents", "general HR query", "process clarification"]

def generate_synthetic_for_category(category, n):
    """Generate n synthetic subject+body pairs for given category."""
    subjects = []
    bodies = []
    for i in range(n):
        # choose a topic seed
        seeds = CATEGORY_TOPICS.get(category, CATEGORY_TOPICS.get(category.title(), GENERIC_TOPICS))
        topic = random.choice(seeds) if seeds else random.choice(GENERIC_TOPICS)
        template = random.choice(TEMPLATES)
        sender = random.choice(["Ankit Sharma","Priya Singh","Rahul Verma","Neha Patel","Karan Mehta"])
        body = template.format(topic=topic, sender=sender)
        # subject - concise
        subj_phrases = [
            f"{category} Request: {topic.title()}",
            f"{topic.title()} - {category}",
            f"{category} Query: {topic.title()}",
            f"Query regarding {topic.title()}",
            f"{topic.title()}"
        ]
        subject = random.choice(subj_phrases)
        subjects.append(subject)
        bodies.append(body)
    return subjects, bodies

# -----------------------
# Load data (repo or upload)
# -----------------------
st.sidebar.header("Dataset options")
uploaded = st.sidebar.file_uploader("Upload dataset JSON (optional)", type=["json"])
if uploaded:
    try:
        df = pd.read_json(uploaded)
        st.sidebar.success("Uploaded dataset loaded for session.")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded JSON: {e}")
        st.stop()
else:
    df = safe_read_json(DATA_FILE)
    if df is None:
        st.stop()

# -----------------------
# Validate fields
# -----------------------
if not all(col in df.columns for col in ["subject","body","category"]):
    st.error("Dataset must contain 'subject', 'body', and 'category' fields. If your dataset uses different names, rename columns accordingly.")
    st.stop()

# combine and clean
df['subject'] = df['subject'].fillna("")
df['body'] = df['body'].fillna("")
df['text'] = (df['subject'] + " " + df['body']).str.strip()
df = df[df['text'].str.len()>0].reset_index(drop=True)
df['cleaned_text'] = df['text'].apply(clean_text)

# detect unique categories
categories = sorted(df['category'].dropna().unique().tolist())
st.sidebar.subheader("Detected categories")
st.sidebar.write(categories)
st.write(f"Detected **{len(categories)}** unique categories.")

# let user set minority threshold and synthetic count (UI override)
st.sidebar.header("Synthetic generation settings")
minority_threshold = st.sidebar.number_input("Minority threshold (category count < this will be boosted)", min_value=1, max_value=200, value=MINORITY_THRESHOLD, step=1)
per_class = st.sidebar.number_input("Synthetic emails per minority class", min_value=1, max_value=200, value=SYNTHETIC_PER_CLASS, step=1)

# show category counts
counts = df['category'].value_counts()
st.subheader("Category distribution (before synthetic augmentation)")
fig, ax = plt.subplots(figsize=(8, max(3, 0.3*len(counts))))
sns.barplot(y=counts.index, x=counts.values, ax=ax)
ax.set_xlabel("Count"); ax.set_ylabel("Category")
st.pyplot(fig)

# identify minority classes
minorities = counts[counts < minority_threshold].index.tolist()
st.write("Minority categories (will be synthetically boosted):", minorities)

# preview synthetic content for one chosen class (optional)
st.subheader("Preview synthetic email samples")
preview_cat = st.selectbox("Pick a category to preview synthetic emails", options=(categories if categories else [None]))
if preview_cat:
    subjs,bods = generate_synthetic_for_category(preview_cat, min(5, per_class))
    for s,b in zip(subjs,bods):
        st.markdown(f"**Subject:** {s}")
        st.text(b)
        st.write("---")

# -----------------------
# Create augmented dataset (real + synthetic)
# -----------------------
augment_btn = st.button("Generate synthetic data and train (S3)")
if augment_btn:
    with st.spinner("Generating synthetic emails and training models..."):
        # Copy original
        df_aug = df.copy()

        # For each minority class, add per_class synthetic emails
        synth_rows = []
        for cat in minorities:
            subs, bods = generate_synthetic_for_category(cat, per_class)
            for s,b in zip(subs,bods):
                synth_rows.append({
                    "subject": s,
                    "body": b,
                    "text": s + " " + b,
                    "category": cat,
                    "cleaned_text": clean_text(s + " " + b)
                })
        if synth_rows:
            df_synth = pd.DataFrame(synth_rows)
            df_aug = pd.concat([df_aug, df_synth], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
            st.success(f"Added {len(df_synth)} synthetic emails across {len(minorities)} minority classes.")
        else:
            st.info("No minority classes detected (increase threshold if you want to force augmentation).")

        st.write("New distribution after augmentation:")
        st.write(df_aug['category'].value_counts())

        # Vectorize
        tfidf = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2), sublinear_tf=True, max_features=40000)
        X = tfidf.fit_transform(df_aug['cleaned_text'].tolist())
        y = df_aug['category'].values

        # Train/test split (stratify only if safe)
        def stratify_safe(y_arr, min_required=2):
            vc = pd.Series(y_arr).value_counts()
            return (vc >= min_required).all()
        strat_flag = stratify_safe(y, min_required=2)
        if strat_flag:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42, stratify=y)
            st.write("Using stratified split.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)
            st.warning("Some classes still tiny — using non-stratified split.")

        # Candidate models
        candidates = {
            "LogisticRegression": LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced'),
            "LinearSVC": LinearSVC(max_iter=5000),
            "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        }

        # Cross-validate quickly to choose best
        cv_scores = {}
        for name, mdl in candidates.items():
            try:
                score = float(np.mean(cross_val_score(mdl, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)))
            except Exception:
                score = 0.0
            cv_scores[name] = score
            st.write(f"{name}: CV weighted-F1 = {score:.4f}")

        best_name = max(cv_scores, key=cv_scores.get)
        st.write("Selected model by CV:", best_name, "score:", cv_scores[best_name])

        # Train best on training set
        best_model = candidates[best_name]
        best_model.fit(X_train, y_train)

        # Evaluate
        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1w = f1_score(y_test, preds, average='weighted')
        st.success(f"Test Accuracy: {acc:.4f}  |  Weighted F1: {f1w:.4f}")
        st.text("Classification report:")
        st.text(classification_report(y_test, preds, zero_division=0))

        # Confusion matrix
        labels = np.unique(np.concatenate([y_test, preds]))
        cm = confusion_matrix(y_test, preds, labels=labels)
        fig, ax = plt.subplots(figsize=(max(6, 0.4*len(labels)), max(4, 0.25*len(labels))))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig)

        # store model and tfidf in session
        st.session_state['tfidf'] = tfidf
        st.session_state['model'] = best_model
        st.session_state['model_name'] = best_name
        st.session_state['df_aug'] = df_aug

# -----------------------
# Prediction UI
# -----------------------
st.header("Prediction")

col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Single email")
    subj = st.text_input("Subject", key="single_subj")
    body = st.text_area("Body", key="single_body")
    if st.button("Classify this email"):
        if 'model' not in st.session_state:
            st.error("No trained model in session. Click 'Generate synthetic data and train (S3)' first.")
        else:
            raw = (subj + " " + body).strip()
            if not raw:
                st.warning("Enter subject or body text.")
            else:
                cleaned = clean_text(raw)
                X_inf = st.session_state['tfidf'].transform([cleaned])
                pred = st.session_state['model'].predict(X_inf)[0]
                st.success(f"Predicted category: **{pred}**")

with col2:
    st.subheader("Batch predict (CSV)")
    uploaded_csv = st.file_uploader("Upload CSV with columns 'subject' and 'body'", type=["csv"])
    if uploaded_csv:
        try:
            bdf = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Read error: {e}")
            bdf = None
        if bdf is not None:
            if 'subject' not in bdf.columns or 'body' not in bdf.columns:
                st.error("CSV must contain 'subject' and 'body'.")
            else:
                bdf['text'] = (bdf['subject'].fillna("") + " " + bdf['body'].fillna("")).str.strip()
                bdf['cleaned_text'] = bdf['text'].apply(clean_text)
                if 'model' in st.session_state:
                    Xb = st.session_state['tfidf'].transform(bdf['cleaned_text'].tolist())
                    bdf['predicted_category'] = st.session_state['model'].predict(Xb)
                    st.dataframe(bdf.head(50))
                    csv = bdf.to_csv(index=False).encode('utf-8')
                    st.download_button("Download predictions", csv, "predictions.csv", "text/csv")
                else:
                    st.error("No trained model in session. Run augmentation+train first.")

st.markdown("---")
st.markdown("**Notes:**\n- I inspected your dataset earlier to detect example categories like HR Request, Event Coordination, Finance, Leave Management, Project Update, Security, Training & Development, Client Communication, Compliance, IT Support. See file snippets for evidence. :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}:contentReference[oaicite:8]{index=8}\n- Synthetic generation uses templates and category-based topic seeds. You can customize CATEGORY_TOPICS dict to improve realism for specific categories.")
