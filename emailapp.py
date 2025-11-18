# emailapp.py
import os
import re
import string
import random
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# -----------------------
# App config
# -----------------------
st.set_page_config(page_title="HR Email Classifier — Auto Synthetic (TE30 + S3)", layout="wide")
st.title("HR Email Categorization — Auto Synthetic (Training/Event TE30 + S3)")

DATA_FILE = "hr_support_emails_2025_6.json"  # must be in repo root
RANDOM_SEED = 42
MINORITY_THRESHOLD_DEFAULT = 15   # categories with count < this are minority
SYNTH_PER_OTHER_CLASS = 30       # S3: 30 synthetic per other minority class
SYNTH_TE_COUNT = 30              # TE30 for Training & Development and Event Coordination

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
            st.error(f"Failed to read JSON `{path}`: {e}")
            return None
    st.error(f"Dataset file not found: {path}. Upload to repo root or use the sidebar uploader.")
    return None

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)   # remove emails
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    toks = [t for t in text.split() if len(t) > 2]
    return " ".join(toks)

# Basic templates for synthetic generation (kept varied)
TEMPLATES = [
    "Hello Team,\n\nWe are running a session on {topic} next week. Please join to learn hands-on techniques.\n\nThanks,\n{sender}",
    "Hi All,\n\nInvitation: {topic}. This event will show practical demos and Q&A. Register to secure your seat.\n\nRegards,\n{sender}",
    "Dear Colleagues,\n\nDon't miss our upcoming workshop on {topic}. The trainer will cover tools and best practices.\n\nBest,\n{sender}",
    "Hello,\n\nWe have scheduled a webinar on {topic} for employees. Please block time on your calendar.\n\nThanks,\n{sender}",
    "Hi,\n\nThis is an invite for the training session: {topic}. Attendance recommended for relevant teams.\n\nRegards,\n{sender}"
]

# candidate topic seeds for many categories (expand as needed)
CATEGORY_TOPICS = {
    "Training & Development": [
        "Google Colab AI features", "Data Science Agent", "Prompt engineering",
        "Model interpretability", "Data visualization in Python", "Deploying ML models",
        "Machine learning fundamentals", "NLP workshop", "Deep learning practicals",
        "MLOps basics"
    ],
    "Event Coordination": [
        "Annual Day logistics", "Team offsite coordination", "Townhall arrangements",
        "Volunteer signup", "Registration for seminar", "Venue booking and seating plan",
        "Event volunteer roster", "Catering and registration", "Session timing and agenda",
        "Stage and AV setup"
    ],
    "Payroll": ["salary not credited", "payslip", "bonus", "salary deduction", "salary breakup"],
    "Finance": ["reimbursement", "invoice", "expense claim", "travel reimbursement"],
    "Leave Management": ["sick leave request", "casual leave", "leave approval", "emergency leave"],
    "IT Support": ["vpn access", "email login", "laptop issue", "system access", "software install"],
    "Security": ["suspicious login", "security alert", "access request", "password reset"],
    "Project Update": ["project status", "milestone update", "release schedule", "timesheet submission"],
    "HR Request": ["experience certificate", "relieving letter", "onboarding docs", "policy clarification"],
    "Training": ["training registration", "course enrollment", "training feedback"],  # fallback synonyms
    "Compliance": ["tax declaration", "policy compliance", "audit schedule", "regulation update"],
    # generic fallback topics for unknown categories
}

GENERIC_TOPICS = ["general HR query", "process clarification", "policy question", "document request"]

SENDER_NAMES = ["Ankit Sharma", "Priya Singh", "Rahul Verma", "Neha Patel", "Karan Mehta", "Aisha Khan", "Rohit Joshi"]

def generate_synthetic_samples_for_category(category, n):
    subjects = []
    bodies = []
    # pick topic seeds: prefer exact category keys, else fallback to title-cased match, else generic
    seeds = CATEGORY_TOPICS.get(category)
    if seeds is None:
        seeds = CATEGORY_TOPICS.get(category.title())
    if seeds is None:
        seeds = GENERIC_TOPICS
    for i in range(n):
        topic = random.choice(seeds)
        template = random.choice(TEMPLATES)
        sender = random.choice(SENDER_NAMES)
        body = template.format(topic=topic, sender=sender)
        # concise subject patterns
        subj_choices = [
            f"{category} — {topic.title()}",
            f"Invitation: {topic.title()}",
            f"{topic.title()} Training",
            f"{category} | {topic.title()}",
            f"Request: {topic.title()}"
        ]
        subject = random.choice(subj_choices)
        subjects.append(subject)
        bodies.append(body)
    return subjects, bodies

# -----------------------
# UI: dataset selection & load
# -----------------------
st.sidebar.header("Dataset options")
uploaded = st.sidebar.file_uploader("Upload JSON dataset (optional) — will be used for this session", type=["json"])
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

# Validate required columns
required_cols = ["subject", "body", "category"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}. Please ensure your JSON has 'subject','body','category'.")
    st.stop()

# prepare text
df['subject'] = df['subject'].fillna("")
df['body'] = df['body'].fillna("")
df['text'] = (df['subject'] + " " + df['body']).str.strip()
df = df[df['text'].str.len() > 0].reset_index(drop=True)
df['cleaned_text'] = df['text'].apply(clean_text)

# show basic info
st.sidebar.write("Rows:", len(df))
categories = sorted(df['category'].dropna().unique().tolist())
st.sidebar.subheader("Detected categories")
st.sidebar.write(categories)

# settings: minority threshold + synthetic counts
st.sidebar.header("Synthetic augmentation settings")
minority_threshold = st.sidebar.number_input("Minority threshold (count < this => minority)", min_value=1, max_value=200, value=MINORITY_THRESHOLD_DEFAULT, step=1)
synthetic_per_other = st.sidebar.number_input("Synthetic per other minority class (S3)", min_value=1, max_value=200, value=SYNTH_PER_OTHER_CLASS, step=1)
synthetic_te_count = st.sidebar.number_input("Synthetic for Training & Event (TE30)", min_value=1, max_value=200, value=SYNTH_TE_COUNT, step=1)

st.header("Dataset overview & EDA (before augmentation)")
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Category distribution")
    counts = df['category'].value_counts()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25*len(counts))))
    sns.barplot(y=counts.index, x=counts.values, ax=ax)
    ax.set_xlabel("Count"); ax.set_ylabel("Category")
    st.pyplot(fig)
with col2:
    st.subheader("Top words (word cloud)")
    txt = " ".join(df['cleaned_text'].tolist())
    if txt.strip() == "":
        st.info("Not enough text for word cloud.")
    else:
        wc = WordCloud(width=900, height=300, background_color='white', max_words=150).generate(txt)
        fig, ax = plt.subplots(figsize=(10,3))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

st.subheader("Top words per some categories (sample)")
sample_show = st.multiselect("Pick categories to preview top words (optional)", options=categories, default=categories[:4])
for cat in sample_show:
    words = " ".join(df[df['category'] == cat]['cleaned_text'].tolist()).split()
    top = Counter(words).most_common(8)
    st.write(f"**{cat}**: {dict(top)}")

# identify minorities
counts = df['category'].value_counts()
minority_classes = counts[counts < minority_threshold].index.tolist()
st.write("Minority classes (count < {}): {}".format(minority_threshold, minority_classes))

# Preview synthetic samples for Training & Development / Event Coordination
st.subheader("Preview TE30 synthetic samples (training & events)")
preview_btn = st.checkbox("Show a quick preview of 5 TE synthetic emails")
if preview_btn:
    te_cats = ["Training & Development", "Event Coordination"]
    for cat in te_cats:
        st.markdown(f"### Samples for: {cat}")
        subjs, bods = generate_synthetic_samples_for_category(cat, 5)
        for s,b in zip(subjs,bods):
            st.markdown(f"**Subject:** {s}")
            st.text(b)
            st.write("---")

# -----------------------
# Augment + Train
# -----------------------
st.header("Augment & Train Models")
st.write("This will: detect minority classes, inject synthetic emails (TE30 for Training & Development & Event Coordination), then train ML models and pick the best.")

augment_and_train = st.button("Generate synthetic (TE30+S3) and train model")

if augment_and_train:
    with st.spinner("Generating synthetic data and training... this may take a short moment"):
        # copy original
        df_aug = df.copy()

        # find minority classes (based on current counts)
        counts = df_aug['category'].value_counts()
        minority_classes = counts[counts < minority_threshold].index.tolist()

        # For TE categories, ensure they exist in dataset; if not, still generate them only if user wants
        te_targets = ["Training & Development", "Event Coordination"]

        synth_rows = []
        # Ensure TE categories get TE synthetic even if not minority (we add only if they exist in categories or want to add new)
        for cat in te_targets:
            # add TE30 only if category exists OR if user explicitly wants to augment missing TE categories
            if cat in df_aug['category'].unique():
                subs, bods = generate_synthetic_samples_for_category(cat, synthetic_te_count)
                for s,b in zip(subs,bods):
                    synth_rows.append({"subject": s, "body": b, "text": s + " " + b, "category": cat, "cleaned_text": clean_text(s + " " + b)})
            else:
                # If the TE category is not present, still add it *only* if the user checked a sidebar option (default: no)
                pass

        # For other minority classes, add S3 synthetic
        for cat in minority_classes:
            # skip if cat is one of TE targets (already handled)
            if cat in te_targets:
                continue
            subs, bods = generate_synthetic_samples_for_category(cat, synthetic_per_other)
            for s,b in zip(subs,bods):
                synth_rows.append({"subject": s, "body": b, "text": s + " " + b, "category": cat, "cleaned_text": clean_text(s + " " + b)})

        # If no minor classes found and no TE synth created, inform the user
        if not synth_rows:
            st.info("No minority classes detected (or TE categories not present). Increase threshold if you want augmentation. No synthetic samples added.")
            df_used = df_aug
        else:
            df_synth = pd.DataFrame(synth_rows)
            df_used = pd.concat([df_aug, df_synth], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
            st.success(f"Added {len(df_synth)} synthetic emails across categories (including TE30).")

        st.write("New category distribution (after augmentation):")
        st.write(df_used['category'].value_counts())

        # Build TF-IDF
        tfidf = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2), sublinear_tf=True, max_features=40000)
        X = tfidf.fit_transform(df_used['cleaned_text'].tolist())
        y = df_used['category'].values

        # stratified split if safe
        def stratify_safe(arr, min_required=2):
            vc = pd.Series(arr).value_counts()
            return (vc >= min_required).all()

        if stratify_safe(y, 2):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_SEED, stratify=y)
            st.write("Using stratified train/test split.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_SEED)
            st.warning("Some classes still have very few samples — using non-stratified split.")

        # Candidate models
        candidates = {
            "LogisticRegression": LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced'),
            "LinearSVC": LinearSVC(max_iter=5000),
            "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED)
        }

        # Quick CV to pick best model (3-fold)
        cv_scores = {}
        for name, mdl in candidates.items():
            try:
                sc = float(np.mean(cross_val_score(mdl, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)))
            except Exception:
                sc = 0.0
            cv_scores[name] = sc
            st.write(f"{name}: CV weighted-F1 = {sc:.4f}")

        best_name = max(cv_scores, key=cv_scores.get)
        st.write("Selected model by CV:", best_name, "score:", cv_scores[best_name])

        best_model = candidates[best_name]
        best_model.fit(X_train, y_train)

        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1w = f1_score(y_test, preds, average='weighted')
        st.success(f"Test Accuracy: {acc:.4f} | Weighted F1: {f1w:.4f}")
        st.text("Classification report:")
        st.text(classification_report(y_test, preds, zero_division=0))

        # Confusion matrix
        labels = np.unique(np.concatenate([y_test, preds]))
        cm = confusion_matrix(y_test, preds, labels=labels)
        fig, ax = plt.subplots(figsize=(max(6, 0.4*len(labels)), max(4, 0.25*len(labels))))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig)

        # store artifacts in session
        st.session_state['tfidf'] = tfidf
        st.session_state['model'] = best_model
        st.session_state['model_name'] = best_name
        st.session_state['df_augmented'] = df_used
        st.success(f"Model '{best_name}' stored in session. You can now run predictions.")

# -----------------------
# Prediction UI
# -----------------------
st.header("Prediction")

colA, colB = st.columns([1,2])
with colA:
    st.subheader("Single email prediction")
    subj = st.text_input("Subject", key="single_subject")
    body = st.text_area("Body", key="single_body")
    if st.button("Classify this email"):
        if 'model' not in st.session_state or 'tfidf' not in st.session_state:
            st.error("No trained model in session. Click 'Generate synthetic (TE30+S3) and train model' first.")
        else:
            raw = (subj + " " + body).strip()
            if not raw:
                st.warning("Please enter subject or body text.")
            else:
                cleaned = clean_text(raw)
                X_inf = st.session_state['tfidf'].transform([cleaned])
                pred = st.session_state['model'].predict(X_inf)[0]
                st.success(f"Predicted category: **{pred}**")

with colB:
    st.subheader("Batch prediction (CSV)")
    uploaded_csv = st.file_uploader("Upload CSV with columns 'subject' and 'body'", type=["csv"])
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
                    st.error("No trained model in session. Run augmentation & training first.")

st.markdown("---")
st.markdown(
    """
    **Deployment notes**
    - Add `emailapp.py` and this dataset file `hr_support_emails_2025_6.json` to your GitHub repo root.
    - Add `requirements.txt` with:
        streamlit
        pandas
        numpy
        scikit-learn
        matplotlib
        seaborn
        wordcloud
    - Deploy on Streamlit Community Cloud. On first run click:
        -> Generate synthetic (TE30+S3) and train model
      After training completes, test your example emails (e.g., Colab AI event invite).
    - If you want the trained model to persist across redeploys, train offline in Colab/local, save the model bundle, then commit `models/email_classifier_bundle.joblib` to the repo and modify the app to load it on startup.
    """
)
