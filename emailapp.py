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
# Page setup & theme CSS
# -----------------------
st.set_page_config(page_title="HR Email Classifier — Professional", layout="wide")
st.markdown(
    """
    <style>
    /* Corporate clean theme: blue + grey */
    .reportview-container {
      background: #f6f8fa;
    }
    .stApp {
      background: #f6f8fa;
      color: #0f1724;
    }
    .header {
      background: linear-gradient(90deg,#0b5ed7 0%, #3b82f6 100%);
      color: white;
      padding: 18px;
      border-radius: 8px;
      box-shadow: 0 2px 6px rgba(11,78,215,0.15);
    }
    .card {
      background: white;
      padding: 14px;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(16,24,40,0.05);
      margin-bottom: 12px;
    }
    .btn-primary {
      background-color: #0b5ed7;
      color: white;
      padding: 8px 16px;
      border-radius: 6px;
      border: none;
    }
    .muted {
      color: #6b7280;
      font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header"><h2 style="margin:0">HR Email Categorization — Professional Dashboard</h2><div class="muted">Automatically organize incoming HR emails into the right categories</div></div>', unsafe_allow_html=True)
st.write("")

# -----------------------
# Config
# -----------------------
DATA_FILE = "hr_support_emails_2025_6.json"  # ensure present in repo root
RANDOM_SEED = 42
MINORITY_THRESHOLD_DEFAULT = 15   # internal threshold used to decide which categories will be boosted
SYNTH_PER_OTHER_CLASS = 30        # internal augmentation for small classes
SYNTH_TE_COUNT = 30               # internal augmentation for Training & Development / Event Coordination

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------
# Helpers (clean, generator, etc.)
# -----------------------
def safe_read_json(path):
    if os.path.exists(path):
        try:
            return pd.read_json(path)
        except Exception as e:
            st.error(f"Failed to read dataset: {e}")
            return None
    st.error(f"Dataset not found at '{path}'. Please upload it to the repo root or use the uploader.")
    return None

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)   # remove emails
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if len(t) > 2]
    return " ".join(tokens)

# gentle HR-friendly templates used *internally* to generate more training examples when needed
TEMPLATES = [
    "Hello Team,\n\nWe are organizing a session on {topic} next week. Please join to learn hands-on techniques.\n\nThanks,\n{sender}",
    "Hi All,\n\nInvitation: {topic}. This event will show practical demos and Q&A. Register to secure your seat.\n\nRegards,\n{sender}",
    "Dear Colleagues,\n\nDon't miss our upcoming workshop on {topic}. The trainer will cover tools and best practices.\n\nBest,\n{sender}",
    "Hello,\n\nWe have scheduled a webinar on {topic} for employees. Please block time on your calendar.\n\nThanks,\n{sender}",
    "Hi,\n\nThis is an invite for the training session: {topic}. Attendance recommended for relevant teams.\n\nRegards,\n{sender}"
]

# topic seeds for categories (improve as you like)
CATEGORY_TOPICS = {
    "Training & Development": [
        "Google Colab AI features", "Data Science Agent", "Prompt engineering",
        "Data visualization in Python", "Deploying ML models", "Machine learning fundamentals",
        "NLP workshop", "Deep learning practicals", "MLOps basics"
    ],
    "Event Coordination": [
        "Annual Day logistics", "Townhall arrangements", "Volunteer signup", "Registration for seminar",
        "Venue booking and seating plan", "Catering and registration", "Stage and AV setup"
    ],
    "Payroll": ["salary not credited", "payslip", "bonus", "salary deduction", "salary breakup"],
    "Finance": ["reimbursement", "invoice", "expense claim", "travel reimbursement"],
    "Leave Management": ["sick leave request", "casual leave", "leave approval", "emergency leave"],
    "IT Support": ["vpn access", "email login", "laptop issue", "system access", "software install"],
    "Security": ["suspicious login", "security alert", "access request", "password reset"],
    "Project Update": ["project status", "milestone update", "release schedule", "timesheet submission"],
    "HR Request": ["experience certificate", "relieving letter", "onboarding docs", "policy clarification"],
    "Compliance": ["tax declaration", "policy compliance", "audit", "regulation update"]
}

GENERIC_TOPICS = ["general HR query", "process clarification", "document request"]
SENDER_NAMES = ["Ankit Sharma", "Priya Singh", "Rahul Verma", "Neha Patel", "Karan Mehta", "Aisha Khan"]

def generate_samples(category, n):
    seeds = CATEGORY_TOPICS.get(category) or CATEGORY_TOPICS.get(category.title()) or GENERIC_TOPICS
    subs, bods = [], []
    for _ in range(n):
        topic = random.choice(seeds)
        template = random.choice(TEMPLATES)
        sender = random.choice(SENDER_NAMES)
        body = template.format(topic=topic, sender=sender)
        subj_choices = [
            f"{category} — {topic.title()}",
            f"Invitation: {topic.title()}",
            f"{topic.title()} Training",
            f"{category} | {topic.title()}",
            f"Request: {topic.title()}"
        ]
        subs.append(random.choice(subj_choices))
        bods.append(body)
    return subs, bods

# -----------------------
# Sidebar controls (minimal & HR-friendly)
# -----------------------
st.sidebar.header("Upload / Update Data")
uploaded = st.sidebar.file_uploader("Upload dataset (optional) — JSON with fields 'subject','body','category'", type=["json"])
if uploaded:
    try:
        df = pd.read_json(uploaded)
        st.sidebar.success("Dataset loaded for this session.")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded JSON: {e}")
        st.stop()
else:
    df = safe_read_json(DATA_FILE)
    if df is None:
        st.stop()

# validation
required = ["subject", "body", "category"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}. Please provide a dataset with 'subject','body','category'.")
    st.stop()

# prepare data
df['subject'] = df['subject'].fillna("")
df['body'] = df['body'].fillna("")
df['text'] = (df['subject'] + " " + df['body']).str.strip()
df = df[df['text'].str.len() > 0].reset_index(drop=True)
df['cleaned_text'] = df['text'].apply(clean_text)

# minimal settings hidden from HR: we only expose a friendly "Update Model" button
st.sidebar.markdown("**Model maintenance**")
st.sidebar.info("Click 'Update Model' to refresh the classifier with latest data (safe for HR users).")
update_model_btn = st.sidebar.button("Update Model")

# -----------------------
# Main layout: 3 columns top (summary cards)
# -----------------------
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.markdown('<div class="card"><h3 style="margin:0">Total emails</h3><div style="font-size:22px;"><b>{}</b></div></div>'.format(len(df)), unsafe_allow_html=True)
with col2:
    unique_cats = df['category'].nunique()
    st.markdown('<div class="card"><h3 style="margin:0">Categories</h3><div style="font-size:22px;"><b>{}</b></div></div>'.format(unique_cats), unsafe_allow_html=True)
with col3:
    st.markdown('<div class="card"><h3 style="margin:0">Last updated</h3><div style="font-size:14px; color: #6b7280;">Automatically updated</div></div>', unsafe_allow_html=True)

st.write("")

# -----------------------
# Tabs: Dashboard / Classify / Bulk
# -----------------------
tabs = st.tabs(["📊 Dashboard", "📮 Classify Email", "📁 Bulk Categorize"])
dashboard_tab, classify_tab, bulk_tab = tabs

# -----------------------
# Dashboard tab
# -----------------------
with dashboard_tab:
    st.header("Overview")
    st.markdown("A quick summary of the most common email types sent to HR.")
    # distribution + wordcloud side-by-side
    c1, c2 = st.columns([1,1])
    with c1:
        st.subheader("Category distribution")
        counts = df['category'].value_counts()
        fig, ax = plt.subplots(figsize=(8, max(3, 0.25*len(counts))))
        sns.barplot(y=counts.index, x=counts.values, ax=ax, palette="Blues_d")
        ax.set_xlabel("Count"); ax.set_ylabel("Category")
        st.pyplot(fig)
    with c2:
        st.subheader("Common words")
        text_all = " ".join(df['cleaned_text'].tolist())
        if len(text_all.strip()) == 0:
            st.info("Not enough text for word cloud.")
        else:
            wc = WordCloud(width=900, height=300, background_color='white', max_words=150).generate(text_all)
            fig2, ax2 = plt.subplots(figsize=(8,3))
            ax2.imshow(wc, interpolation='bilinear')
            ax2.axis('off')
            st.pyplot(fig2)

    st.markdown("### Example recent emails")
    try:
        st.dataframe(df[['subject','category']].sample(min(6,len(df))).rename(columns={'subject':'Subject','category':'Category'}))
    except Exception:
        st.write("Not enough rows to preview.")

# -----------------------
# Minimal backend: augment & train when user clicks Update Model
# (Note: we hide all augmentation details from HR)
# -----------------------
def augment_and_train_internal(df_input, minority_threshold=MINORITY_THRESHOLD_DEFAULT,
                               synth_other=SYNTH_PER_OTHER_CLASS, synth_te=SYNTH_TE_COUNT):
    # prepare working copy
    df_work = df_input.copy()
    counts = df_work['category'].value_counts()
    minority_classes = counts[counts < minority_threshold].index.tolist()

    # Always provide extra examples for Training & Development and Event Coordination if present
    te_targets = ["Training & Development", "Event Coordination"]

    synth_rows = []
    # add TE samples quietly
    for cat in te_targets:
        if cat in df_work['category'].unique():
            subs, bods = generate_samples(cat, synth_te)
            for s,b in zip(subs,bods):
                synth_rows.append({"subject": s, "body": b, "text": s + " " + b, "category": cat, "cleaned_text": clean_text(s + " " + b)})

    # add samples for other minority classes
    for cat in minority_classes:
        if cat in te_targets:
            continue
        subs, bods = generate_samples(cat, synth_other)
        for s,b in zip(subs,bods):
            synth_rows.append({"subject": s, "body": b, "text": s + " " + b, "category": cat, "cleaned_text": clean_text(s + " " + b)})

    if synth_rows:
        df_synth = pd.DataFrame(synth_rows)
        df_aug = pd.concat([df_work, df_synth], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        df_aug = df_work

    # TF-IDF
    tfidf = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2), sublinear_tf=True, max_features=40000)
    X = tfidf.fit_transform(df_aug['cleaned_text'].tolist())
    y = df_aug['category'].values

    # stratified split safe check
    vc = pd.Series(y).value_counts()
    if (vc >= 2).all():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_SEED, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_SEED)

    # candidate models
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced'),
        "LinearSVC": LinearSVC(max_iter=5000),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED)
    }

    # quick CV selection
    cv_scores = {}
    for name, mdl in candidates.items():
        try:
            sc = float(np.mean(cross_val_score(mdl, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)))
        except Exception:
            sc = 0.0
        cv_scores[name] = sc

    best_name = max(cv_scores, key=cv_scores.get)
    best_model = candidates[best_name]
    best_model.fit(X_train, y_train)

    # evaluation
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1w = f1_score(y_test, preds, average='weighted')

    # return artifacts (kept in session)
    return {
        "tfidf": tfidf,
        "model": best_model,
        "model_name": best_name,
        "metrics": {"accuracy": acc, "f1_weighted": f1w},
        "report": classification_report(y_test, preds, zero_division=0),
        "confusion": (y_test, preds),
        "augmented_df": df_aug
    }

# -----------------------
# Run update if requested (HR sees only progress and success)
# -----------------------
if update_model_btn:
    with st.spinner("Updating model — this may take a short while. The app will inform you when complete..."):
        artifacts = augment_and_train_internal(df)
        st.session_state['tfidf'] = artifacts['tfidf']
        st.session_state['model'] = artifacts['model']
        st.session_state['model_name'] = artifacts['model_name']
        st.session_state['metrics'] = artifacts['metrics']
        st.session_state['report'] = artifacts['report']
        st.session_state['augmented_df'] = artifacts['augmented_df']
        st.success("Model update complete.")
        st.info(f"Model: {artifacts['model_name']} — Accuracy: {artifacts['metrics']['accuracy']:.3f}, Weighted F1: {artifacts['metrics']['f1_weighted']:.3f}")

# If a model is already present (from prior session), show its status
if 'model' in st.session_state:
    st.info(f"Active model: {st.session_state.get('model_name','(unknown)')} — Ready for predictions.")

# -----------------------
# Classify Email tab
# -----------------------
with classify_tab:
    st.header("Classify a single email")
    st.markdown("Paste the email subject and body below and click **Classify**. The app will show the most appropriate HR category.")
    subj = st.text_input("Subject", placeholder="e.g., Request for Sick Leave")
    body = st.text_area("Body", placeholder="Paste the email body here...")
    if st.button("Classify"):
        if 'model' not in st.session_state or 'tfidf' not in st.session_state:
            st.error("No model loaded. Please click 'Update Model' in the left panel to prepare the classifier.")
        else:
            raw = (subj + " " + body).strip()
            if raw == "":
                st.warning("Please enter a subject or body to classify.")
            else:
                cleaned = clean_text(raw)
                X_inf = st.session_state['tfidf'].transform([cleaned])
                pred = st.session_state['model'].predict(X_inf)[0]
                # display in professional card
                st.markdown('<div class="card"><h3 style="margin:0">Predicted Category</h3><div style="font-size:20px; padding-top:8px;"><b>{}</b></div></div>'.format(pred), unsafe_allow_html=True)
                # optional explanation snippet (top words) — simple heuristic
                st.markdown("**Why this category?**")
                words = cleaned.split()
                if len(words) > 0:
                    common = ", ".join(words[:8])
                    st.write(f"Top words used for this email: {common}")

# -----------------------
# Bulk categorize tab
# -----------------------
with bulk_tab:
    st.header("Bulk categorize emails")
    st.markdown("Upload a CSV of emails (columns: 'subject' and 'body'). The app will return a CSV with predicted categories.")
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_csv is not None:
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
                if 'model' not in st.session_state or 'tfidf' not in st.session_state:
                    st.error("No model available. Please click 'Update Model' in the left panel to prepare the classifier.")
                else:
                    Xb = st.session_state['tfidf'].transform(bdf['cleaned_text'].tolist())
                    bdf['Predicted Category'] = st.session_state['model'].predict(Xb)
                    st.dataframe(bdf.head(50))
                    csv = bdf.to_csv(index=False).encode('utf-8')
                    st.download_button("Download categorized CSV", csv, "categorized_emails.csv", "text/csv")

# -----------------------
# Footer notes
# -----------------------
st.markdown("---")
st.markdown(
    """
    **Notes for HR users**
    - Use **Classify Email** for single emails and **Bulk categorize** to process many emails at once.
    - Click **Update Model** to refresh the classifier with the latest data. This runs automatically without showing technical details.
    - For administrators: to persist a trained model between deployments, train offline and add the saved model to the repository.
    """
)
