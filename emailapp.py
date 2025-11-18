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
# App config & theme
# -----------------------
st.set_page_config(page_title="HR Email Classifier — Professional", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f6f8fa; color: #0f1724; }
    .header { background: linear-gradient(90deg,#0b5ed7 0%, #3b82f6 100%); color: white; padding: 16px; border-radius: 8px; }
    .card { background: white; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(16,24,40,0.05); margin-bottom: 12px; }
    .muted { color: #6b7280; font-size: 0.95rem; }
    .btn-primary { background-color: #0b5ed7; color: white; padding: 8px 16px; border-radius: 6px; border: none; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="header"><h2 style="margin:0">HR Email Categorization — Professional</h2><div class="muted">Auto-classify incoming HR emails into appropriate categories</div></div>', unsafe_allow_html=True)
st.write("")

# -----------------------
# Global constants
# -----------------------
DATA_FILE = "hr_support_emails_2025_6.json"  # dataset in repo root
RANDOM_SEED = 42
MINORITY_THRESHOLD_DEFAULT = 15
SYNTH_PER_OTHER_CLASS = 30
SYNTH_TE_COUNT = 30   # Training & Event specialized additions
BOOST_WEIGHT = 0.9    # how much keyword matches can nudge the score (0.0 - 1.5). 0.9 is moderate-strong.

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------
# Utility functions
# -----------------------
def safe_read_json(path):
    if os.path.exists(path):
        try:
            return pd.read_json(path)
        except Exception as e:
            st.error(f"Failed to read dataset: {e}")
            return None
    st.error(f"Dataset not found at '{path}'. Upload to repo root or use the uploader.")
    return None

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if len(t) > 2]
    return " ".join(tokens)

# -----------------------
# Category topics & keywords (used for synthetic generation & boosting)
# -----------------------
CATEGORY_TOPICS = {
    "Event Coordination": ["event", "seminar", "townhall", "annual day", "registration", "venue", "logistics", "volunteer"],
    "HR Request": ["experience certificate", "relieving", "onboarding", "id card", "documents", "letter", "application"],
    "Compliance": ["policy", "compliance", "audit", "tds", "tax", "form16", "regulation"],
    "Training & Development": ["training", "workshop", "webinar", "bootcamp", "skill", "coaching", "learning"],
    "Client Communication": ["client", "deliverable", "feedback", "client meeting", "proposal", "client update"],
    "Leave Management": ["leave", "sick", "casual", "vacation", "absence", "approval"],
    "Finance": ["salary", "payslip", "payment", "reimbursement", "bonus", "credit", "arrears", "payroll"],
    "IT Support": ["login", "password", "vpn", "laptop", "email login", "software", "network", "access"],
    "Security": ["suspicious", "access", "security", "unauthorized", "breach", "badge"],
    "Project Update": ["status", "milestone", "deadline", "release", "progress", "sprint", "timesheet"]
}

CATEGORY_KEYWORDS = {cat: set([w.lower() for w in words]) for cat, words in CATEGORY_TOPICS.items()}

# fallback generic topics
GENERIC_TOPICS = ["general HR query", "policy clarification", "document request"]

TEMPLATES = [
    "Hello Team,\n\nWe are running a session on {topic} next week. Please join.\n\nThanks,\n{sender}",
    "Hi All,\n\nInvitation: {topic}. Please register to attend.\n\nRegards,\n{sender}",
    "Dear Colleagues,\n\nDon't miss the workshop on {topic}. It will be practical and hands-on.\n\nBest,\n{sender}",
    "Hello,\n\nWe scheduled a session on {topic}. Please block your calendar.\n\nThanks,\n{sender}"
]

SENDER_NAMES = ["Ankit Sharma", "Priya Singh", "Rahul Verma", "Neha Patel", "Karan Mehta", "Aisha Khan", "Rohit Joshi"]

def generate_samples(category, n):
    seeds = CATEGORY_TOPICS.get(category) or GENERIC_TOPICS
    subs, bods = [], []
    for _ in range(n):
        topic = random.choice(seeds)
        template = random.choice(TEMPLATES)
        sender = random.choice(SENDER_NAMES)
        body = template.format(topic=topic, sender=sender)
        subj_choices = [
            f"{category} — {topic.title()}",
            f"Invitation: {topic.title()}",
            f"{topic.title()}",
            f"{category} | {topic.title()}"
        ]
        subs.append(random.choice(subj_choices))
        bods.append(body)
    return subs, bods

# -----------------------
# Keyword boosting helper
# -----------------------
def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

def predict_with_boost(model, tfidf, raw_text, classes, keyword_map, boost_weight=BOOST_WEIGHT):
    """
    Obtain base scores (probabilities) from model (if possible), fallback to normalized decision_function,
    then add a keyword-based boost per class, and return final top prediction and top-n list.
    """
    cleaned = clean_text(raw_text)
    X = tfidf.transform([cleaned])

    # base scores / probabilities
    if hasattr(model, "predict_proba"):
        base = model.predict_proba(X)[0]  # shape (n_classes,)
    else:
        # use decision_function as proxy, then softmax
        try:
            df_scores = model.decision_function(X)
            if df_scores.ndim == 1:
                # binary case -> convert to two-class softmax
                scores = np.vstack([-df_scores, df_scores]).T
            else:
                scores = df_scores
            base = softmax(scores[0])
        except Exception:
            # fallback: uniform
            base = np.ones(len(classes)) / len(classes)

    # build keyword match scores
    words = set(cleaned.split())
    kw_scores = np.zeros(len(classes), dtype=float)
    for i, cat in enumerate(classes):
        kws = keyword_map.get(cat, set())
        if not kws:
            continue
        # count matched keywords
        matches = sum(1 for w in kws if w in words)
        # shape the match contribution: saturate at 3 matches
        kw_scores[i] = min(matches / 3.0, 1.0)

    # combine: weighted addition (model-first)
    final = base + boost_weight * kw_scores
    # normalize to pseudo-probabilities
    final = final / final.sum()

    # prepare top3 list
    top_idx = np.argsort(final)[::-1]
    top_list = [(classes[i], float(final[i])) for i in top_idx]
    return top_list[0], top_list  # best (cat, prob), full ranked list

# -----------------------
# Load dataset (repo or upload)
# -----------------------
st.sidebar.header("Dataset & Model")
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

# Validate
required = ["subject", "body", "category"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}. Please provide 'subject','body','category'.")
    st.stop()

# Prepare
df['subject'] = df['subject'].fillna("")
df['body'] = df['body'].fillna("")
df['text'] = (df['subject'] + " " + df['body']).str.strip()
df = df[df['text'].str.len() > 0].reset_index(drop=True)
df['cleaned_text'] = df['text'].apply(clean_text)

# sidebar: short instructions & update button
st.sidebar.markdown("**Update Model**")
st.sidebar.info("Click 'Update Model' to refresh classifier with latest data (recommended after uploading new data).")
update_btn = st.sidebar.button("Update Model")

# top cards
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.markdown(f'<div class="card"><h3 style="margin:0">Total emails</h3><div style="font-size:22px;"><b>{len(df)}</b></div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="card"><h3 style="margin:0">Categories</h3><div style="font-size:22px;"><b>{df["category"].nunique()}</b></div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="card"><h3 style="margin:0">Status</h3><div style="font-size:14px; color:#6b7280;">Model not trained yet</div></div>', unsafe_allow_html=True)

# tabs
tabs = st.tabs(["📊 Dashboard", "📮 Classify Email", "📁 Bulk Categorize"])
dashboard_tab, classify_tab, bulk_tab = tabs

# -----------------------
# Dashboard
# -----------------------
with dashboard_tab:
    st.header("Overview")
    st.markdown("A quick summary of recent emails received by HR.")
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
        corpus = " ".join(df['cleaned_text'].tolist())
        if corpus.strip():
            wc = WordCloud(width=900, height=300, background_color='white', max_words=150).generate(corpus)
            fig2, ax2 = plt.subplots(figsize=(8,3))
            ax2.imshow(wc, interpolation='bilinear'); ax2.axis('off'); st.pyplot(fig2)
        else:
            st.info("Not enough text for word cloud.")

    st.markdown("### Sample emails")
    try:
        st.dataframe(df[['subject','category']].sample(min(6,len(df))).rename(columns={'subject':'Subject','category':'Category'}))
    except Exception:
        st.write("No preview available.")

# -----------------------
# Internal augment & train (hidden details)
# -----------------------
def augment_and_train(df_input,
                      minority_threshold=MINORITY_THRESHOLD_DEFAULT,
                      synth_other=SYNTH_PER_OTHER_CLASS,
                      synth_te=SYNTH_TE_COUNT):
    # copy
    dfw = df_input.copy()
    counts = dfw['category'].value_counts()
    minority = counts[counts < minority_threshold].index.tolist()

    # TE targets always get extra samples if present
    te_targets = ["Training & Development", "Event Coordination"]
    synth_rows = []

    # TE specialized additions
    for cat in te_targets:
        if cat in dfw['category'].unique():
            subs, bods = generate_samples(cat, synth_te)
            for s,b in zip(subs,bods):
                synth_rows.append({"subject": s, "body": b, "text": s+" "+b, "category": cat, "cleaned_text": clean_text(s+" "+b)})

    # other minority classes
    for cat in minority:
        if cat in te_targets:
            continue
        subs, bods = generate_samples(cat, synth_other)
        for s,b in zip(subs,bods):
            synth_rows.append({"subject": s, "body": b, "text": s+" "+b, "category": cat, "cleaned_text": clean_text(s+" "+b)})

    if synth_rows:
        df_synth = pd.DataFrame(synth_rows)
        df_aug = pd.concat([dfw, df_synth], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        df_aug = dfw

    # vectorize and train
    tfidf = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2), sublinear_tf=True, max_features=40000)
    X = tfidf.fit_transform(df_aug['cleaned_text'].tolist())
    y = df_aug['category'].values

    # safe stratify
    vc = pd.Series(y).value_counts()
    if (vc >= 2).all():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_SEED, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_SEED)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced'),
        "LinearSVC": LinearSVC(max_iter=5000),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED)
    }

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

    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1w = f1_score(y_test, preds, average='weighted')

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
# Trigger update (HR sees only progress)
# -----------------------
if update_btn:
    with st.spinner("Updating model — this may take a short while..."):
        artifacts = augment_and_train(df)
        st.session_state['tfidf'] = artifacts['tfidf']
        st.session_state['model'] = artifacts['model']
        st.session_state['model_name'] = artifacts['model_name']
        st.session_state['metrics'] = artifacts['metrics']
        st.session_state['report'] = artifacts['report']
        st.session_state['augmented_df'] = artifacts['augmented_df']
        st.success("Model updated successfully.")
        st.info(f"Model: {artifacts['model_name']} — Accuracy: {artifacts['metrics']['accuracy']:.3f}, Weighted F1: {artifacts['metrics']['f1_weighted']:.3f}")

# if model present, update status card
if 'model' in st.session_state:
    st.markdown(f'<div class="card"><h4 style="margin:0">Active model</h4><div style="font-size:16px; padding-top:6px;"><b>{st.session_state.get("model_name","(unknown)")}</b></div></div>', unsafe_allow_html=True)

# -----------------------
# Classify Email tab (HR-friendly)
# -----------------------
with classify_tab:
    st.header("Classify a single email")
    st.markdown("Paste the subject and body of the email and click **Classify**. The app will show the predicted HR category.")
    subj = st.text_input("Subject", placeholder="e.g., Salary Not Credited")
    body = st.text_area("Body", placeholder="Paste the email body here...")
    if st.button("Classify"):
        if 'model' not in st.session_state or 'tfidf' not in st.session_state:
            st.error("No model available. Click 'Update Model' in the left panel to prepare the classifier.")
        else:
            raw = (subj + " " + body).strip()
            if raw == "":
                st.warning("Please enter text to classify.")
            else:
                classes = list(st.session_state['model'].classes_)
                best, ranked = predict_with_boost(st.session_state['model'], st.session_state['tfidf'], raw, classes, CATEGORY_KEYWORDS, boost_weight=BOOST_WEIGHT)
                pred_cat, pred_prob = best[0], best[1] if isinstance(best, tuple) else (best[0], best[1])  # compatibility
                # display neat card
                st.markdown(f'<div class="card"><h3 style="margin:0">Predicted category</h3><div style="font-size:20px; padding-top:8px;"><b>{pred_cat}</b></div></div>', unsafe_allow_html=True)
                st.markdown("**Top suggestions**")
                for c, p in ranked[:5]:
                    st.write(f"- {c} — {p:.2%}")
                st.markdown("**Key words found**")
                hits = [k for k in CATEGORY_KEYWORDS if any(kw in clean_text(raw).split() for kw in CATEGORY_KEYWORDS[k])]
                if hits:
                    st.write(", ".join(hits))
                else:
                    st.write("No strong category-specific keywords detected.")

# -----------------------
# Bulk categorize
# -----------------------
with bulk_tab:
    st.header("Bulk categorize emails")
    st.markdown("Upload a CSV with columns `subject` and `body`. The app will return a CSV with a new column `Predicted Category`.")
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
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
                if 'model' not in st.session_state or 'tfidf' not in st.session_state:
                    st.error("No model available. Click 'Update Model' to prepare the classifier.")
                else:
                    classes = list(st.session_state['model'].classes_)
                    preds = []
                    top3 = []
                    for txt in bdf['text'].tolist():
                        best, ranked = predict_with_boost(st.session_state['model'], st.session_state['tfidf'], txt, classes, CATEGORY_KEYWORDS, boost_weight=BOOST_WEIGHT)
                        preds.append(best[0])
                        top3.append("; ".join([f"{c}:{p:.2%}" for c,p in ranked[:3]]))
                    bdf['Predicted Category'] = preds
                    bdf['Top 3 (cat:prob)'] = top3
                    st.dataframe(bdf.head(50))
                    csv = bdf.to_csv(index=False).encode('utf-8')
                    st.download_button("Download categorized CSV", csv, "categorized_emails.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown(
    """
    **Notes for HR users**
    - Use **Classify Email** for single messages and **Bulk categorize** to process many emails at once.
    - Click **Update Model** after uploading new data to refresh the classifier. The app will augment training data internally to ensure balanced results.
    - The system combines learned patterns with small, safe keyword guidance to improve accuracy for each category.
    """
)
