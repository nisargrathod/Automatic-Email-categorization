# ==========================================================
# HR EMAIL CLASSIFIER — FINAL FULL VERSION (All Fixes Applied)
# ==========================================================

import os
import re
import string
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------
# Page UI Style
# -----------------------------------------------------
st.set_page_config(page_title="HR Email Classifier — Professional", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f6f8fa; color:#0f1724; }
    .header {
        background: linear-gradient(90deg,#0b5ed7 0%, #3b82f6 100%);
        color: white; padding: 16px; border-radius: 8px;
    }
    .card {
        background: white; padding: 12px; border-radius: 8px;
        box-shadow: 0 1px 3px rgba(16,24,40,0.05); margin-bottom: 12px;
    }
    .muted { color: #6b7280; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="header"><h2 style="margin:0">HR Email Categorization — Professional</h2>'
    '<div class="muted">Automatically classify incoming HR emails into categories</div></div>',
    unsafe_allow_html=True,
)
st.write("")

# -----------------------------------------------------
# Constants
# -----------------------------------------------------
DATA_FILE = "hr_support_emails_2025_6.json"
RANDOM_SEED = 42
SYNTH_PER_OTHER_CLASS = 30
SYNTH_TE_COUNT = 30
SYNTH_PAYROLL_COUNT = 400
BOOST_WEIGHT = 1.2
MINORITY_THRESHOLD_DEFAULT = 15

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if len(t) > 2]
    return " ".join(tokens)


def safe_read_json(path):
    if os.path.exists(path):
        try:
            return pd.read_json(path)
        except Exception as e:
            st.error(f"Failed to read dataset: {e}")
            return None
    st.error(f"Dataset not found at path: {path}")
    return None


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# -----------------------------------------------------
# Category topics & keywords
# -----------------------------------------------------
CATEGORY_TOPICS = {
    "Event Coordination": [
        "event", "seminar", "townhall", "registration", "venue", "logistics", "volunteer"
    ],

    "HR Request": [
        "experience certificate", "relieving", "onboarding", "id card", "documents", "letter"
    ],

    "Compliance": [
        "policy", "compliance", "audit", "tds", "tax", "form16", "regulation"
    ],

    "Training & Development": [
        "training", "workshop", "webinar", "bootcamp", "skill", "learning"
    ],

    "Client Communication": [
        "client", "deliverable", "feedback", "meeting", "proposal"
    ],

    "Leave Management": [
        "leave", "sick", "casual", "vacation", "absence", "approval"
    ],

    "Finance": [
        "reimbursement", "invoice", "expense", "travel reimbursement", "payment"
    ],

    "IT Support": [
        "login", "password", "vpn", "laptop", "email login", "software", "network"
    ],

    "Security": [
        "suspicious", "access", "security", "unauthorized", "breach", "badge"
    ],

    # FIX: removed misleading 'update'
    "Project Update": [
        "status", "milestone", "deadline", "release", "progress", "sprint"
    ],

    # FIX: strong payroll keywords
    "Payroll / Salary Issues": [
        "salary", "salary not credited", "salary delay", "payslip",
        "payslip not received", "salary discrepancy", "bonus not received",
        "arrears", "incorrect salary amount", "bank transfer failed",
        "payroll correction", "tax deduction issue", "salary pending",
        "overtime payment", "payment not received", "salary missing",
        "salary not received", "salary clarification", "payroll"
    ],
}

CATEGORY_KEYWORDS = {
    cat: set([kw.lower() for kw in kws]) for cat, kws in CATEGORY_TOPICS.items()
}

# -----------------------------------------------------
# Synthetic generator
# -----------------------------------------------------
GENERIC_TOPICS = ["general HR query", "policy clarification", "document request"]
TEMPLATES = [
    "Hello Team,\n\nWe are running a session on {topic} next week. Please join.\n\nThanks,\n{sender}",
    "Hi All,\n\nInvitation: {topic}. Please register to attend.\n\nRegards,\n{sender}",
    "Dear Colleagues,\n\nDon't miss our workshop on {topic}.\n\nBest,\n{sender}",
    "Hello,\n\nWe scheduled a session on {topic}. Please block your calendar.\n\nThanks,\n{sender}",
]
SENDERS = [
    "Ankit Sharma", "Priya Singh", "Rahul Verma",
    "Neha Patel", "Karan Mehta", "Aisha Khan", "Rohit Joshi"
]


def generate_samples(category, n):
    seeds = CATEGORY_TOPICS.get(category, GENERIC_TOPICS)
    subs, bods = [], []

    for _ in range(n):
        topic = random.choice(seeds)
        template = random.choice(TEMPLATES)
        sender = random.choice(SENDERS)
        body = template.format(topic=topic, sender=sender)

        subj_candidates = [
            f"{category}: {topic.title()}",
            f"{topic.title()}",
            f"Regarding {topic.title()}",
            f"Request: {topic.title()}",
        ]
        subs.append(random.choice(subj_candidates))
        bods.append(body)

    return subs, bods


# -----------------------------------------------------
# Prediction with keyword boosting
# -----------------------------------------------------
def predict_with_boost(model, tfidf, raw_text, classes):
    cleaned = clean_text(raw_text)
    X = tfidf.transform([cleaned])

    # Base model score
    if hasattr(model, "predict_proba"):
        base = model.predict_proba(X)[0]
    else:
        try:
            df = model.decision_function(X)
            if df.ndim == 1:
                df = np.vstack([-df, df]).T
            base = softmax(df[0])
        except:
            base = np.ones(len(classes)) / len(classes)

    # Keyword boost
    words = set(cleaned.split())
    kw_scores = np.zeros(len(classes))

    for i, cat in enumerate(classes):
        matches = 0
        for kw in CATEGORY_KEYWORDS.get(cat, []):
            if " " in kw and kw in cleaned:
                matches += 1
            elif kw in words:
                matches += 1
        kw_scores[i] = min(matches / 2.0, 1.0)

    final = base + BOOST_WEIGHT * kw_scores
    final /= final.sum()

    order = np.argsort(final)[::-1]
    ranked = [(classes[i], float(final[i])) for i in order]
    best = ranked[0]

    return best, ranked


# -----------------------------------------------------
# Load dataset (upload or default file)
# -----------------------------------------------------
st.sidebar.header("Dataset & Model Controls")
uploaded = st.sidebar.file_uploader("Upload dataset JSON (optional)", type=["json"])

if uploaded:
    try:
        df = pd.read_json(uploaded)
        st.sidebar.success("Uploaded dataset loaded.")
    except:
        st.sidebar.error("Invalid JSON file.")
        st.stop()
else:
    df = safe_read_json(DATA_FILE)
    if df is None:
        st.stop()

# Validate
if not {"subject", "body", "category"}.issubset(df.columns):
    st.error("Dataset must contain: subject, body, category")
    st.stop()

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["text"] = df["subject"] + " " + df["body"]
df["cleaned_text"] = df["text"].apply(clean_text)


# -----------------------------------------------------
# Augmentation engine
# -----------------------------------------------------
def augment_and_train(df_base):
    dfw = df_base.copy()
    counts = dfw["category"].value_counts()
    minority = counts[counts < MINORITY_THRESHOLD_DEFAULT].index.tolist()

    synth_rows = []

    # Payroll always added
    payroll_cat = "Payroll / Salary Issues"
    subs_p, bods_p = generate_samples(payroll_cat, SYNTH_PAYROLL_COUNT)
    for s, b in zip(subs_p, bods_p):
        synth_rows.append({
            "subject": s, "body": b, "text": s + " " + b,
            "category": payroll_cat,
            "cleaned_text": clean_text(s + " " + b)
        })

    # TE30
    for cat in ["Training & Development", "Event Coordination"]:
        if cat in dfw["category"].unique():
            subs, bods = generate_samples(cat, SYNTH_TE_COUNT)
            for s, b in zip(subs, bods):
                synth_rows.append({
                    "subject": s, "body": b, "text": s + " " + b,
                    "category": cat,
                    "cleaned_text": clean_text(s + " " + b)
                })

    # S3 for minority
    for cat in minority:
        if cat in ["Training & Development", "Event Coordination", payroll_cat]:
            continue
        subs, bods = generate_samples(cat, SYNTH_PER_OTHER_CLASS)
        for s, b in zip(subs, bods):
            synth_rows.append({
                "subject": s, "body": b, "text": s + " " + b,
                "category": cat,
                "cleaned_text": clean_text(s + " " + b)
            })

    if synth_rows:
        df_synth = pd.DataFrame(synth_rows)
        df_aug = pd.concat([dfw, df_synth], ignore_index=True)
        df_aug = df_aug.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        df_aug = dfw

    tfidf = TfidfVectorizer(
        max_df=0.9,
        min_df=1,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=40000,
    )
    X = tfidf.fit_transform(df_aug["cleaned_text"])
    y = df_aug["category"]

    # Safe stratification
    if (y.value_counts() >= 2).all():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.18, random_state=RANDOM_SEED, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.18, random_state=RANDOM_SEED
        )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, solver="liblinear", class_weight="balanced"
        ),
        "LinearSVC": LinearSVC(max_iter=5000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED
        ),
    }

    scores = {}
    for name, model in models.items():
        try:
            f1 = np.mean(
                cross_val_score(model, X_train, y_train, scoring="f1_weighted", cv=3)
            )
        except:
            f1 = 0
        scores[name] = f1

    best_name = max(scores, key=scores.get)
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)

    return {
        "tfidf": tfidf,
        "model": best_model,
        "model_name": best_name,
        "metrics": {
            "accuracy": accuracy_score(y_test, preds),
            "f1_weighted": f1_score(y_test, preds, average="weighted"),
        },
        "report": classification_report(y_test, preds, zero_division=0),
        "confusion": (y_test, preds),
        "augmented_df": df_aug,
    }


# -----------------------------------------------------
# TOP CARDS
# -----------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<div class="card"><h3>Total Emails</h3><div style="font-size:22px;"><b>{len(df)}</b></div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="card"><h3>Categories</h3><div style="font-size:22px;"><b>{df["category"].nunique()}</b></div></div>',
        unsafe_allow_html=True,
    )
with col3:
    status = (
        f"Model Loaded: {st.session_state.get('model_name', 'Not trained')}"
        if "model" in st.session_state
        else "No model trained"
    )
    st.markdown(
        f'<div class="card"><h3>Status</h3><div class="muted">{status}</div></div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------
# UPDATE MODEL
# -----------------------------------------------------
st.sidebar.markdown("### Update Model")
update_btn = st.sidebar.button("🔄 Update Model Now")

if update_btn:
    with st.spinner("Training model..."):
        result = augment_and_train(df)
        st.session_state.update(result)
        st.success(
            f"Model updated — {result['model_name']} | "
            f"Acc: {result['metrics']['accuracy']:.3f} | "
            f"F1: {result['metrics']['f1_weighted']:.3f}"
        )

# -----------------------------------------------------
# TABS
# -----------------------------------------------------
tabs = st.tabs(["📊 Dashboard", "📮 Classify Email", "📁 Bulk Categorize"])
dashboard_tab, classify_tab, bulk_tab = tabs

# -----------------------------------------------------
# DASHBOARD
# -----------------------------------------------------
with dashboard_tab:
    st.header("📊 HR Analytics Dashboard")

    # Category distribution
    st.subheader("Category Distribution")
    counts = df["category"].value_counts()
    fig, ax = plt.subplots(figsize=(9, max(3, 0.3 * len(counts))))
    sns.barplot(y=counts.index, x=counts.values, palette="Blues_d", ax=ax)
    st.pyplot(fig)

    # Word cloud
    st.subheader("Most Common Words")
    try:
        wc = WordCloud(width=1000, height=300, background_color="white").generate(
            " ".join(df["cleaned_text"])
        )
        fig_wc, ax_wc = plt.subplots(figsize=(10, 3))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    except:
        st.info("Not enough text for word cloud.")

    # Confusion matrix
    if "model" in st.session_state and "confusion" in st.session_state:
        st.subheader("Confusion Matrix")
        y_test, preds = st.session_state["confusion"]
        labels = sorted(list(set(y_test)))
        cm = confusion_matrix(y_test, preds, labels=labels)
        fig_cm, ax_cm = plt.subplots(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax_cm)
        st.pyplot(fig_cm)

# -----------------------------------------------------
# SINGLE EMAIL CLASSIFIER
# -----------------------------------------------------
with classify_tab:
    st.header("📮 Classify a Single Email")
    subj = st.text_input("Subject", placeholder="Salary Not Credited")
    body = st.text_area("Body", placeholder="Paste email body here...")

    if st.button("Classify"):
        if "model" not in st.session_state:
            st.error("Please update the model first.")
        else:
            raw = (subj + " " + body).strip()
            classes = list(st.session_state["model"].classes_)
            best, ranked = predict_with_boost(
                st.session_state["model"],
                st.session_state["tfidf"],
                raw,
                classes,
            )

            pred_cat = best[0]
            st.markdown(
                f'<div class="card"><h3>Predicted Category</h3>'
                f'<div style="font-size:20px;"><b>{pred_cat}</b></div></div>',
                unsafe_allow_html=True,
            )

            st.markdown("**Top Suggestions**")
            for c, p in ranked[:5]:
                st.write(f"- {c} — {p:.1%}")

# -----------------------------------------------------
# BULK CATEGORIZE
# -----------------------------------------------------
with bulk_tab:
    st.header("📁 Bulk Categorize Emails")
    file = st.file_uploader("Upload CSV with subject,body columns", type=["csv"])

    if file:
        dfb = pd.read_csv(file)
        if not {"subject", "body"}.issubset(dfb.columns):
            st.error("CSV missing subject/body columns.")
        else:
            if "model" not in st.session_state:
                st.error("Please update the model first.")
            else:
                dfb["text"] = dfb["subject"].fillna("") + " " + dfb["body"].fillna("")
                preds = []
                top3 = []
                classes = list(st.session_state["model"].classes_)
                for t in dfb["text"]:
                    best, ranked = predict_with_boost(
                        st.session_state["model"],
                        st.session_state["tfidf"],
                        t,
                        classes,
                    )
                    preds.append(best[0])
                    top3.append("; ".join([f"{c}:{p:.2%}" for c, p in ranked[:3]]))

                dfb["Predicted Category"] = preds
                dfb["Top 3"] = top3

                st.dataframe(dfb.head(50))
                st.download_button(
                    "Download Categorized CSV",
                    dfb.to_csv(index=False).encode("utf-8"),
                    "categorized_emails.csv",
                    "text/csv",
                )

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.markdown("---")
st.markdown(
    """
    **Usage Tips**
    - Use “Classify Email” for a quick check.
    - Use “Bulk Categorize” for large sets of inbox messages.
    - Click “Update Model” frequently to keep the classifier accurate.
    - Add more real emails (especially salary/payroll) into dataset for even better accuracy.
    """
)
