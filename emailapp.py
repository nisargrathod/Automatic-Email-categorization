# ==========================================================
# HR EMAIL CLASSIFIER — FINAL (HR KPIs + Guidance)
# Single-file Streamlit app
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
# Page UI Style — CLEAN CORPORATE THEME
# -----------------------------------------------------
st.set_page_config(page_title="HR Email Classifier", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f5f6f8; color:#0f172a; }

    .main-header {
        background: #0b5ed7;
        padding: 22px;
        border-radius: 8px;
        color: white;
        text-align: left;
    }

    .main-header h2 {
        margin: 0;
        font-weight: 600;
    }

    .main-header p {
        margin: 0;
        font-size: 15px;
        opacity: 0.95;
    }

    .card {
        background: white;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 18px;
    }

    .kpi-title { color: #6b7280; font-size:14px; margin:0; }
    .kpi-value { color: #0b5ed7; font-size:22px; font-weight:700; margin:6px 0 0 0; }

    .guide { background: #ffffff; padding: 14px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cleaner professional header
st.markdown(
    """
    <div class="main-header">
        <h2>HR Email Classification System</h2>
        <p>Accurately categorize employee emails using a lightweight ML engine</p>
    </div>
    """,
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
# Text Cleaning
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
        except:
            st.error("Invalid JSON format.")
            return None
    st.error("Dataset not found.")
    return None

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# -----------------------------------------------------
# Category Keywords
# -----------------------------------------------------
CATEGORY_TOPICS = {
    "Event Coordination": ["event","seminar","townhall","venue","logistics"],
    "HR Request": ["experience certificate","id card","relieving","documents"],
    "Compliance": ["policy","tds","tax","audit","form16"],
    "Training & Development": ["training","session","workshop","learning"],
    "Client Communication": ["client","proposal","meeting","feedback"],
    "Leave Management": ["leave","sick","casual","vacation"],
    "Finance": ["reimbursement","invoice","expense","payment"],
    "IT Support": ["login","password","vpn","laptop","network"],
    "Security": ["access","security","unauthorized","breach"],
    "Project Update": ["status","milestone","deadline","sprint","progress"],
    "Payroll / Salary Issues": [
        "salary","salary delay","payslip","salary not received","bonus",
        "arrears","tax deduction","overtime payment","payroll correction","payroll"
    ],
}

CATEGORY_KEYWORDS = {
    cat: set([kw.lower() for kw in kws]) for cat, kws in CATEGORY_TOPICS.items()
}

# -----------------------------------------------------
# Synthetic Generator
# -----------------------------------------------------
GENERIC_TOPICS = ["general", "policy clarification", "update"]
TEMPLATES = [
    "Hello Team,\n\nWe have a session on {topic}. Please join.\n\nThanks,\n{sender}",
    "Hi All,\n\nInvitation: {topic}. Kindly register.\n\nRegards,\n{sender}",
    "Greetings,\n\nUpcoming workshop: {topic}.\n\nBest,\n{sender}",
]
SENDERS = ["Ankit", "Priya", "Rahul", "Neha", "Karan"]

def generate_samples(category, n):
    seeds = CATEGORY_TOPICS.get(category, GENERIC_TOPICS)
    subs, bods = [], []
    for _ in range(n):
        topic = random.choice(seeds)
        template = random.choice(TEMPLATES)
        sender = random.choice(SENDERS)
        body = template.format(topic=topic, sender=sender)
        subs.append(f"{topic.title()}")
        bods.append(body)
    return subs, bods

# -----------------------------------------------------
# Prediction Engine (Boosted)
# -----------------------------------------------------
def predict_with_boost(model, tfidf, raw_text, classes):
    cleaned = clean_text(raw_text)
    X = tfidf.transform([cleaned])

    if hasattr(model, "predict_proba"):
        base = model.predict_proba(X)[0]
    else:
        try:
            df = model.decision_function(X)
            if df.ndim == 1: df = np.vstack([-df, df]).T
            base = softmax(df[0])
        except:
            base = np.ones(len(classes)) / len(classes)

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
    best = (classes[order[0]], float(final[order[0]]))

    return best

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------
st.sidebar.header("Dataset & Model")
uploaded = st.sidebar.file_uploader("Upload dataset (.json)", type=["json"])

if uploaded:
    df = pd.read_json(uploaded)
    st.sidebar.success("Dataset uploaded.")
else:
    df = safe_read_json(DATA_FILE)
    if df is None: st.stop()

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["text"] = df["subject"] + " " + df["body"]
df["cleaned_text"] = df["text"].apply(clean_text)

# -----------------------------------------------------
# Training Engine
# -----------------------------------------------------
def augment_and_train(df_base):

    dfw = df_base.copy()
    counts = dfw["category"].value_counts()
    minority = counts[counts < MINORITY_THRESHOLD_DEFAULT].index.tolist()

    synth_rows = []

    # Payroll boost — SYNTH_PAYROLL_COUNT samples
    subs, bods = generate_samples("Payroll / Salary Issues", SYNTH_PAYROLL_COUNT)
    for s, b in zip(subs, bods):
        synth_rows.append({
            "subject": s, "body": b, "text": s+" "+b,
            "category": "Payroll / Salary Issues",
            "cleaned_text": clean_text(s+" "+b)
        })

    # TE for two classes
    for cat in ["Training & Development", "Event Coordination"]:
        subs, bods = generate_samples(cat, SYNTH_TE_COUNT)
        for s, b in zip(subs, bods):
            synth_rows.append({
                "subject": s, "body": b, "text": s+" "+b,
                "category": cat,
                "cleaned_text": clean_text(s+" "+b)
            })

    # S3 for minority
    for cat in minority:
        if cat in ["Payroll / Salary Issues", "Training & Development", "Event Coordination"]:
            continue
        subs, bods = generate_samples(cat, SYNTH_PER_OTHER_CLASS)
        for s, b in zip(subs, bods):
            synth_rows.append({
                "subject": s, "body": b, "text": s+" "+b,
                "category": cat,
                "cleaned_text": clean_text(s+" "+b)
            })

    df_aug = dfw.copy()
    if synth_rows:
        df_synth = pd.DataFrame(synth_rows)
        df_aug = pd.concat([df_aug, df_synth]).sample(frac=1, random_state=RANDOM_SEED)

    tfidf = TfidfVectorizer(
        max_df=0.9, min_df=1,
        ngram_range=(1,2),
        sublinear_tf=True,
        max_features=40000
    )
    X = tfidf.fit_transform(df_aug["cleaned_text"])
    y = df_aug["category"]

    if (y.value_counts() >= 2).all():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.18, random_state=RANDOM_SEED, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.18, random_state=RANDOM_SEED
        )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced"),
        "LinearSVC": LinearSVC(max_iter=5000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
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
        "model": best_model,
        "tfidf": tfidf,
        "model_name": best_name,
        "metrics": {
            "accuracy": accuracy_score(y_test, preds),
            "f1_weighted": f1_score(y_test, preds, average="weighted")
        },
        "confusion": (y_test, preds),
    }

# -----------------------------------------------------
# Sidebar — Update Model Button
# -----------------------------------------------------
if st.sidebar.button("🔄 Retrain Model"):
    with st.spinner("Training model..."):
        result = augment_and_train(df)
        st.session_state.update(result)
        st.sidebar.success("Model updated successfully!")

# -----------------------------------------------------
# Tabs
# -----------------------------------------------------
tabs = st.tabs(["📊 Dashboard", "📮 Classify Email", "📁 Bulk Categorization"])
dashboard_tab, classify_tab, bulk_tab = tabs

# -----------------------------------------------------
# Dashboard (HR KPIs)
# -----------------------------------------------------
with dashboard_tab:
    st.header("📊 HR Dashboard")

    # Prepare simple KPI metrics useful for HR
    df_dash = df.copy()

    # Payroll Issues count (rule-based detection)
    payroll_keywords = [
        "salary", "not credited", "payslip", "bonus", "arrears",
        "payment delay", "payroll", "salary missing", "salary not received"
    ]
    def has_payroll_issue(text):
        t = str(text).lower()
        return any(kw in t for kw in payroll_keywords)
    df_dash["is_payroll"] = df_dash["text"].apply(has_payroll_issue)
    payroll_count = int(df_dash["is_payroll"].sum())

    # Most frequent category
    if len(df_dash) > 0:
        most_freq_category = df_dash["category"].value_counts().idxmax()
    else:
        most_freq_category = "—"

    # Sentiment overview (simple rule-based)
    positive_words = ["thank", "thanks", "appreciate", "resolved", "good", "happy"]
    negative_words = ["issue", "problem", "delay", "complaint", "not working", "error", "frustrat"]
    def sentiment_score(text):
        t = str(text).lower()
        score = 0
        for w in positive_words:
            if w in t: score += 1
        for w in negative_words:
            if w in t: score -= 1
        return score
    df_dash["sentiment_score"] = df_dash["text"].apply(sentiment_score)
    avg_sent = df_dash["sentiment_score"].mean()
    if avg_sent > 0.5:
        sentiment_label = "Positive"
    elif avg_sent < -0.5:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # Urgency score (0-100) — rule-based
    urgent_keywords = ["urgent", "immediate", "asap", "escalate", "important", "not credited"]
    def urgency_hits(text):
        t = str(text).lower()
        return sum(1 for kw in urgent_keywords if kw in t)
    df_dash["urgency_hits"] = df_dash["text"].apply(urgency_hits)
    # scale to 0-100 (assume ~3 hits = high urgency)
    avg_urg = df_dash["urgency_hits"].mean() if len(df_dash) > 0 else 0.0
    urgency_score = min(100, int((avg_urg / 3.0) * 100))

    # Layout: 4 KPI cards in a row
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f'<div class="card"><div class="kpi-title">Payroll Issues</div><div class="kpi-value">{payroll_count}</div></div>',
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f'<div class="card"><div class="kpi-title">Most Frequent Category</div><div class="kpi-value">{most_freq_category}</div></div>',
            unsafe_allow_html=True
        )

    with k3:
        st.markdown(
            f'<div class="card"><div class="kpi-title">Sentiment Overview</div><div class="kpi-value">{sentiment_label}</div></div>',
            unsafe_allow_html=True
        )

    with k4:
        st.markdown(
            f'<div class="card"><div class="kpi-title">Urgency Score</div><div class="kpi-value">{urgency_score} / 100</div></div>',
            unsafe_allow_html=True
        )

    st.write("")  # small gap

    # Category distribution chart (kept, useful)
    st.subheader("Category Distribution")
    counts = df_dash["category"].value_counts()
    fig, ax = plt.subplots(figsize=(9, max(3, 0.25*len(counts))))
    sns.barplot(y=counts.index, x=counts.values, palette="Blues_d", ax=ax)
    ax.set_xlabel("Count"); ax.set_ylabel("")
    st.pyplot(fig)

    # Word cloud (optional)
    st.subheader("Common Words")
    try:
        corpus = " ".join(df_dash["cleaned_text"].tolist())
        if corpus.strip():
            wc = WordCloud(width=900, height=300, background_color="white", max_words=150).generate(corpus)
            fig2, ax2 = plt.subplots(figsize=(9,3))
            ax2.imshow(wc, interpolation="bilinear"); ax2.axis("off"); st.pyplot(fig2)
        else:
            st.info("Not enough text for word cloud.")
    except Exception:
        st.info("Word cloud unavailable.")

    # Confusion matrix (if available)
    if "model" in st.session_state and "confusion" in st.session_state:
        st.subheader("Confusion Matrix (last training)")
        y_test, preds = st.session_state["confusion"]
        labels = sorted(list(set(y_test)))
        cm = confusion_matrix(y_test, preds, labels=labels)
        fig_cm, ax_cm = plt.subplots(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax_cm)
        st.pyplot(fig_cm)

# -----------------------------------------------------
# Single Email Classification
# -----------------------------------------------------
with classify_tab:
    st.header("📮 Single Email Classification")

    subj = st.text_input("Subject")
    body = st.text_area("Body")

    if st.button("Classify Email"):
        if "model" not in st.session_state:
            st.error("Please retrain the model first.")
        else:
            raw = (subj + " " + body).strip()
            classes = list(st.session_state["model"].classes_)
            pred = predict_with_boost(
                st.session_state["model"],
                st.session_state["tfidf"],
                raw,
                classes,
            )

            st.markdown(
                f"""
                <div class="card">
                    <h3>Predicted Category</h3>
                    <p style="font-size:20px; font-weight:600;">{pred[0]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------------------------------
# Bulk Categorization
# -----------------------------------------------------
with bulk_tab:
    st.header("📁 Bulk Categorization Tool")

    file = st.file_uploader("Upload CSV (subject, body)", type=["csv"])

    if file:
        dfb = pd.read_csv(file)

        if not {"subject","body"}.issubset(dfb.columns):
            st.error("CSV must contain 'subject' and 'body' columns.")
        else:
            if "model" not in st.session_state:
                st.error("Please retrain the model first.")
            else:
                dfb["text"] = dfb["subject"].fillna("") + " " + dfb["body"].fillna("")

                preds = []
                classes = list(st.session_state["model"].classes_)

                for t in dfb["text"]:
                    pred = predict_with_boost(
                        st.session_state["model"],
                        st.session_state["tfidf"],
                        t,
                        classes,
                    )
                    preds.append(pred[0])

                dfb["Predicted Category"] = preds

                st.dataframe(dfb.head(50))

                st.download_button(
                    "Download Results",
                    dfb.to_csv(index=False).encode("utf-8"),
                    "categorized.csv",
                    "text/csv",
                )

# -----------------------------------------------------
# HR Guidance — "How to use" and "What is this for"
# -----------------------------------------------------
st.markdown("---")
st.markdown('<div class="card"><h3 style="margin-top:0">How HR should use this tool</h3>', unsafe_allow_html=True)
st.markdown("""
1. **Retrain Model (when needed):**  
   - Click **🔄 Retrain Model** on the left after uploading new emails (JSON). This refreshes the classifier with latest data and synthetic augmentation for balanced categories.

2. **Use Single Email Classification for Triage:**  
   - Paste subject + body and click **Classify Email** to get an immediate category (e.g., Payroll / Salary Issues). Use this to quickly route incoming emails to the right team.

3. **Use Bulk Categorization for large batches:**  
   - Upload a CSV with `subject` and `body` columns to categorize hundreds of emails at once. Download the results for integration into your ticketing or HRMS.

4. **Interpret the Dashboard KPIs:**  
   - **Payroll Issues:** shows how many emails in the dataset mention payroll-related keywords — helps prioritize payroll triage.  
   - **Most Frequent Category:** shows which HR request type is most common.  
   - **Sentiment Overview:** gives a quick read on overall employee tone in recent emails.  
   - **Urgency Score:** indicates how many requests include urgent language and deserve immediate attention.

5. **Take Action:**  
   - Use the dashboard to identify spikes (e.g., payroll issues) and take immediate steps — open a triage queue, notify payroll team, or schedule a townhall.

6. **Keep improving the model:**  
   - If you see misclassified emails (e.g., Payroll classified as Project Update), collect a small set of correctly labeled emails and add to the JSON dataset, then click Retrain Model.

**Important:** This tool is ML-first with gentle keyword guidance. It helps triage and organize HR workload — decisions and final routing should still be reviewed by HR personnel.
""")

st.markdown("</div>", unsafe_allow_html=True)
