# ==========================================================
# HR EMAIL CLASSIFIER — Professional + Pro Dashboard Features
# Single-file Streamlit app — drop into repo as emailapp.py
# ==========================================================

import os
import re
import string
import random
from datetime import datetime, timedelta
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Page & Theme
# -------------------------
st.set_page_config(page_title="HR Email Classifier — Pro Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; color:#0f1724; }
    .header { background: linear-gradient(90deg,#0b5ed7 0%, #3b82f6 100%); padding:18px; border-radius:10px; color:white; }
    .header h1 { margin:0; font-weight:600; }
    .card { background: white; padding:14px; border-radius:10px; box-shadow: 0 2px 6px rgba(3,10,41,0.06); margin-bottom:14px; }
    .kpi { font-size:22px; font-weight:700; color:#0b5ed7; }
    .muted { color:#6b7280; }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="header"><h1>HR Email Classification — Pro Dashboard</h1><div class="muted">Actionable insights to prioritize HR work</div></div>', unsafe_allow_html=True)
st.write("")

# -------------------------
# Constants
# -------------------------
DATA_FILE = "hr_support_emails_2025_6.json"
RANDOM_SEED = 42
SYNTH_PER_OTHER_CLASS = 30
SYNTH_TE_COUNT = 30
SYNTH_PAYROLL_COUNT = 400
BOOST_WEIGHT = 1.2
MINORITY_THRESHOLD_DEFAULT = 15

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------
# Helpers
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    toks = [t for t in text.split() if len(t) > 2]
    return " ".join(toks)

def safe_read_json(path):
    if os.path.exists(path):
        try:
            return pd.read_json(path)
        except Exception as e:
            st.error(f"Failed to read dataset: {e}")
            return None
    st.error(f"Dataset not found at: {path}")
    return None

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

# -------------------------
# Category topics & keywords (kept concise)
# -------------------------
CATEGORY_TOPICS = {
    "Event Coordination": ["event","seminar","townhall","venue","logistics"],
    "HR Request": ["experience certificate","relieving","onboarding","documents"],
    "Compliance": ["policy","tds","tax","audit","form16"],
    "Training & Development": ["training","workshop","webinar","learning"],
    "Client Communication": ["client","deliverable","feedback","meeting"],
    "Leave Management": ["leave","sick","vacation","approval"],
    "Finance": ["reimbursement","invoice","expense","payment"],
    "IT Support": ["login","password","vpn","laptop","network"],
    "Security": ["access","unauthorized","breach","security"],
    "Project Update": ["status","milestone","deadline","progress","sprint"],
    "Payroll / Salary Issues": [
        "salary","salary not credited","salary delay","payslip","arrears",
        "bonus not received","incorrect salary","bank transfer failed","payroll"
    ],
}
CATEGORY_KEYWORDS = {k:set([w.lower() for w in v]) for k,v in CATEGORY_TOPICS.items()}

# -------------------------
# Synthetic generation templates (internal)
# -------------------------
TEMPLATES = [
    "Hello Team,\n\nWe are running a session on {topic} next week. Please join.\n\nThanks,\n{sender}",
    "Hi All,\n\nInvitation: {topic}. Please register.\n\nRegards,\n{sender}",
    "Dear Colleagues,\n\nUpcoming: {topic}. Please attend.\n\nBest,\n{sender}",
]
SENDER_NAMES = ["Ankit Sharma","Priya Singh","Rahul Verma","Neha Patel","Karan Mehta"]

def generate_samples(category, n):
    seeds = CATEGORY_TOPICS.get(category, ["general"])
    subs, bods = [], []
    for _ in range(n):
        topic = random.choice(seeds)
        body = random.choice(TEMPLATES).format(topic=topic, sender=random.choice(SENDER_NAMES))
        subj = f"{topic.title()}"
        subs.append(subj)
        bods.append(body)
    return subs, bods

# -------------------------
# Prediction + keyword boost
# -------------------------
def predict_with_boost(model, tfidf, raw_text, classes, boost_weight=BOOST_WEIGHT):
    cleaned = clean_text(raw_text)
    X = tfidf.transform([cleaned])
    if hasattr(model, "predict_proba"):
        base = model.predict_proba(X)[0]
    else:
        try:
            df_scores = model.decision_function(X)
            if df_scores.ndim == 1:
                df_scores = np.vstack([-df_scores, df_scores]).T
            base = softmax(df_scores[0])
        except:
            base = np.ones(len(classes))/len(classes)

    words = set(cleaned.split())
    kw_scores = np.zeros(len(classes))
    for i, cat in enumerate(classes):
        kws = CATEGORY_KEYWORDS.get(cat, set())
        matches = 0
        for kw in kws:
            if " " in kw and kw in cleaned:
                matches += 1
            elif kw in words:
                matches += 1
        kw_scores[i] = min(matches / 2.0, 1.0)

    final = base + boost_weight * kw_scores
    final = final / final.sum()
    order = np.argsort(final)[::-1]
    ranked = [(classes[i], float(final[i])) for i in order]
    return ranked[0], ranked

# -------------------------
# Load dataset
# -------------------------
st.sidebar.header("Dataset & Model")
uploaded = st.sidebar.file_uploader("Upload dataset (JSON, optional)", type=["json"])
if uploaded:
    try:
        df = pd.read_json(uploaded)
        st.sidebar.success("Uploaded dataset loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded file: {e}")
        st.stop()
else:
    df = safe_read_json(DATA_FILE)
    if df is None:
        st.stop()

# Validate and prepare
if not {"subject","body","category"}.issubset(df.columns):
    st.error("Dataset must include: subject, body, category")
    st.stop()

df['subject'] = df['subject'].fillna("")
df['body'] = df['body'].fillna("")
df['text'] = (df['subject'] + " " + df['body']).str.strip()
df = df[df['text'].str.len() > 0].reset_index(drop=True)
df['cleaned_text'] = df['text'].apply(clean_text)

# Normalise department if present
has_department = 'department' in df.columns
if has_department:
    df['department'] = df['department'].fillna("Unknown").astype(str)

# Date handling: detect common date columns, else create synthetic timeline (index-based)
date_col = None
for c in ['date','received_at','timestamp','created_at','received','sent_date']:
    if c in df.columns:
        date_col = c
        break

if date_col:
    try:
        df['__date'] = pd.to_datetime(df[date_col], errors='coerce')
        # fallback for nulls -> today's minus index days
        null_mask = df['__date'].isna()
        if null_mask.any():
            fallback_dates = [datetime.today() - timedelta(days=i) for i in range(null_mask.sum())]
            df.loc[null_mask, '__date'] = fallback_dates
    except:
        df['__date'] = pd.to_datetime(datetime.today())
else:
    # create approximate date: distribute messages over the last 90 days by index
    n = len(df)
    today = datetime.today()
    df['__date'] = [today - timedelta(days=int((i/n)*90)) for i in range(n)]

# -------------------------
# Augment & Train engine
# -------------------------
def augment_and_train(df_input,
                      minority_threshold=MINORITY_THRESHOLD_DEFAULT,
                      synth_other=SYNTH_PER_OTHER_CLASS,
                      synth_te=SYNTH_TE_COUNT,
                      synth_payroll=SYNTH_PAYROLL_COUNT):
    dfw = df_input.copy()
    counts = dfw['category'].value_counts()
    minority = counts[counts < minority_threshold].index.tolist()
    synth_rows = []

    # Payroll synthetic always (strong signal)
    payroll_label = "Payroll / Salary Issues"
    subs_p, bods_p = generate_samples(payroll_label, synth_payroll)
    for s,b in zip(subs_p,bods_p):
        synth_rows.append({"subject":s,"body":b,"text":s+" "+b,"category":payroll_label,"cleaned_text":clean_text(s+" "+b)})

    # TE30
    for cat in ["Training & Development","Event Coordination"]:
        if cat in dfw['category'].unique():
            subs, bods = generate_samples(cat, synth_te)
            for s,b in zip(subs,bods):
                synth_rows.append({"subject":s,"body":b,"text":s+" "+b,"category":cat,"cleaned_text":clean_text(s+" "+b)})

    # S3
    for cat in minority:
        if cat in ["Training & Development","Event Coordination",payroll_label]:
            continue
        subs,bods = generate_samples(cat, synth_other)
        for s,b in zip(subs,bods):
            synth_rows.append({"subject":s,"body":b,"text":s+" "+b,"category":cat,"cleaned_text":clean_text(s+" "+b)})

    if synth_rows:
        df_synth = pd.DataFrame(synth_rows)
        df_aug = pd.concat([dfw, df_synth], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        df_aug = dfw

    tfidf = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2), sublinear_tf=True, max_features=40000)
    X = tfidf.fit_transform(df_aug['cleaned_text'].tolist())
    y = df_aug['category'].values

    # stratify if possible
    vc = pd.Series(y).value_counts()
    if (vc >= 2).all():
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.18, random_state=RANDOM_SEED, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.18, random_state=RANDOM_SEED)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced'),
        "LinearSVC": LinearSVC(max_iter=5000),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED)
    }
    cv_scores = {}
    for name, mdl in candidates.items():
        try:
            sc = float(np.mean(cross_val_score(mdl, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)))
        except:
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
        "metrics": {"accuracy":acc, "f1_weighted":f1w},
        "report": classification_report(y_test, preds, zero_division=0),
        "confusion": (y_test, preds),
        "augmented_df": df_aug
    }

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
st.sidebar.info("Upload dataset or use repo dataset. Click 'Retrain Model' after uploading new data.")
if st.sidebar.button("🔄 Retrain Model"):
    with st.spinner("Retraining model — this may take a short while..."):
        artifacts = augment_and_train(df)
        st.session_state.update(artifacts)
        st.sidebar.success(f"Model trained: {artifacts['model_name']} — Acc:{artifacts['metrics']['accuracy']:.3f}")

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Dashboard","Classify Email","Bulk Categorize"])
dashboard_tab, classify_tab, bulk_tab = tabs

# -------------------------
# PRO DASHBOARD (Option C) — cards + charts + pro features
# -------------------------
with dashboard_tab:
    st.header("HR Executive Dashboard")
    st.markdown("Key insights and prioritized actions — designed for HR decision makers.")

    # Prepare dashboard data
    df_dash = df.copy()
    # payroll detection
    payroll_kw = ["salary","payslip","not credited","arrears","salary not received","payroll","salary delay","bonus","salary missing","payment not received"]
    df_dash['is_payroll'] = df_dash['text'].str.lower().apply(lambda t: any(kw in t for kw in payroll_kw))

    payroll_count = int(df_dash['is_payroll'].sum())
    trending_category = df_dash['category'].value_counts().idxmax()

    # sentiment (simple rule-based)
    pos_kw = ["thank","thanks","resolved","appreciate","helpful"]
    neg_kw = ["issue","problem","delay","not","complaint","error","disappointed"]
    def sent_score(text):
        t = text.lower()
        s = 0
        for w in pos_kw:
            if w in t: s += 1
        for w in neg_kw:
            if w in t: s -= 1
        return s
    df_dash['sent_score'] = df_dash['text'].apply(sent_score)
    avg_sent = df_dash['sent_score'].mean()
    if avg_sent > 0.5:
        sentiment_label = "Positive"
    elif avg_sent < -0.5:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # urgency score
    urgent_kw = ["urgent","immediate","asap","escalate","important","not credited"]
    df_dash['urgency_hits'] = df_dash['text'].str.lower().apply(lambda t: sum(1 for kw in urgent_kw if kw in t))
    # scale urgency to 0-100
    avg_urgency = df_dash['urgency_hits'].mean()
    urgency_index = min(100, int((avg_urgency/3)*100))

    # upcoming events (by keywords & date mentions)
    event_kw = ["training","webinar","session","event","workshop","seminar"]
    date_pat = r"\b(20[2-9]\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b"
    df_dash['is_event'] = df_dash['text'].str.lower().apply(lambda t: any(kw in t for kw in event_kw) or re.search(date_pat, t))
    event_count = int(df_dash['is_event'].sum())

    # KPI cards (5)
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="card"><div class="muted">Payroll / Salary Issues</div><div class="kpi">{payroll_count}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><div class="muted">Trending HR Category</div><div class="kpi">{trending_category}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><div class="muted">Employee Sentiment</div><div class="kpi">{sentiment_label}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="card"><div class="muted">Urgency Index</div><div class="kpi">{urgency_index} / 100</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="card"><div class="muted">Upcoming Trainings / Events</div><div class="kpi">{event_count}</div></div>', unsafe_allow_html=True)

    st.write("")  # spacing

    # Category distribution (bar)
    st.subheader("Category Trend")
    counts = df_dash['category'].value_counts()
    fig, ax = plt.subplots(figsize=(9, max(3, 0.28*len(counts))))
    sns.barplot(x=counts.values, y=counts.index, palette="Blues_d", ax=ax)
    ax.set_xlabel("Count"); ax.set_ylabel("")
    st.pyplot(fig)

    # Pro feature: Inbox Timeline
    st.subheader("Inbox Timeline")
    timeline_df = df_dash.copy()
    timeline_df['date_bucket'] = timeline_df['__date'].dt.to_period('W').dt.start_time
    timeline_series = timeline_df.groupby('date_bucket').size().reset_index(name='count')
    if len(timeline_series) > 1:
        fig_t, ax_t = plt.subplots(figsize=(10,3))
        ax_t.plot(timeline_series['date_bucket'], timeline_series['count'], marker='o')
        ax_t.set_xlabel("Week"); ax_t.set_ylabel("Emails")
        ax_t.grid(alpha=0.15)
        st.pyplot(fig_t)
    else:
        st.info("Not enough time-distributed data for timeline.")

    # Pro feature: Department x Category heatmap (if department present)
    if has_department:
        st.subheader("Department × Category Heatmap")
        pivot = pd.crosstab(df_dash['department'], df_dash['category'])
        # keep top 12 departments
        top_depts = pivot.sum(axis=1).sort_values(ascending=False).head(12).index
        pivot_top = pivot.loc[top_depts]
        fig_h, ax_h = plt.subplots(figsize=(10, max(3, 0.25*pivot_top.shape[0])))
        sns.heatmap(pivot_top, cmap="Blues", annot=False, ax=ax_h)
        ax_h.set_xlabel("Category"); ax_h.set_ylabel("Department")
        st.pyplot(fig_h)

    # Pro feature: Sentiment Gauge (simple donut showing negative/neutral/positive ratio)
    st.subheader("Sentiment Breakdown")
    sent_counts = pd.cut(df_dash['sent_score'], bins=[-999, -1, 0, 999], labels=["Negative","Neutral","Positive"]).value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0)
    fig_s, ax_s = plt.subplots(figsize=(5,3))
    ax_s.pie(sent_counts.values, labels=sent_counts.index, autopct='%1.0f%%', startangle=140, wedgeprops=dict(width=0.4))
    ax_s.axis('equal')
    st.pyplot(fig_s)

    # Pro feature: AI-generated Recommended Actions (simple rule-based suggestions)
    st.subheader("Recommended Actions")
    recs = []
    if payroll_count > 5:
        recs.append(f"Prioritize payroll mailbox — {payroll_count} suspected payroll tickets.")
    if urgency_index > 50:
        recs.append("Set up an 'Urgent' triage channel for asap/escalated emails.")
    if event_count >= 3:
        recs.append("Confirm logistics for upcoming trainings/events and notify participants.")
    # trending category action
    recs.append(f"Investigate spike in '{trending_category}' requests and assign owners.")
    # department actions
    if has_department:
        dept_top = df_dash['department'].value_counts().idxmax()
        recs.append(f"Top contributing department: {dept_top} — review common requests for coaching.")
    # add fallback
    if not recs:
        recs = ["No urgent actions detected. Maintain regular SLA."]

    for i, r in enumerate(recs, 1):
        st.markdown(f"**{i}.** {r}")

# -------------------------
# CLASSIFY EMAIL (single)
# -------------------------
with classify_tab:
    st.header("Classify a Single Email")
    subj = st.text_input("Subject", "")
    body = st.text_area("Body", "")
    if st.button("Classify"):
        if 'model' not in st.session_state:
            st.error("No model available. Click 'Retrain Model' in the left panel to train.")
        else:
            raw = (subj + " " + body).strip()
            classes = list(st.session_state['model'].classes_)
            best, ranked = predict_with_boost(st.session_state['model'], st.session_state['tfidf'], raw, classes)
            st.markdown(f'<div class="card"><h3>Predicted Category</h3><p style="font-size:20px; font-weight:600">{best[0]}</p></div>', unsafe_allow_html=True)

# -------------------------
# BULK CATEGORIZE
# -------------------------
with bulk_tab:
    st.header("Bulk Categorize Emails")
    uploaded_csv = st.file_uploader("Upload CSV with columns: subject, body", type=["csv"])
    if uploaded_csv:
        try:
            bdf = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            bdf = None
        if bdf is not None:
            if not {'subject','body'}.issubset(bdf.columns):
                st.error("CSV must contain 'subject' and 'body' columns.")
            else:
                if 'model' not in st.session_state:
                    st.error("Please retrain the model first.")
                else:
                    bdf['text'] = bdf['subject'].fillna("") + " " + bdf['body'].fillna("")
                    preds = []
                    for t in bdf['text'].tolist():
                        best, ranked = predict_with_boost(st.session_state['model'], st.session_state['tfidf'], t, list(st.session_state['model'].classes_))
                        preds.append(best[0])
                    bdf['Predicted Category'] = preds
                    st.dataframe(bdf.head(50))
                    csv = bdf.to_csv(index=False).encode('utf-8')
                    st.download_button("Download categorized CSV", csv, "categorized_emails.csv", "text/csv")

# -------------------------
# Footer (minimal)
# -------------------------
st.markdown("---")
st.markdown('<div class="muted">Pro Dashboard • HR Email Classifier</div>', unsafe_allow_html=True)
