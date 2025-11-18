# =====================================================
# HR EMAIL CLASSIFIER — PRO-MAX (UI-2..UI-5 combined)
# Single-file Streamlit app — drop into emailapp.py
# =====================================================

import os, re, random, string
from collections import Counter, deque
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="HR Email Classifier — Pro-Max", layout="wide", initial_sidebar_state="expanded")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_FILE = "hr_support_emails_2025_6.json"

# augmentation / synth
MINORITY_THRESHOLD_DEFAULT = 15
SYNTH_PER_OTHER_CLASS = 30
SYNTH_TE_COUNT = 30
SYNTH_PAYROLL_COUNT = 400
BOOST_WEIGHT = 1.2

# ---------------------------
# THEMES (UI-2..UI-4)
# ---------------------------
THEMES = {
    "Modern Gradient": {
        "bg":"linear-gradient(90deg,#6a11cb 0%, #2575fc 100%)",
        "panel_bg":"rgba(255,255,255,0.96)",
        "text_color":"#0f1724",
        "muted":"#334155"
    },
    "Corporate Clean": {
        "bg":"#f6f8fa",
        "panel_bg":"#ffffff",
        "text_color":"#0f1724",
        "muted":"#6b7280"
    },
    "Dark Professional": {
        "bg":"#0b1220",
        "panel_bg":"#0f1724",
        "text_color":"#e6eef8",
        "muted":"#9aa6bb"
    },
    "Minimal White": {
        "bg":"#ffffff",
        "panel_bg":"#ffffff",
        "text_color":"#0b1220",
        "muted":"#475569"
    }
}

# ---------------------------
# Theme selector in sidebar
# ---------------------------
st.sidebar.title("Pro-Max Controls")
theme_choice = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=1)
theme = THEMES[theme_choice]

# optional logo url
logo_url = st.sidebar.text_input("Logo URL (optional)", value="")

# small layout tweaks via CSS per theme
st.markdown(f"""
    <style>
      .app-header {{
        background: {theme['bg']};
        padding: 18px;
        border-radius: 10px;
        color: white;
      }}
      .panel {{
        background: {theme['panel_bg']};
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 6px 18px rgba(16,24,40,0.06);
      }}
      .muted {{ color: {theme['muted']}; }}
      .big-num {{ font-size: 22px; font-weight:700; color: {theme['text_color']}; }}
      .card-title {{ font-size:14px; color:{theme['muted']}; margin:0 0 6px 0; }}
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header (clean & professional)
# ---------------------------
logo_html = f'<img src="{logo_url}" style="height:48px; margin-right:12px; border-radius:6px;">' if logo_url else ""
st.markdown(f"""
<div class="app-header" style="display:flex; align-items:center;">
  <div style="flex:1; display:flex; align-items:center;">
    {logo_html}
    <div>
      <div style="font-size:20px; font-weight:700">HR Email Classifier — Pro-Max</div>
      <div style="font-size:13px; opacity:0.95">Accurate email triage for HR teams — Payroll detection, Events, IT, Leave & more</div>
    </div>
  </div>
  <div style="text-align:right; min-width:260px;">
    <div style="font-size:12px; color:rgba(255,255,255,0.9)"><b>Theme:</b> {theme_choice}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------------------
# Utilities
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if len(t)>2]
    return " ".join(tokens)

def safe_read_json(path):
    if os.path.exists(path):
        try:
            return pd.read_json(path)
        except Exception as e:
            st.sidebar.error(f"Failed to load JSON: {e}")
            return None
    st.sidebar.warning("Dataset not found in repo root. Upload via sidebar or add file to repo.")
    return None

# ---------------------------
# Category keywords + seeds (all enhanced)
# ---------------------------
CATEGORY_TOPICS = {
    "Event Coordination": ["event","seminar","townhall","registration","venue","logistics","volunteer"],
    "HR Request": ["experience certificate","relieving","onboarding","id card","documents","letter"],
    "Compliance": ["policy","compliance","audit","tds","tax","form16","regulation"],
    "Training & Development": ["training","workshop","webinar","bootcamp","skill","learning"],
    "Client Communication": ["client","deliverable","feedback","meeting","proposal"],
    "Leave Management": ["leave","sick","casual","vacation","absence","approval"],
    "Finance": ["reimbursement","invoice","expense","travel reimbursement","payment"],
    "IT Support": ["login","password","vpn","laptop","email login","software","network"],
    "Security": ["suspicious","access","security","unauthorized","breach","badge"],
    "Project Update": ["status","milestone","deadline","release","progress","sprint"],
    "Payroll / Salary Issues": [
        "salary","salary not credited","salary delay","payslip",
        "payslip not received","salary discrepancy","bonus not received",
        "arrears","incorrect salary amount","bank transfer failed",
        "payroll correction","tax deduction issue","salary pending",
        "overtime payment","payment not received","salary missing",
        "salary not received","salary clarification","payroll"
    ],
}

CATEGORY_KEYWORDS = {k:set([w.lower() for w in v]) for k,v in CATEGORY_TOPICS.items()}

# ---------------------------
# Synthetic generator
# ---------------------------
GENERIC_TEMPLATES = [
    "Hello Team,\n\nWe are conducting a {topic} on {date}. Kindly attend.\n\nThanks,\n{sender}",
    "Hi,\n\nInvitation: {topic}. Please register.\n\nRegards,\n{sender}",
    "Greetings,\n\nThere will be a session about {topic} next week.\n\nBest,\n{sender}"
]
SENDER_POOL = ["Ankit Sharma","Priya Singh","Rahul Verma","Neha Patel","Karan Mehta","Aisha Khan"]

def generate_samples(category, n):
    seeds = CATEGORY_TOPICS.get(category, ["general HR"])
    subs,bods = [],[]
    for _ in range(n):
        topic = random.choice(seeds)
        sender = random.choice(SENDER_POOL)
        template = random.choice(GENERIC_TEMPLATES)
        # random date-ish text to diversify
        date = f"{random.randint(1,28)} {random.choice(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])}"
        body = template.format(topic=topic, date=date, sender=sender)
        subj = f"{topic.title()}"
        subs.append(subj)
        bods.append(body)
    return subs,bods

# ---------------------------
# TF-IDF + models + boosting
# ---------------------------
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict_with_boost(model, tfidf, raw_text, classes, boost_weight=BOOST_WEIGHT):
    cleaned = clean_text(raw_text)
    X = tfidf.transform([cleaned])
    # base scores
    if hasattr(model, "predict_proba"):
        base = model.predict_proba(X)[0]
    else:
        try:
            df = model.decision_function(X)
            if df.ndim == 1:
                df = np.vstack([-df, df]).T
            base = softmax(df[0])
        except:
            base = np.ones(len(classes))/len(classes)
    # keyword nudge
    words = set(cleaned.split())
    kw = np.zeros(len(classes))
    for i,c in enumerate(classes):
        matches = 0
        for k in CATEGORY_KEYWORDS.get(c,[]):
            if " " in k and k in cleaned:
                matches += 1
            elif k in words:
                matches += 1
        kw[i] = min(matches/2.0, 1.0)
    final = base + boost_weight * kw
    final = final / final.sum()
    order = np.argsort(final)[::-1]
    ranked = [(classes[i], float(final[i])) for i in order]
    return ranked[0], ranked

# ---------------------------
# Load dataset (repo or upload)
# ---------------------------
st.sidebar.subheader("Dataset")
uploaded = st.sidebar.file_uploader("Upload dataset JSON (optional)", type=["json"])
if uploaded:
    try:
        df = pd.read_json(uploaded)
        st.sidebar.success("Uploaded dataset loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded JSON: {e}")
        st.stop()
else:
    df = safe_read_json(DATA_FILE)
    if df is None:
        st.stop()

# required fields check
if not {"subject","body","category"}.issubset(set(df.columns)):
    st.error("Dataset must contain columns: subject, body, category")
    st.stop()

# optional department field; if not present we create a dummy distribution
if "department" not in df.columns:
    # create a balanced-ish pseudo-department column for analytics demo
    depts = ["Operations","IT","Development","Data Science","Marketing","Product","QA","Finance","HR","Admin","Sales","Support"]
    df["department"] = [random.choice(depts) for _ in range(len(df))]

# text fields
df['subject'] = df['subject'].fillna("")
df['body'] = df['body'].fillna("")
df['text'] = (df['subject'] + " " + df['body']).str.strip()
df = df[df['text'].str.len()>0].reset_index(drop=True)
df['cleaned_text'] = df['text'].apply(clean_text)

# ---------------------------
# Augment & train function (hidden details)
# ---------------------------
def augment_and_train(df_in,
                      minority_threshold=MINORITY_THRESHOLD_DEFAULT,
                      synth_other=SYNTH_PER_OTHER_CLASS,
                      synth_te=SYNTH_TE_COUNT,
                      synth_payroll=SYNTH_PAYROLL_COUNT):
    dfw = df_in.copy()
    counts = dfw['category'].value_counts()
    minority = counts[counts < minority_threshold].index.tolist()
    synth_rows = []

    # Add payroll always
    payroll_label = "Payroll / Salary Issues"
    subs,bods = generate_samples(payroll_label, synth_payroll)
    for s,b in zip(subs,bods):
        synth_rows.append({"subject":s, "body":b, "text": s+" "+b, "category": payroll_label, "cleaned_text": clean_text(s+" "+b)})

    # TE30 for two categories
    for cat in ["Training & Development", "Event Coordination"]:
        if cat in dfw['category'].unique():
            subs,bods = generate_samples(cat, synth_te)
            for s,b in zip(subs,bods):
                synth_rows.append({"subject":s, "body":b, "text": s+" "+b, "category": cat, "cleaned_text": clean_text(s+" "+b)})

    # S3 for minority other categories
    for cat in minority:
        if cat in ["Training & Development", "Event Coordination", payroll_label]:
            continue
        subs,bods = generate_samples(cat, synth_other)
        for s,b in zip(subs,bods):
            synth_rows.append({"subject":s, "body":b, "text": s+" "+b, "category": cat, "cleaned_text": clean_text(s+" "+b)})

    if synth_rows:
        df_synth = pd.DataFrame(synth_rows)
        df_aug = pd.concat([dfw, df_synth], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        df_aug = dfw

    tfidf = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2), sublinear_tf=True, max_features=40000)
    X = tfidf.fit_transform(df_aug['cleaned_text'].tolist())
    y = df_aug['category'].values

    # safe split
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

    return {"tfidf":tfidf, "model":best_model, "model_name":best_name, "metrics":{"accuracy":acc,"f1_weighted":f1w}, "report": classification_report(y_test,preds, zero_division=0), "confusion":(y_test,preds), "augmented_df": df_aug}

# ---------------------------
# Session storage initialization
# ---------------------------
if 'corrections' not in st.session_state:
    st.session_state['corrections'] = []   # store (text, correct_label)
if 'inbox_queue' not in st.session_state:
    st.session_state['inbox_queue'] = deque(maxlen=500)

# ---------------------------
# Sidebar controls & retrain
# ---------------------------
st.sidebar.markdown("### Model & Data")
if st.sidebar.button("🔄 Update / Retrain Model"):
    with st.spinner("Retraining model (this may take ~30-60s)..."):
        artifacts = augment_and_train(df)
        st.session_state['tfidf'] = artifacts['tfidf']
        st.session_state['model'] = artifacts['model']
        st.session_state['model_name'] = artifacts['model_name']
        st.session_state['metrics'] = artifacts['metrics']
        st.session_state['report'] = artifacts['report']
        st.session_state['confusion'] = artifacts['confusion']
        st.session_state['augmented_df'] = artifacts['augmented_df']
        st.sidebar.success(f"Model updated — {artifacts['model_name']} | Acc {artifacts['metrics']['accuracy']:.3f}")

# quick model info
if 'model' in st.session_state:
    st.sidebar.markdown(f"**Active model:** {st.session_state.get('model_name')}")
    st.sidebar.markdown(f"**Acc:** {st.session_state.get('metrics',{}).get('accuracy','-'):.3f}  •  **F1:** {st.session_state.get('metrics',{}).get('f1_weighted','-'):.3f}")

# department filter
departments = sorted(df['department'].unique().tolist())
dept_filter = st.sidebar.multiselect("Filter departments (multi)", options=departments, default=[])

# inbox simulator controls
st.sidebar.markdown("---")
st.sidebar.markdown("### Inbox Simulator")
sim_mode = st.sidebar.selectbox("Simulator mode", ["Random samples","Generate custom","From dataset"])
sim_count = st.sidebar.slider("Items to simulate", 1, 50, 6)
if st.sidebar.button("Load simulator items"):
    samples = []
    if sim_mode == "Random samples":
        # use some sample emails built from dataset + synth
        idxs = np.random.choice(len(df), min(len(df), sim_count), replace=False)
        for i in idxs:
            samples.append({"subject":df.loc[i,'subject'], "body":df.loc[i,'body'], "category":df.loc[i,'category'], "source":"dataset"})
    elif sim_mode == "Generate custom":
        # generate synth from a random category
        cats = list(CATEGORY_TOPICS.keys())
        for _ in range(sim_count):
            c = random.choice(cats)
            s,b = generate_samples(c,1)
            samples.append({"subject": s[0], "body": b[0], "category": c, "source":"synth"})
    else:
        # sample from dataset by department filter
        pool = df if not dept_filter else df[df['department'].isin(dept_filter)]
        if pool.empty:
            st.sidebar.warning("No dataset rows for selected departments.")
        else:
            idxs = np.random.choice(pool.index, min(len(pool), sim_count), replace=False)
            for i in idxs:
                samples.append({"subject":df.loc[i,'subject'], "body":df.loc[i,'body'], "category":df.loc[i,'category'], "source":"dataset"})
    # push to queue
    for s in samples:
        st.session_state['inbox_queue'].append(s)
    st.sidebar.success(f"Loaded {len(samples)} messages into simulator queue.")

# ---------------------------
# Main layout with tabs
# ---------------------------
tabs = st.tabs(["Dashboard","Classify Email","Bulk Categorize","Inbox Simulator","Department Analytics","Admin"])
tab_dashboard, tab_single, tab_bulk, tab_sim, tab_dept, tab_admin = tabs

# ---------------------------
# Dashboard (UI-3 polished)
# ---------------------------
with tab_dashboard:
    st.header("📊 HR Overview Dashboard")
    st.markdown("Insights from employee communication to help HR manage operations efficiently.")

    df_dash = df.copy()

    # -------- Payroll Issue Count --------
    payroll_keywords = [
        "salary", "not credited", "payslip", "bonus", "arrears",
        "increment", "hike", "payment delay", "payroll", "ctc",
        "overtime", "wage", "deduction"
    ]

    def has_payroll_issue(text):
        text = text.lower()
        return any(kw in text for kw in payroll_keywords)

    df_dash["is_payroll"] = df_dash["text"].apply(has_payroll_issue)
    payroll_count = df_dash["is_payroll"].sum()

    # -------- Trending Category --------
    trending_category = df_dash["category"].value_counts().idxmax()

    # -------- Sentiment Score --------
    positive_words = ["thank you", "appreciate", "good", "resolved", "great"]
    negative_words = ["issue", "error", "not working", "delay", "problem", "complaint"]

    def sentiment_score(text):
        text = text.lower()
        score = 0
        for w in positive_words:
            if w in text: score += 1
        for w in negative_words:
            if w in text: score -= 1
        return score

    df_dash["sentiment"] = df_dash["text"].apply(sentiment_score)
    avg_sentiment = df_dash["sentiment"].mean()

    sentiment_label = (
        "Positive 😀" if avg_sentiment > 0.5
        else "Negative 😟" if avg_sentiment < -0.5
        else "Neutral 😐"
    )

    # -------- Urgency Score --------
    urgent_keywords = ["urgent", "immediate", "asap", "not credited", "delay", "escalate"]

    def urgency_level(text):
        text = text.lower()
        return sum(kw in text for kw in urgent_keywords)

    df_dash["urgency"] = df_dash["text"].apply(urgency_level)
    urgency_score = int((df_dash["urgency"].mean() / 3) * 100)
    urgency_score = min(100, urgency_score)

    # -------- Upcoming Training/Event --------
    event_keywords = ["training", "session", "webinar", "meeting", "event", "workshop"]

    date_pattern = r"\b(2025|2026|\bJan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b"

    def has_event(text):
        text = text.lower()
        return any(kw in text for kw in event_keywords) or re.search(date_pattern, text)

    df_dash["event_flag"] = df_dash["text"].apply(has_event)
    event_count = df_dash["event_flag"].sum()

    # -------- KPI CARDS LAYOUT --------
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(
            f"""
            <div class="card">
                <h4>Payroll Issues</h4>
                <h3 style="color:#0b5ed7">{payroll_count}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f"""
            <div class="card">
                <h4>Trending Category</h4>
                <h3 style="color:#0b5ed7">{trending_category}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f"""
            <div class="card">
                <h4>Sentiment</h4>
                <h3 style="color:#0b5ed7">{sentiment_label}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
            <div class="card">
                <h4>Urgency Level</h4>
                <h3 style="color:#0b5ed7">{urgency_score} / 100</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k5:
        st.markdown(
            f"""
            <div class="card">
                <h4>Upcoming Events / Trainings</h4>
                <h3 style="color:#0b5ed7">{event_count}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------- Category Distribution Chart --------
    st.subheader("📌 Category Trend")

    counts = df["category"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4 + len(counts)*0.25))
    sns.barplot(x=counts.values, y=counts.index, palette="Blues_d", ax=ax)
    st.pyplot(fig)

# ---------------------------
# Single classify (simple, no suggestions)
# ---------------------------
with tab_single:
    st.subheader("Classify a single email")
    s_subj = st.text_input("Subject", key="single_subj")
    s_body = st.text_area("Body", key="single_body")
    if st.button("Classify", key="classify_btn"):
        if 'model' not in st.session_state:
            st.error("Model not trained. Click 'Update / Retrain Model' in sidebar.")
        else:
            text = (s_subj + " " + s_body).strip()
            if not text:
                st.warning("Please enter subject or body.")
            else:
                classes = list(st.session_state['model'].classes_)
                best, ranked = predict_with_boost(st.session_state['model'], st.session_state['tfidf'], text, classes)
                pred_label = best[0]
                conf = best[1] if isinstance(best, tuple) else best[1]
                # display only predicted category (clean professional card)
                st.markdown(f'<div class="card"><h3 style="margin:0">Predicted Category</h3><div style="font-size:18px; font-weight:700;">{pred_label}</div></div>', unsafe_allow_html=True)

# ---------------------------
# Bulk categorize
# ---------------------------
with tab_bulk:
    st.subheader("Bulk categorize (CSV)")
    uploaded_csv = st.file_uploader("Upload CSV (subject,body)", type=["csv"], key="bulk")
    if uploaded_csv is not None:
        bdf = pd.read_csv(uploaded_csv)
        if not {'subject','body'}.issubset(bdf.columns):
            st.error("CSV must contain 'subject' and 'body' columns.")
        else:
            if 'model' not in st.session_state:
                st.error("Model not trained.")
            else:
                bdf['text'] = bdf['subject'].fillna("") + " " + bdf['body'].fillna("")
                preds = []
                for t in bdf['text'].tolist():
                    best, ranked = predict_with_boost(st.session_state['model'], st.session_state['tfidf'], t, list(st.session_state['model'].classes_))
                    preds.append(best[0])
                bdf['Predicted Category'] = preds
                st.dataframe(bdf.head(50))
                csv = bdf.to_csv(index=False).encode('utf-8')
                st.download_button("Download results", csv, "categorized_bulk.csv", "text/csv")

# ---------------------------
# Inbox Simulator (UI-4)
# ---------------------------
with tab_sim:
    st.subheader("Inbox Simulator — review & label (collect corrections)")
    qlen = len(st.session_state['inbox_queue'])
    st.info(f"Simulator queue: {qlen} items (use sidebar loader)")

    if qlen == 0:
        st.write("Load items from sidebar (Random / Generated / From dataset).")
    else:
        # show first item
        item = st.session_state['inbox_queue'].popleft()
        subj = item.get('subject','')
        body = item.get('body','')
        orig_cat = item.get('category','')
        st.markdown(f"**Subject:** {subj}")
        st.write(body)
        # predicted (if model exists)
        if 'model' in st.session_state:
            best, ranked = predict_with_boost(st.session_state['model'], st.session_state['tfidf'], subj+" "+body, list(st.session_state['model'].classes_))
            st.markdown(f"**Predicted:** {best[0]} — {best[1]:.1%}")
        else:
            st.markdown("**Predicted:** (model not trained)")

        # accept / correct UI
        cols = st.columns([1,1,1])
        with cols[0]:
            if st.button("✅ Accept classification"):
                st.success("Marked accepted.")
        with cols[1]:
            # choose correct label
            new_label = st.selectbox("Correct label", options=sorted(df['category'].unique().tolist()+["Payroll / Salary Issues"]))
            if st.button("✍️ Save correction"):
                st.session_state['corrections'].append({"subject":subj,"body":body,"correct_label":new_label})
                st.success("Saved correction sample.")
        with cols[2]:
            if st.button("⏭ Skip"):
                st.info("Skipped.")

    # export corrections
    if st.session_state['corrections']:
        if st.button("Export corrections CSV"):
            cor = pd.DataFrame(st.session_state['corrections'])
            st.download_button("Download corrections", cor.to_csv(index=False).encode('utf-8'), "corrections.csv", "text/csv")
            st.success("Corrections ready for download.")

# ---------------------------
# Department Analytics (UI-5)
# ---------------------------
with tab_dept:
    st.subheader("Department Analytics")
    dept_sel = st.selectbox("Department", options=["All"]+departments)
    filt = df if dept_sel=="All" else df[df['department']==dept_sel]
    st.markdown(f"Showing {len(filt)} emails for: **{dept_sel}**")
    # simple pivot
    pivot = filt['category'].value_counts().reset_index().rename(columns={'index':'category','category':'count'})
    fig, ax = plt.subplots(figsize=(8, max(3,0.25*len(pivot))))
    sns.barplot(y=pivot['category'], x=pivot['count'], ax=ax, palette="Blues_d")
    st.pyplot(fig)

# ---------------------------
# Admin tab (advanced)
# ---------------------------
with tab_admin:
    st.subheader("Admin / Maintenance")
    st.markdown("Advanced controls and reports.")
    if 'model' in st.session_state:
        st.write("Active model:", st.session_state['model_name'])
        st.write("Metrics:", st.session_state.get('metrics',{}))
        if st.button("Show classification report"):
            st.text(st.session_state.get('report','(none)'))
    st.markdown("You can retrain the model, export augmented dataset, or upload a saved model bundle (future).")
    if st.button("Export augmented dataset (current)"):
        if 'augmented_df' in st.session_state:
            df_aug = st.session_state['augmented_df']
            st.download_button("Download augmented dataset", df_aug.to_csv(index=False).encode('utf-8'), "augmented_dataset.csv", "text/csv")
        else:
            st.warning("No augmented dataset available (train first).")

# ---------------------------
# End
# ---------------------------
st.markdown("---")
st.markdown("<div style='text-align:center; color:rgba(0,0,0,0.45)'>Pro-Max App — HR Email Classifier • Built for demo & production</div>", unsafe_allow_html=True)
