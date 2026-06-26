import streamlit as st
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import loguniform
import pandas as pd
import os

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fact Classifier | SVM + T5",
    page_icon="🔬",
    layout="centered",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(14, 165, 233, 0.22), transparent 26%),
            radial-gradient(circle at top right, rgba(20, 184, 166, 0.16), transparent 22%),
            linear-gradient(180deg, #081120 0%, #0f172a 40%, #111827 100%);
    }
    .main .block-container {
        max-width: 1040px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        color: #e5e7eb;
    }
    .stSidebar {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div {
        color: #e5e7eb;
    }
    h1, h2, h3, h4, .stMarkdown, .stCaption {
        letter-spacing: -0.01em;
    }
    .hero-card,
    .panel-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(30, 41, 59, 0.92));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
        padding: 1.25rem 1.35rem;
        backdrop-filter: blur(10px);
        color: #f8fafc;
    }
    .content-card {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
        padding: 1rem 1.1rem 1.15rem;
        margin-top: 1rem;
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 0.35rem;
        color: #ffffff;
    }
    .hero-subtitle {
        color: #cbd5e1;
        margin-bottom: 0.9rem;
        font-size: 1rem;
    }
    .section-label {
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #67e8f9;
        margin-bottom: 0.4rem;
        font-weight: 700;
    }
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.28);
        background: rgba(15, 23, 42, 0.94);
        color: #f8fafc;
    }
    .stTextArea label {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    .stTextArea textarea::placeholder {
        color: #94a3b8;
    }
    .stButton button {
        border-radius: 12px;
        font-weight: 600;
        padding-top: 0.65rem;
        padding-bottom: 0.65rem;
        border: 1px solid rgba(148, 163, 184, 0.28);
    }
    div[data-testid="stButton"] button {
        background: rgba(30, 41, 59, 0.96);
        color: #f8fafc;
    }
    div[data-testid="stButton"] button:hover {
        background: rgba(51, 65, 85, 1);
        color: #ffffff;
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #0f766e, #1d4ed8);
        border: none;
        color: #ffffff;
    }
    .stButton button[kind="primary"]:hover {
        filter: brightness(1.04);
    }
    .verdict-fact {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 1px solid #4ade80;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        color: #14532d;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .verdict-nonfact {
        background: linear-gradient(135deg, #ffe4e6, #fecdd3);
        border: 1px solid #fb7185;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        color: #9f1239;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .tag-fact {
        background:#dcfce7; color:#14532d;
        border:1px solid #4ade80; border-radius:999px;
        padding:3px 10px; font-size:12px; margin-right:4px;
    }
    .tag-nonfact {
        background:#ffe4e6; color:#9f1239;
        border:1px solid #fb7185; border-radius:999px;
        padding:3px 10px; font-size:12px; margin-right:4px;
    }
    .mono { font-family: monospace; font-size: 12px; color: #cbd5e1; }
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 0.75rem 0.9rem;
        color: #f8fafc;
    }
    .stDataFrame {
        background: rgba(15, 23, 42, 0.88);
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.16);
    }
    .example-note {
        color: #e2e8f0;
        font-size: 0.86rem;
        line-height: 1.35;
        margin: 0.45rem 0 0.85rem;
        min-height: 2.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Embedding helper ─────────────────────────────────────────────────────────
def attention_mean_pool(model, tokenizer, texts, batch_size=32):
    """T5 attention mean pooling — matches your training exactly."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            output = model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1)
        emb = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        all_embeddings.append(emb.cpu().numpy())
    return np.vstack(all_embeddings)


# ─── Load T5 encoder (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading T5 encoder…")
def load_t5():
    local_dir = "./t5_cache"
    tokenizer = T5Tokenizer.from_pretrained("t5-base", cache_dir=local_dir)
    encoder   = T5EncoderModel.from_pretrained("t5-base", cache_dir=local_dir).to(DEVICE)
    encoder.eval()
    return tokenizer, encoder


# ─── Load or train SVM (cached) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading / training SVM…")
def load_svm(_tokenizer, _encoder):
    """
    Priority:
    1. Pre-saved embeddings (x_train_t5.npy + y_train.npy) in working dir
    2. data1300.csv  → embed on the fly → train SVM
    3. Tiny fallback demo SVM trained on 10 hard-coded examples
    """

    # ── Option 1: pre-saved embeddings ──────────────────────────────────────
    if os.path.exists("x_train_t5.npy") and os.path.exists("y_train.npy"):
        x_train = np.load("x_train_t5.npy")
        y_train  = np.load("y_train.npy")
        st.sidebar.success("Loaded saved embeddings")

    # ── Option 2: CSV present ────────────────────────────────────────────────
    elif os.path.exists("data1300.csv"):
        df = pd.read_csv("data1300.csv")
        l_df = df.copy()
        l_df["doc_length"] = l_df["Doc"].astype(str).apply(lambda x: len(x.split()))

        bins       = [0, 5, 10, 15, 20, 30, np.inf]
        bin_labels = ["0-5", "6-10", "11-15", "16-20", "21-30", "30+"]
        l_df["length_bin"] = pd.cut(l_df["doc_length"], bins=bins, labels=bin_labels)

        facts     = l_df[l_df["Label"] == "Fact"]
        non_facts = l_df[l_df["Label"] == "Non-Fact"]

        resampled = []
        for b in bin_labels:
            fb = facts[facts["length_bin"] == b]
            nfb = non_facts[non_facts["length_bin"] == b]
            if len(fb) == 0 or len(nfb) == 0:
                continue
            resampled.append(fb.sample(n=min(len(fb), len(nfb)), random_state=42))

        l_df = pd.concat(resampled + [non_facts]).sample(frac=1, random_state=42).reset_index(drop=True)
        l_df = l_df.drop(columns=["doc_length", "length_bin"])

        train_df, _ = train_test_split(l_df, test_size=0.2, random_state=42)
        y_train = train_df["Label"].map({"Fact": 1, "Non-Fact": 0}).values

        with st.spinner("Embedding training data with T5 (one-time)…"):
            x_train = attention_mean_pool(_encoder, _tokenizer, train_df["Doc"].tolist())

        np.save("x_train_t5.npy", x_train)
        np.save("y_train.npy", y_train)
        st.sidebar.success("Embedded and cached from CSV")

    # ── Option 3: demo fallback ──────────────────────────────────────────────
    else:
        demo_texts = [
            "The Eiffel Tower is located in Paris, France.",
            "Water boils at 100 degrees Celsius at sea level.",
            "The human brain has approximately 86 billion neurons.",
            "Mount Everest is the tallest mountain on Earth.",
            "Shakespeare wrote Hamlet around 1600.",
            "Dragons are real and live in the mountains.",
            "The moon is made of cheese.",
            "Coffee was invented in ancient Egypt.",
            "I think summer is the best season.",
            "Everyone secretly loves Mondays.",
        ]
        demo_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        x_train = attention_mean_pool(_encoder, _tokenizer, demo_texts)
        y_train  = np.array(demo_labels)
        st.sidebar.warning("Demo mode: upload data1300.csv to use the trained model")

    # ── Fit SVM ──────────────────────────────────────────────────────────────
    search = RandomizedSearchCV(
        SVC(class_weight="balanced", random_state=42, probability=True),
        {"C": loguniform(1e-2, 1e2), "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
        n_iter=5, cv=min(3, len(np.unique(y_train))),
        scoring="f1_macro", n_jobs=1, random_state=42,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='section-label'>Model info</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-card">
            <strong>Embedding</strong><br>
            google/t5-base · attention mean pooling · dimension 768<br><br>
            <strong>Classifier</strong><br>
            SVM with RBF or linear kernel<br>
            class_weight='balanced' · tuned with RandomizedSearchCV<br><br>
            <strong>Label mapping</strong><br>
            Fact = 1 · Non-Fact = 0
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Load models ─────────────────────────────────────────────────────────────
tokenizer, encoder = load_t5()
svm, best_params   = load_svm(tokenizer, encoder)

with st.sidebar:
    st.divider()
    st.markdown("### Best SVM params")
    for k, v in best_params.items():
        st.markdown(f"`{k}` = **{v}**")


# ─── Main UI ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Fact vs. Non-Fact Classification</div>
        <div class="hero-subtitle">SVM + T5 with attention mean pooling for binary sentence-level classification.</div>
        <div class="section-label">Model overview</div>
        <div class="mono">Pretrained encoder features are cached when available, then passed to a tuned SVM classifier.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Initialise session state ──────────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "history" not in st.session_state:
    st.session_state.history = []

# Examples
EXAMPLES = [
    "The Eiffel Tower is located in Paris, France.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The human brain has about 86 billion neurons.",
    "Mount Everest is the tallest mountain on Earth.",
    "Dragons are real and live in the mountains.",
    "The moon is made of cheese.",
    "I think pineapple belongs on pizza.",
    "Coffee was invented in ancient Egypt.",
]

def set_example(text):
    """Callback: write chosen example into session state before rerun."""
    st.session_state.input_text = text

st.markdown("<div class='content-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>Sample inputs</div>", unsafe_allow_html=True)
cols = st.columns(4)
for i, ex in enumerate(EXAMPLES):
    cols[i % 4].button(
        f"Example {i + 1}",
        key=f"ex_{i}",
        on_click=set_example,
        args=(ex,),
        use_container_width=True,
    )
    cols[i % 4].markdown(f"<div class='example-note'>{ex}</div>", unsafe_allow_html=True)

# Input — bound to session_state key so it survives reruns
st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
sentence = st.text_area(
    "Enter a sentence:",
    key="input_text",
    height=120,
    placeholder="Type any sentence here...",
    label_visibility="visible",
)

classify_btn = st.button("Classify", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ─── Predict ─────────────────────────────────────────────────────────────────
if classify_btn and sentence.strip():
    with st.spinner("Encoding with T5 and running SVM..."):
        emb   = attention_mean_pool(encoder, tokenizer, [sentence.strip()])
        proba = svm.predict_proba(emb)[0]       # [P(Non-Fact), P(Fact)]
        pred  = svm.predict(emb)[0]             # 0 or 1

    label      = "Fact" if pred == 1 else "Non-Fact"
    fact_prob  = float(proba[1])
    conf       = float(max(proba))
    is_fact    = pred == 1

    # Verdict
    css_class = "verdict-fact" if is_fact else "verdict-nonfact"
    st.markdown(f'<div class="{css_class}">{label}</div>', unsafe_allow_html=True)
    st.markdown("")

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", label)
    c2.metric("Confidence", f"{conf*100:.1f}%")
    c3.metric("P(Fact)", f"{fact_prob:.3f}")

    # Probability bar
    st.markdown("**Fact probability**")
    st.progress(fact_prob)
    st.markdown(
        f'<p class="mono">Non-Fact ◀──── {fact_prob:.3f} ────▶ Fact</p>',
        unsafe_allow_html=True
    )

    # SVM decision info
    with st.expander("SVM decision details"):
        decision = svm.decision_function(emb)[0]
        st.markdown(f"**Decision function value:** `{decision:.4f}`")
        st.markdown(f"**Kernel:** `{best_params.get('kernel', 'rbf')}`")
        st.markdown(f"**C:** `{best_params.get('C', '—')}`")
        st.markdown(f"**Gamma:** `{best_params.get('gamma', 'scale')}`")
        st.markdown(f"**Embedding shape:** `{emb.shape}`  (T5 hidden dim)")
        raw_col1, raw_col2 = st.columns(2)
        raw_col1.metric("P(Non-Fact)", f"{proba[0]:.4f}")
        raw_col2.metric("P(Fact)",     f"{proba[1]:.4f}")

    # History
    st.session_state.history.insert(0, {
        "Sentence": sentence.strip()[:60] + ("…" if len(sentence) > 60 else ""),
        "Label": label,
        "Confidence": f"{conf*100:.1f}%",
        "P(Fact)": f"{fact_prob:.3f}",
    })
    st.session_state.history = st.session_state.history[:10]

elif classify_btn:
    st.warning("Please enter a sentence first.")

# ─── History ─────────────────────────────────────────────────────────────────
if st.session_state.get("history"):
    st.divider()
    st.markdown("#### Recent predictions")
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)