"""
app.py
------
Streamlit frontend for the Fruit Quality Analyzer.

Run:
    streamlit run app.py

Features:
  • Upload an image → instant quality + shelf-life prediction
  • Confidence breakdown chart
  • Prediction history table
  • Aggregate stats dashboard
"""

import sys
import io
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Fruit Quality Analyzer",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — clean light mode with botanical accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #FAFAF7;
    color: #1C1C1E;
}

/* ── Headings ── */
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

/* ── Header banner ── */
.main-header {
    background: linear-gradient(135deg, #2D6A4F 0%, #40916C 50%, #52B788 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(45, 106, 79, 0.20);
}
.main-header h1 {
    color: #fff;
    font-size: 2.4rem;
    margin: 0 0 4px;
    letter-spacing: -0.5px;
}
.main-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin: 0;
    font-weight: 300;
}

/* ── Upload box ── */
section[data-testid="stFileUploader"] {
    border: 2px dashed #52B788;
    border-radius: 12px;
    background: #F0FFF4;
    padding: 8px;
}

/* ── Result cards ── */
.result-card {
    background: #fff;
    border-radius: 14px;
    padding: 22px 26px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    border-left: 5px solid #52B788;
    margin-bottom: 16px;
}
.result-card.danger { border-left-color: #E63946; }
.result-card.warning { border-left-color: #F4A261; }
.result-card.ok { border-left-color: #2DC653; }

/* ── Metric tiles ── */
.metric-tile {
    background: #fff;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-top: 3px solid #52B788;
}
.metric-tile .val { font-size: 2rem; font-weight: 600; color: #2D6A4F; }
.metric-tile .lbl { font-size: 0.82rem; color: #6B7280; margin-top: 4px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #F4FBF4; }

/* ── Buttons ── */
.stButton > button {
    background: #2D6A4F;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 8px 22px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    transition: background 0.2s;
}
.stButton > button:hover { background: #1B4332; }

/* ── Progress bars ── */
.stProgress > div > div { background-color: #52B788 !important; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid #E5E7EB; margin: 20px 0; }

/* ── Table ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI model…")
def get_model_and_classes():
    """Load model once and cache in session."""
    from predict import load_model_and_meta
    return load_model_and_meta()


def run_prediction(img_rgb: np.ndarray, filename: str = "upload.jpg") -> dict:
    from predict import predict_from_array
    return predict_from_array(img_rgb, filename=filename)


def status_color(status: str) -> str:
    return {
        "Excellent": "#2DC653",
        "Good":      "#52B788",
        "Fair":      "#F4A261",
        "Poor":      "#E76F51",
        "Very Poor": "#E63946",
        "Expired":   "#9B2226",
    }.get(status, "#6B7280")


def make_confidence_chart(result: dict) -> plt.Figure:
    """Horizontal bar chart for all class probabilities."""
    probs      = result["all_probs"]
    classes    = list(probs.keys())
    values     = list(probs.values())
    top_class  = result["predicted_class"]

    colors = ["#2D6A4F" if c == top_class else "#D1E7DD" for c in classes]

    fig, ax = plt.subplots(figsize=(6, max(3, len(classes) * 0.55)))
    bars = ax.barh(classes, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.bar_label(bars, labels=[f"{v*100:.1f}%" for v in values], padding=4, fontsize=9)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Confidence", fontsize=9)
    ax.set_title("Class Probabilities", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("#FAFAF7")
    ax.set_facecolor("#FAFAF7")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🍃 About")
    st.markdown(
        "This app uses a **MobileNetV2** deep learning model trained on "
        "fruit & vegetable quality images to predict freshness and estimate shelf life."
    )
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Upload a fruit or vegetable image\n2. View predicted quality level\n3. Check estimated shelf life")
    st.markdown("---")

    # Model info
    meta_path = Path(__file__).parent / "model" / "model_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        st.markdown("**Model details:**")
        st.caption(f"🧠 Backbone: `{meta.get('backbone', 'N/A')}`")
        st.caption(f"📦 Classes: `{meta.get('num_classes', '?')}`")
        trained_at = meta.get("trained_at", "")
        if trained_at:
            st.caption(f"🗓 Trained: `{trained_at[:10]}`")
    else:
        st.warning("⚠️ Model not trained yet.\nRun `python train.py` first.")

    st.markdown("---")
    # History controls
    st.markdown("**Database**")
    if st.button("🗑️ Clear history"):
        from database import clear_history
        clear_history()
        st.success("History cleared!")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>🍎 Fruit Quality Analyzer</h1>
  <p>AI-powered quality classification & shelf-life estimation using Deep Learning</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_predict, tab_history, tab_stats = st.tabs(["🔍 Predict", "📋 History", "📊 Statistics"])


# ── Tab 1: Predict ────────────────────────────────────────────────────────────
with tab_predict:
    col_upload, col_result = st.columns([1, 1.4], gap="large")

    with col_upload:
        st.markdown("#### Upload Image")
        uploaded = st.file_uploader(
            "Choose a fruit or vegetable image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            # Decode
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption=uploaded.name, use_column_width=True)

            predict_btn = st.button("🔬 Analyse Quality", use_container_width=True)
        else:
            st.info("👆 Upload an image to get started")
            predict_btn = False

    with col_result:
        st.markdown("#### Analysis Result")

        if uploaded and predict_btn:
            with st.spinner("Analysing with AI model…"):
                try:
                    result = run_prediction(img_rgb, filename=uploaded.name)

                    pred_class = result["predicted_class"]
                    confidence = result["confidence"]
                    shelf      = result["shelf_life"]
                    sc         = status_color(shelf["status"])

                    # ── Quality card ───────────────────────────────────────
                    card_class = (
                        "danger"  if shelf["status"] in ("Expired", "Very Poor") else
                        "warning" if shelf["status"] in ("Poor", "Fair") else
                        "ok"
                    )
                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <div style="font-size:2.2rem">{shelf['emoji']}</div>
                        <div style="font-size:1.6rem; font-weight:700; color:{sc}; font-family:'DM Serif Display',serif;">
                            {pred_class.upper()}
                        </div>
                        <div style="color:#6B7280; font-size:0.9rem; margin-top:4px;">
                            Confidence: <strong>{confidence*100:.1f}%</strong>
                            &nbsp;|&nbsp; Status: <strong>{shelf['status']}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Shelf life panel ──────────────────────────────────
                    st.markdown(f"""
                    <div class="result-card" style="border-left-color:{sc}; margin-top:0;">
                        <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:1px; color:#6B7280;">
                            Estimated Shelf Life
                        </div>
                        <div style="font-size:1.7rem; font-weight:600; color:{sc}; margin:6px 0 2px;">
                            {shelf['days']}
                        </div>
                        <div style="font-size:0.85rem; color:#6B7280;">
                            Based on visual quality assessment
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Top-3 ─────────────────────────────────────────────
                    st.markdown("**Top-3 predictions**")
                    for i, t in enumerate(result["top3"]):
                        conf_pct = t["confidence"] * 100
                        st.markdown(
                            f"{'🥇' if i==0 else '🥈' if i==1 else '🥉'} "
                            f"**{t['class']}** — {conf_pct:.1f}%"
                        )
                        st.progress(t["confidence"])

                    # ── Confidence chart ──────────────────────────────────
                    with st.expander("📊 All class probabilities"):
                        fig = make_confidence_chart(result)
                        st.pyplot(fig, use_container_width=True)

                except FileNotFoundError:
                    st.error(
                        "⚠️ Model not found. Please train the model first:\n\n"
                        "```bash\npython train.py --dataset dataset/\n```"
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        elif not uploaded:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#9CA3AF;">
                <div style="font-size:4rem; margin-bottom:12px;">🍊</div>
                <div style="font-size:1.1rem;">Upload an image to see the analysis</div>
            </div>
            """, unsafe_allow_html=True)


# ── Tab 2: History ────────────────────────────────────────────────────────────
with tab_history:
    from database import fetch_history
    import pandas as pd

    st.markdown("#### Recent Predictions")
    records = fetch_history(limit=100)

    if not records:
        st.info("No predictions yet. Upload an image in the **Predict** tab.")
    else:
        df = pd.DataFrame([{
            "ID":          r["id"],
            "Timestamp":   r["timestamp"],
            "File":        r.get("filename") or "—",
            "Predicted":   r["predicted_class"],
            "Confidence":  f"{r['confidence']*100:.1f}%",
            "Shelf Life":  r.get("shelf_life_days") or "—",
            "Status":      r.get("shelf_status") or "—",
        } for r in records])

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"{len(df)} records shown")


# ── Tab 3: Statistics ─────────────────────────────────────────────────────────
with tab_stats:
    from database import fetch_stats
    import pandas as pd

    stats = fetch_stats()
    st.markdown("#### Prediction Statistics")

    # KPI row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="val">{stats['total']}</div>
            <div class="lbl">Total Predictions</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="val">{stats['avg_confidence']*100:.1f}%</div>
            <div class="lbl">Avg. Confidence</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        n_classes = len(stats["by_class"])
        st.markdown(f"""
        <div class="metric-tile">
            <div class="val">{n_classes}</div>
            <div class="lbl">Classes Seen</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if stats["by_class"]:
        # Pie chart of predictions by class
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = list(stats["by_class"].keys())
        sizes  = list(stats["by_class"].values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )
        ax.set_title("Predictions by Class", fontsize=12, fontweight="bold")
        fig.patch.set_facecolor("#FAFAF7")
        st.pyplot(fig, use_container_width=False)

        # Table
        df_stats = pd.DataFrame([
            {"Class": k, "Count": v, "Share": f"{v/stats['total']*100:.1f}%"}
            for k, v in sorted(stats["by_class"].items(), key=lambda x: -x[1])
        ])
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
    else:
        st.info("No data yet. Make some predictions first.")
