import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import base64
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PANOPTICON · Avian Diagnostics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Coccidiosis', 'Healthy', 'NewCastle', 'Salmonella']

CLASS_META = {
    'Coccidiosis': {
        'code':     'COC-7',
        'risk':     'HIGH',
        'risk_val': 75,
        'color':    '#F97316',
        'glow':     'rgba(249,115,22,0.35)',
        'border':   '#7C3000',
        'icon':     '⚠',
        'icd':      'B67.0',
        'action':   'Administer anticoccidial agents immediately. Isolate affected flock segment. Increase litter monitoring frequency to every 4 hours.',
        'transmission': 'Fecal-oral / Oocysts',
        'incubation':   '4 – 7 days',
        'mortality':    'Up to 30% if untreated',
    },
    'Healthy': {
        'code':     'CLR-0',
        'risk':     'NONE',
        'risk_val': 0,
        'color':    '#22D3A5',
        'glow':     'rgba(34,211,165,0.30)',
        'border':   '#064E3B',
        'icon':     '✓',
        'icd':      'Z00.0',
        'action':   'No intervention required. Continue standard biosecurity protocol. Schedule next diagnostic scan in 72 hours.',
        'transmission': 'N/A',
        'incubation':   'N/A',
        'mortality':    'Negligible',
    },
    'NewCastle': {
        'code':     'NDV-1',
        'risk':     'CRITICAL',
        'risk_val': 98,
        'color':    '#EF4444',
        'glow':     'rgba(239,68,68,0.40)',
        'border':   '#7F1D1D',
        'icon':     '✕',
        'icd':      'B34.8',
        'action':   'IMMEDIATE veterinary notification required. Initiate flock quarantine. Report to PDMA within 2 hours per regulatory mandate. Do not move birds.',
        'transmission': 'Airborne / Direct contact',
        'incubation':   '2 – 15 days',
        'mortality':    'Up to 100% in naive flocks',
    },
    'Salmonella': {
        'code':     'SAL-3',
        'risk':     'HIGH',
        'risk_val': 80,
        'color':    '#A855F7',
        'glow':     'rgba(168,85,247,0.35)',
        'border':   '#4C1D95',
        'icon':     '⚠',
        'icd':      'A02.0',
        'action':   'Contact veterinarian for antibiotic susceptibility testing. Review biosecurity and water sanitation. Mandatory food-chain notification if birds are near processing age.',
        'transmission': 'Fecal-oral / Contaminated feed',
        'incubation':   '6 – 72 hours',
        'mortality':    '5 – 20% in young chicks',
    }
}

# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500&display=swap');
    :root {
        --bg-void:        #050608;
        --bg-panel:       #0A0C10;
        --bg-card:        #0E1117;
        --bg-elevated:    #141820;
        --border-subtle:  rgba(255,255,255,0.06);
        --border-mid:     rgba(255,255,255,0.10);
        --border-strong:  rgba(255,255,255,0.18);
        --accent-cyan:    #00E5FF;
        --accent-cyan-dim:#007A8C;
        --text-primary:   #F0F4FF;
        --text-secondary: #8892A4;
        --text-tertiary:  #4A5568;
        --font-display:   'Syne', sans-serif;
        --font-mono:      'JetBrains Mono', monospace;
        --font-body:      'Inter', sans-serif;
        --radius-sm:      6px;
        --radius-md:      12px;
        --radius-lg:      20px;
    }
    /* ── Reset & base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg-void) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
    }
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu, footer { display: none !important; }
    [data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    /* ── Scanline overlay ── */
    body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,229,255,0.012) 2px,
        rgba(0,229,255,0.012) 4px
    );
    pointer-events: none;
    z-index: 9999;
}
@media (max-width: 768px) {
    body::before { display: none; }
}
    /* ── Top header bar ── */
    .pnp-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 48px;
        border-bottom: 1px solid var(--border-subtle);
        background: rgba(5,6,8,0.95);
        backdrop-filter: blur(20px);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    .pnp-logo {
        font-family: var(--font-display);
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 0.3em;
        color: var(--accent-cyan);
        text-transform: uppercase;
    }
    .pnp-logo span {
        color: var(--text-tertiary);
        font-weight: 400;
    }
    .pnp-status-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 100px;
        border: 1px solid rgba(0,229,255,0.2);
        background: rgba(0,229,255,0.05);
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--accent-cyan);
        letter-spacing: 0.08em;
    }
    .pnp-status-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--accent-cyan);
        box-shadow: 0 0 8px var(--accent-cyan);
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%,100% { opacity:1; transform:scale(1); }
        50%      { opacity:0.5; transform:scale(0.75); }
    }
    .pnp-build {
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--text-tertiary);
        letter-spacing: 0.05em;
    }
    /* ── Hero section ── */
    .pnp-hero {
        padding: 64px 48px 48px;
        position: relative;
        overflow: hidden;
    }
    .pnp-hero::before {
        content: '';
        position: absolute;
        top: -120px; right: -80px;
        width: 600px; height: 600px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0,229,255,0.04) 0%, transparent 70%);
        pointer-events: none;
    }
    .pnp-hero-eyebrow {
        font-family: var(--font-mono);
        font-size: 11px;
        letter-spacing: 0.25em;
        color: var(--accent-cyan);
        text-transform: uppercase;
        margin-bottom: 16px;
    }
    .pnp-hero-title {
        font-family: var(--font-display);
        font-size: clamp(36px, 5vw, 64px);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        margin-bottom: 16px;
    }
    .pnp-hero-title em {
        font-style: normal;
        color: var(--accent-cyan);
    }
    .pnp-hero-sub {
        font-size: 15px;
        color: var(--text-secondary);
        line-height: 1.7;
        max-width: 520px;
        font-weight: 300;
    }
    /* ── Stat strip ── */
    .pnp-stats {
        display: flex;
        gap: 1px;
        margin: 40px 48px 0;
        background: var(--border-subtle);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        overflow: hidden;
    }
    .pnp-stat {
        flex: 1;
        padding: 20px 28px;
        background: var(--bg-panel);
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .pnp-stat-val {
        font-family: var(--font-display);
        font-size: 26px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    .pnp-stat-label {
        font-family: var(--font-mono);
        font-size: 10px;
        letter-spacing: 0.15em;
        color: var(--text-tertiary);
        text-transform: uppercase;
    }
    /* ── Main layout grid ── */
    .pnp-main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    padding: 40px 48px;
    align-items: start;
}
@media (max-width: 768px) {
    .pnp-main {
        grid-template-columns: 1fr;
        padding: 20px 16px;
        gap: 16px;
    }
    .pnp-topbar {
        padding: 14px 16px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .pnp-build { display: none; }
    .pnp-hero {
        padding: 32px 16px 24px;
    }
    .pnp-hero-title {
        font-size: 32px;
    }
    .pnp-stats {
        margin: 24px 16px 0;
        flex-wrap: wrap;
    }
    .pnp-stat {
        min-width: 80px;
        padding: 14px 16px;
    }
    .pnp-stat-val { font-size: 20px; }
    .pnp-footer {
        flex-direction: column;
        gap: 12px;
        padding: 20px 16px;
        text-align: center;
    }
    .pnp-footer-disclaimer { text-align: center; }
    .pnp-verdict-name { font-size: 28px; }
    .pnp-panel-body { padding: 16px; }
    .pnp-verdict { padding: 20px 16px; }
    .pnp-action { padding: 16px; }
    .pnp-breakdown { padding: 16px; }
    .pnp-meta-cell { padding: 12px 14px; }
}
    /* ── Upload panel ── */
    .pnp-upload-panel {
        background: var(--bg-panel);
        border: 1px solid var(--border-mid);
        border-radius: var(--radius-lg);
        overflow: hidden;
    }
    .pnp-panel-header {
        padding: 20px 28px;
        border-bottom: 1px solid var(--border-subtle);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .pnp-panel-title {
        font-family: var(--font-display);
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text-secondary);
    }
    .pnp-panel-badge {
        font-family: var(--font-mono);
        font-size: 10px;
        padding: 3px 10px;
        border-radius: 100px;
        border: 1px solid var(--border-mid);
        color: var(--text-tertiary);
        letter-spacing: 0.08em;
    }
    .pnp-panel-body { padding: 28px; }
    /* ── Streamlit file uploader reskin ── */
    [data-testid="stFileUploader"] {
        width: 100% !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background: var(--bg-card) !important;
        border: 1.5px dashed var(--border-mid) !important;
        border-radius: var(--radius-md) !important;
        transition: border-color 0.2s, background 0.2s !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--accent-cyan) !important;
        background: rgba(0,229,255,0.03) !important;
    }
    [data-testid="stFileUploadDropzone"] p,
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] small {
        font-family: var(--font-mono) !important;
        color: var(--text-tertiary) !important;
        font-size: 12px !important;
    }
    [data-testid="stBaseButton-secondary"] {
        background: transparent !important;
        border: 1px solid var(--border-mid) !important;
        color: var(--accent-cyan) !important;
        font-family: var(--font-mono) !important;
        font-size: 12px !important;
        border-radius: var(--radius-sm) !important;
        transition: all 0.2s !important;
    }
    [data-testid="stBaseButton-secondary"]:hover {
        border-color: var(--accent-cyan) !important;
        background: rgba(0,229,255,0.06) !important;
    }
    /* ── Image display card ── */
    .pnp-img-card {
        position: relative;
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid var(--border-mid);
        background: var(--bg-card);
        margin-top: 20px;
    }
    .pnp-img-card img {
        width: 100%;
        display: block;
        border-radius: var(--radius-md);
    }
    .pnp-img-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(to top, rgba(5,6,8,0.7) 0%, transparent 50%);
        border-radius: var(--radius-md);
        pointer-events: none;
    }
    .pnp-img-meta {
        position: absolute;
        bottom: 14px; left: 16px;
        font-family: var(--font-mono);
        font-size: 10px;
        color: rgba(255,255,255,0.5);
        letter-spacing: 0.08em;
    }
    .pnp-img-corner {
        position: absolute;
        top: 12px; right: 12px;
        width: 28px; height: 28px;
        border-top: 2px solid var(--accent-cyan);
        border-right: 2px solid var(--accent-cyan);
        border-radius: 0 4px 0 0;
        opacity: 0.6;
    }
    /* ── Scanning animation ── */
    .pnp-scan-wrapper {
        position: relative;
        margin-top: 20px;
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid rgba(0,229,255,0.2);
    }
    .pnp-scan-bar {
        position: absolute;
        left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
        box-shadow: 0 0 12px var(--accent-cyan);
        animation: scan-sweep 1.8s ease-in-out infinite;
        z-index: 10;
    }
    @keyframes scan-sweep {
        0%   { top: 0%; opacity: 0; }
        10%  { opacity: 1; }
        90%  { opacity: 1; }
        100% { top: 100%; opacity: 0; }
    }
    .pnp-scan-grid {
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(0,229,255,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,229,255,0.04) 1px, transparent 1px);
        background-size: 24px 24px;
        pointer-events: none;
    }
    .pnp-scan-label {
        position: absolute;
        top: 12px; left: 16px;
        font-family: var(--font-mono);
        font-size: 10px;
        letter-spacing: 0.15em;
        color: var(--accent-cyan);
        text-transform: uppercase;
        z-index: 11;
        animation: blink-text 1s step-end infinite;
    }
    @keyframes blink-text {
        0%,100% { opacity: 1; }
        50%      { opacity: 0.3; }
    }
    /* ── Results panel ── */
    .pnp-result-panel {
        background: var(--bg-panel);
        border: 1px solid var(--border-mid);
        border-radius: var(--radius-lg);
        overflow: hidden;
        animation: panel-appear 0.5s ease-out;
    }
    @keyframes panel-appear {
        from { opacity:0; transform:translateY(16px); }
        to   { opacity:1; transform:translateY(0); }
    }
    /* ── Verdict block ── */
    .pnp-verdict {
        padding: 32px 28px;
        position: relative;
        overflow: hidden;
    }
    .pnp-verdict-glow {
        position: absolute;
        top: -60px; right: -60px;
        width: 280px; height: 280px;
        border-radius: 50%;
        pointer-events: none;
        opacity: 0.6;
    }
    .pnp-verdict-eyebrow {
        font-family: var(--font-mono);
        font-size: 10px;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        margin-bottom: 12px;
        opacity: 0.7;
    }
    .pnp-verdict-name {
        font-family: var(--font-display);
        font-size: clamp(28px, 4vw, 44px);
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1;
        margin-bottom: 8px;
    }
    .pnp-verdict-conf {
        font-family: var(--font-mono);
        font-size: 13px;
        opacity: 0.6;
        margin-bottom: 20px;
    }
    .pnp-risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 16px;
        border-radius: 100px;
        border-width: 1px;
        border-style: solid;
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    /* ── Confidence bar ── */
    .pnp-conf-track {
        height: 3px;
        background: var(--border-subtle);
        border-radius: 100px;
        margin-top: 20px;
        overflow: visible;
        position: relative;
    }
    .pnp-conf-fill {
        height: 100%;
        border-radius: 100px;
        position: relative;
        transition: width 1.2s cubic-bezier(0.16,1,0.3,1);
    }
    .pnp-conf-fill::after {
        content: '';
        position: absolute;
        right: -1px; top: 50%;
        transform: translateY(-50%);
        width: 8px; height: 8px;
        border-radius: 50%;
        background: inherit;
        box-shadow: 0 0 10px currentColor;
    }
    /* ── Metadata grid ── */
    .pnp-meta-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1px;
        background: var(--border-subtle);
        border-top: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
    }
    .pnp-meta-cell {
        background: var(--bg-panel);
        padding: 16px 20px;
    }
    .pnp-meta-key {
        font-family: var(--font-mono);
        font-size: 9px;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--text-tertiary);
        margin-bottom: 5px;
    }
    .pnp-meta-val {
        font-family: var(--font-mono);
        font-size: 12px;
        color: var(--text-primary);
        font-weight: 500;
    }
    /* ── Action block ── */
    .pnp-action {
        padding: 24px 28px;
        border-top: 1px solid var(--border-subtle);
    }
    .pnp-action-label {
        font-family: var(--font-mono);
        font-size: 9px;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: var(--text-tertiary);
        margin-bottom: 10px;
    }
    .pnp-action-text {
        font-size: 13px;
        line-height: 1.7;
        color: var(--text-secondary);
        font-weight: 300;
    }
    /* ── Probability breakdown ── */
    .pnp-breakdown {
        padding: 24px 28px;
        border-top: 1px solid var(--border-subtle);
    }
    .pnp-breakdown-title {
        font-family: var(--font-mono);
        font-size: 9px;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: var(--text-tertiary);
        margin-bottom: 18px;
    }
    .pnp-prob-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
    }
    .pnp-prob-label {
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--text-secondary);
        width: 110px;
        flex-shrink: 0;
    }
    .pnp-prob-bar-bg {
        flex: 1;
        height: 4px;
        background: var(--bg-elevated);
        border-radius: 100px;
        overflow: hidden;
    }
    .pnp-prob-bar-fill {
        height: 100%;
        border-radius: 100px;
    }
    .pnp-prob-pct {
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--text-tertiary);
        width: 48px;
        text-align: right;
        flex-shrink: 0;
    }
    /* ── Idle state panel ── */
    .pnp-idle {
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 56px 28px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-height: 420px;
    }
    .pnp-idle-icon {
        font-size: 40px;
        margin-bottom: 20px;
        opacity: 0.25;
    }
    .pnp-idle-title {
        font-family: var(--font-display);
        font-size: 18px;
        font-weight: 700;
        color: var(--text-secondary);
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .pnp-idle-sub {
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--text-tertiary);
        letter-spacing: 0.1em;
        line-height: 1.8;
    }
    .pnp-idle-classes {
        display: flex;
        gap: 8px;
        margin-top: 28px;
        flex-wrap: wrap;
        justify-content: center;
    }
    .pnp-idle-class {
        font-family: var(--font-mono);
        font-size: 10px;
        padding: 5px 12px;
        border-radius: 100px;
        border: 1px solid var(--border-mid);
        color: var(--text-tertiary);
        letter-spacing: 0.08em;
    }
    /* ── Footer ── */
    .pnp-footer {
        padding: 24px 48px;
        border-top: 1px solid var(--border-subtle);
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 8px;
    }
    .pnp-footer-text {
        font-family: var(--font-mono);
        font-size: 10px;
        color: var(--text-tertiary);
        letter-spacing: 0.08em;
        line-height: 1.8;
    }
    .pnp-footer-disclaimer {
        font-family: var(--font-mono);
        font-size: 9px;
        color: var(--text-tertiary);
        opacity: 0.5;
        letter-spacing: 0.06em;
        text-align: right;
        max-width: 400px;
        line-height: 1.7;
    }
    /* ── Streamlit image override ── */
    [data-testid="stImage"] img {
        border-radius: var(--radius-md) !important;
    }
    /* ── Spinner override ── */
    [data-testid="stSpinner"] {
        color: var(--accent-cyan) !important;
    }
      @media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
}
    </style>
    """, unsafe_allow_html=True)

 


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model('model.keras')


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Image → base64 ────────────────────────────────────────────────────────────
def img_to_b64(image: Image.Image, max_size: int = 600) -> str:
    img = image.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=88)
    return base64.b64encode(buf.getvalue()).decode()


# ── Render probability rows ───────────────────────────────────────────────────
def render_prob_rows(probs: dict) -> str:
    rows = ""
    for cls, pct in sorted(probs.items(), key=lambda x: -x[1]):
        color  = CLASS_META[cls]['color']
        rows += f"""
        <div class="pnp-prob-row">
            <span class="pnp-prob-label">{cls}</span>
            <div class="pnp-prob-bar-bg">
                <div class="pnp-prob-bar-fill"
                     style="width:{pct:.1f}%; background:{color}; opacity:0.8;"></div>
            </div>
            <span class="pnp-prob-pct">{pct:.1f}%</span>
        </div>"""
    return rows


# ── App ───────────────────────────────────────────────────────────────────────
inject_css()
model = load_model()

# Top bar
st.markdown("""
<div class="pnp-topbar">
    <div class="pnp-logo">PANOPTICON<span> · AVIAN DIAGNOSTICS</span></div>
    <div class="pnp-status-pill">
        <div class="pnp-status-dot"></div>
        SYSTEM ONLINE
    </div>
    <div class="pnp-build">BUILD 2.4.1 · MobileNetV2 · 96.42% ACC</div>
</div>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="pnp-hero">
    <div class="pnp-hero-eyebrow">// AI-POWERED FLOCK HEALTH SURVEILLANCE</div>
    <div class="pnp-hero-title">Rapid Avian<br><em>Disease Diagnosis</em></div>
    <div class="pnp-hero-sub">
        Upload a fecal sample image from your commercial broiler shed.
        Panopticon's deep learning engine delivers a clinical-grade
        pathogen classification in under two seconds.
    </div>
</div>
<div class="pnp-stats">
    <div class="pnp-stat">
        <div class="pnp-stat-val">96.42%</div>
        <div class="pnp-stat-label">Test Accuracy</div>
    </div>
    <div class="pnp-stat">
        <div class="pnp-stat-val">4</div>
        <div class="pnp-stat-label">Pathogen Classes</div>
    </div>
    <div class="pnp-stat">
        <div class="pnp-stat-val">&lt; 2s</div>
        <div class="pnp-stat-label">Inference Time</div>
    </div>
    <div class="pnp-stat">
        <div class="pnp-stat-val">224²</div>
        <div class="pnp-stat-label">Input Resolution</div>
    </div>
    <div class="pnp-stat">
        <div class="pnp-stat-val">MNv2</div>
        <div class="pnp-stat-label">Architecture</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main two-column layout
col_left, col_right = st.columns([1, 1], gap="small")

st.markdown('<div class="pnp-main-wrap">', unsafe_allow_html=True)
col_left, col_right = st.columns([1, 1], gap="small")
# ... all your existing col_left and col_right code ...
st.markdown('</div>', unsafe_allow_html=True)

with col_left:
    st.markdown("""
    <div class="pnp-upload-panel">
        <div class="pnp-panel-header">
            <span class="pnp-panel-title">Sample Input</span>
            <span class="pnp-panel-badge">JPEG · PNG · SUPPORTED</span>
        </div>
        <div class="pnp-panel-body">
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Drop fecal sample image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        image   = Image.open(uploaded)
        b64     = img_to_b64(image)
        w, h    = image.size
        fmt     = image.format or "JPEG"
        kb      = round(uploaded.size / 1024, 1)

        st.markdown(f"""
        <div class="pnp-img-card">
            <img src="data:image/jpeg;base64,{b64}" />
            <div class="pnp-img-overlay"></div>
            <div class="pnp-img-corner"></div>
            <div class="pnp-img-meta">{w}×{h}px · {fmt} · {kb} KB</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

with col_right:
    if not uploaded:
        st.markdown("""
        <div class="pnp-idle">
            <div class="pnp-idle-icon">🔬</div>
            <div class="pnp-idle-title">Awaiting Sample</div>
            <div class="pnp-idle-sub">
                Upload an image to the left panel<br>
                to initiate diagnostic analysis
            </div>
            <div class="pnp-idle-classes">
                <span class="pnp-idle-class">Coccidiosis</span>
                <span class="pnp-idle-class">Healthy</span>
                <span class="pnp-idle-class">NewCastle</span>
                <span class="pnp-idle-class">Salmonella</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        scan_slot = st.empty()
        scan_slot.markdown(f"""
        <div class="pnp-scan-wrapper">
            <div class="pnp-scan-bar"></div>
            <div class="pnp-scan-grid"></div>
            <div class="pnp-scan-label">SCANNING ···</div>
            <img src="data:image/jpeg;base64,{b64}"
                 style="width:100%;display:block;opacity:0.55;filter:saturate(0.3) brightness(0.6);" />
        </div>
        """, unsafe_allow_html=True)

        t0    = time.time()
        arr   = preprocess(image)
        preds = model.predict(arr, verbose=0)
        elapsed = time.time() - t0

        scan_slot.empty()

        pred_idx   = int(np.argmax(preds))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(np.max(preds)) * 100
        meta       = CLASS_META[pred_class]
        probs      = {CLASS_NAMES[i]: float(preds[0][i]) * 100 for i in range(4)}

        # ── Verdict ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="pnp-result-panel">
        <div class="pnp-verdict">
            <div class="pnp-verdict-glow"
                 style="background:radial-gradient(circle,{meta['glow']} 0%,transparent 70%);"></div>
            <div class="pnp-verdict-eyebrow" style="color:{meta['color']};">
                ◆ DIAGNOSIS COMPLETE · {elapsed*1000:.0f}ms
            </div>
            <div class="pnp-verdict-name" style="color:{meta['color']};">{pred_class}</div>
            <div class="pnp-verdict-conf">
                Confidence: {confidence:.2f}% · Code: {meta['code']} · ICD: {meta['icd']}
            </div>
            <div class="pnp-risk-badge"
                 style="color:{meta['color']};border-color:{meta['color']}40;background:{meta['glow']};">
                <span>{meta['icon']}</span>
                <span>RISK LEVEL: {meta['risk']}</span>
            </div>
            <div class="pnp-conf-track">
                <div class="pnp-conf-fill"
                     style="width:{confidence:.1f}%;
                            background:linear-gradient(90deg,{meta['color']}88,{meta['color']});
                            box-shadow:0 0 8px {meta['glow']};"></div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metadata ─────────────────────────────────────────────
        st.markdown(f"""
        <div class="pnp-meta-grid" style="border-radius:0;margin-top:1px;">
            <div class="pnp-meta-cell">
                <div class="pnp-meta-key">Transmission</div>
                <div class="pnp-meta-val">{meta['transmission']}</div>
            </div>
            <div class="pnp-meta-cell">
                <div class="pnp-meta-key">Incubation Period</div>
                <div class="pnp-meta-val">{meta['incubation']}</div>
            </div>
            <div class="pnp-meta-cell">
                <div class="pnp-meta-key">Mortality Risk</div>
                <div class="pnp-meta-val">{meta['mortality']}</div>
            </div>
            <div class="pnp-meta-cell">
                <div class="pnp-meta-key">Inference Time</div>
                <div class="pnp-meta-val">{elapsed*1000:.1f} ms</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Action ───────────────────────────────────────────────
        st.markdown(f"""
        <div class="pnp-action" style="background:var(--bg-panel);
             border:1px solid var(--border-subtle);
             border-top:none;padding:24px 28px;margin-top:1px;">
            <div class="pnp-action-label">Recommended Action</div>
            <div class="pnp-action-text">{meta['action']}</div>
        </div>
        """, unsafe_allow_html=True)

        # # ── Probability breakdown ─────────────────────────────────
        st.markdown("""
        <div class="pnp-breakdown" style="border-radius:0 0 var(--radius-lg) var(--radius-lg);
             border-top:none;margin-top:1px;">
            <div class="pnp-breakdown-title">Class Probability Distribution</div>
        </div>
        """, unsafe_allow_html=True)

        for cls, pct in sorted(probs.items(), key=lambda x: -x[1]):
            color = CLASS_META[cls]['color']
            st.markdown(
                f'<div class="pnp-prob-row">'
                f'<span class="pnp-prob-label">{cls}</span>'
                f'<div class="pnp-prob-bar-bg">'
                f'<div class="pnp-prob-bar-fill" style="width:{pct:.1f}%;background:{color};opacity:0.8;"></div>'
                f'</div>'
                f'<span class="pnp-prob-pct">{pct:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True
            )
# Footer
st.markdown("""
<div class="pnp-footer">
    <div class="pnp-footer-text">
        PANOPTICON · Avian Diagnostics Platform<br>
        Pakistan Commercial Broiler Industry · MobileNetV2 · TensorFlow
    </div>
    <div class="pnp-footer-disclaimer">
        FOR RESEARCH AND ACADEMIC USE ONLY.<br>
        NOT A SUBSTITUTE FOR LICENSED VETERINARY DIAGNOSIS.
    </div>
</div>
""", unsafe_allow_html=True)
