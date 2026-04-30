import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# ── Streamlit'in kendi UI'ını tamamen gizle ──
st.set_page_config(
    page_title="US Accidents Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Streamlit chrome'unu (header, footer, sidebar toggle) tamamen kaldır
st.markdown("""
<style>
/* Streamlit header & toolbar */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="collapsedControl"],
#MainMenu,
footer,
header { display: none !important; }

/* Padding sıfırla */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Dış sarmalayıcı */
.appview-container,
.main,
[data-testid="stAppViewContainer"] {
    padding: 0 !important;
    margin: 0 !important;
    background: #0d0f0e !important;
}

/* iframe tam ekran */
iframe {
    display: block;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ── HTML dosyasını oku ──
html_path = Path(__file__).parent / "dashboard.html"

if html_path.exists():
    html_content = html_path.read_text(encoding="utf-8")
    # Tam ekran — yüksekliği yeterince büyük ver, scrollbar dashboard'un kendi scroll'u
    # app.py dosyanın en sonundaki satırı bununla değiştir:
    components.html(html_content, height=2500, scrolling=True)
else:
    st.error("dashboard.html bulunamadı. app.py ile aynı klasörde olmalı.")
