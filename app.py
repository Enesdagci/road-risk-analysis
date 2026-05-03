"""
Yol Risk Analizi — Streamlit Arayüzü v3
Kurulum:
    pip install streamlit folium streamlit-folium joblib lightgbm requests

Çalıştırma:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# SAYFA AYARLARI
# =============================================================================
st.set_page_config(
    page_title="RiskRadar — Yol Risk Analizi",
    page_icon="🛣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# TASARIM — Koyu, profesyonel, dashboard tarzı
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=Inter:wght@300;400;500&display=swap');

    /* ANA ARKA PLAN */
    .stApp {
        background: #0a0e1a;
        color: #e8eaf0;
    }
    section[data-testid="stSidebar"] {
        background: #0f1420;
        border-right: 1px solid #1e2535;
    }
    section[data-testid="stSidebar"] * {
        color: #c8cdd8 !important;
    }

    /* BAŞLIK */
    .main-header {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        margin: 0 0 0.2rem 0;
        line-height: 1;
    }
    .main-sub {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: #4a90a4;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }

    /* RİSK KUTULARI */
    .risk-card {
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0 1rem 0;
        position: relative;
        overflow: hidden;
        letter-spacing: -0.02em;
    }
    .risk-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at center, rgba(255,255,255,0.04) 0%, transparent 70%);
        pointer-events: none;
    }
    .risk-low  {
        background: linear-gradient(135deg, #0d2e1e 0%, #0a2018 100%);
        color: #4ade80;
        border: 1px solid #1a5c34;
        box-shadow: 0 0 30px rgba(74,222,128,0.08);
    }
    .risk-mid  {
        background: linear-gradient(135deg, #2e1e04 0%, #251a04 100%);
        color: #fbbf24;
        border: 1px solid #5c3d0a;
        box-shadow: 0 0 30px rgba(251,191,36,0.08);
    }
    .risk-high {
        background: linear-gradient(135deg, #2e0a0a 0%, #200808 100%);
        color: #f87171;
        border: 1px solid #5c1414;
        box-shadow: 0 0 30px rgba(248,113,113,0.12);
    }
    .risk-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        font-weight: 400;
        opacity: 0.7;
        margin-top: 0.3rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* METRİK KARTLARI */
    .metric-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 10px;
        margin: 1rem 0;
    }
    .metric-box {
        background: #131929;
        border: 1px solid #1e2840;
        border-radius: 12px;
        padding: 1rem 0.75rem;
        text-align: center;
    }
    .metric-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
    }
    .metric-lbl {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }

    /* OLASIILIK BARLARI */
    .prob-bar-container { margin: 1.2rem 0; }
    .prob-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .prob-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: #8892a4;
        width: 60px;
        flex-shrink: 0;
    }
    .prob-bar-bg {
        flex: 1;
        background: #1a2035;
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    .prob-pct {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: #c8cdd8;
        width: 40px;
        text-align: right;
    }

    /* BÖLÜM BAŞLIKLARI */
    .section-title {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1a2035;
    }

    /* ÖNERİ KUTUSU */
    .oneri-box {
        background: #131929;
        border: 1px solid #1e2840;
        border-left: 3px solid #4a90a4;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #9ba8bc;
        line-height: 1.6;
        margin: 0.75rem 0;
    }

    /* SIDEBAR STİL */
    .sidebar-section {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        color: #3a4a6a !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 1.2rem 0 0.5rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e2535;
    }

    /* FAKTÖR BADGE */
    .factor-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 0.75rem 0;
    }
    .factor-badge {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        padding: 4px 10px;
        border-radius: 20px;
        letter-spacing: 0.04em;
    }
    .factor-danger { background:#2e0a0a; color:#f87171; border:1px solid #5c1414; }
    .factor-warn   { background:#2e1e04; color:#fbbf24; border:1px solid #5c3d0a; }
    .factor-ok     { background:#0d2e1e; color:#4ade80; border:1px solid #1a5c34; }

    /* GENEL STİL */
    .stButton button {
        background: linear-gradient(135deg, #1e3a5f 0%, #162d4a 100%) !important;
        color: #7eb8d4 !important;
        border: 1px solid #2a4f74 !important;
        border-radius: 10px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #254878 0%, #1e3d60 100%) !important;
        border-color: #3a6a94 !important;
        color: #a8d4e8 !important;
    }
    div[data-testid="stMetric"] {
        background: #131929;
        border: 1px solid #1e2840;
        border-radius: 10px;
        padding: 0.75rem;
    }
    .stExpander {
        background: #131929 !important;
        border: 1px solid #1e2840 !important;
        border-radius: 10px !important;
    }

    /* Streamlit elementlerini gizle */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 1.5rem 2rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL YÜKLEME
# =============================================================================
@st.cache_resource
def load_model():
    try:
        model  = joblib.load("trafik_risk_modeli.joblib")
        scaler = joblib.load("trafik_scaler.joblib")
        return model, scaler
    except FileNotFoundError:
        st.error("Model dosyaları bulunamadı! 'trafik_risk_modeli.joblib' ve 'trafik_scaler.joblib' bu klasörde olmalı.")
        st.stop()

model, scaler = load_model()

# =============================================================================
# HAVA VERİSİ
# =============================================================================
def get_weather(lat, lon):
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,precipitation,weathercode,windspeed_10m,visibility",
                "timezone": "auto"
            }, timeout=5
        ).json()
        c = r["current"]
        return {
            "temperature"  : c["temperature_2m"],
            "precipitation": c.get("precipitation", 0.0),
            "wind_speed"   : c["windspeed_10m"],
            "visibility"   : round(c.get("visibility", 16093), 0),
            "weather_code" : c["weathercode"]
        }
    except:
        return None

def weather_code_to_group(code):
    if code == 0: return 0
    elif code in [1,2,3]: return 1
    elif code in [45,48]: return 2
    elif code in [51,53,55,61,63,65,80,81,82]: return 3
    elif code in [71,73,75,77,85,86]: return 4
    return 0

WEATHER_LABELS = {0:"Açık", 1:"Bulutlu", 2:"Sis", 3:"Yağmur", 4:"Kar"}
WEATHER_ICONS  = {0:"☀️", 1:"☁️", 2:"🌫️", 3:"🌧️", 4:"❄️"}

# =============================================================================
# FEATURE OLUŞTURMA & TAHMİN
# =============================================================================
def c_to_f(c): return c * 9/5 + 32
def kmh_to_mph(k): return k * 0.621371

def build_features(p):
    hour = p["hour"]; month = p["month"]; dow = p["day_of_week"]
    is_weekend  = 1 if dow >= 5 else 0
    is_rush     = 1 if hour in [7,8,9,16,17,18] else 0
    is_night    = 1 if (hour < 6 or hour >= 20) else 0
    temp_f      = c_to_f(p["temperature"])
    vis_mi      = p["visibility"] / 1609.34
    wind_mph    = kmh_to_mph(p["wind_speed"])
    wg          = p["weather_group"]
    bad_weather = 1 if wg in [2,3,4] else 0
    low_vis     = 1 if vis_mi < 5 else 0
    ext_temp    = 1 if (temp_f < 32 or temp_f > 95) else 0
    junc = p["junction"]; sig = p["traffic_signal"]
    cros = p["crossing"]; stp = p["stop"]; hw = p["is_highway"]
    complex_road = junc + sig + cros + stp
    season = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}.get(month, 0)
    risk_score = bad_weather + is_night + low_vis + ext_temp + (1 if complex_road>0 else 0) + hw

    row = {
        'Temperature(F)':temp_f,'Humidity(%)':p["humidity"],'Pressure(in)':p["pressure"],
        'Visibility(mi)':vis_mi,'Wind_Speed(mph)':wind_mph,'Precipitation(in)':p["precipitation"],
        'Amenity':0,'Bump':0,'Crossing':cros,'Give_Way':0,'Junction':junc,'No_Exit':0,
        'Railway':0,'Roundabout':0,'Station':0,'Stop':stp,'Traffic_Calming':0,
        'Traffic_Signal':sig,'hour':hour,'month':month,'day_of_week':dow,
        'is_weekend':is_weekend,'is_rush_hour':is_rush,'is_night':is_night,
        'Weather_Group':wg,'bad_weather':bad_weather,'low_visibility':low_vis,
        'extreme_temp':ext_temp,'complex_road':complex_road,
        'night_bad_weather':is_night*bad_weather,'rush_complex_road':is_rush*(1 if complex_road>0 else 0),
        'night_low_vis':is_night*low_vis,'is_highway':hw,'season':season,'risk_score':risk_score,
    }
    df = pd.DataFrame([row])[list(scaler.feature_names_in_)]
    return df

def get_risk_factors(p):
    """Hangi faktörler riski etkiliyor"""
    factors = []
    wg = p["weather_group"]
    if wg == 3: factors.append(("Yağmur", "danger"))
    elif wg == 4: factors.append(("Kar", "danger"))
    elif wg == 2: factors.append(("Sis", "danger"))
    elif wg == 1: factors.append(("Bulutlu", "warn"))
    else: factors.append(("Açık hava", "ok"))
    hour = p["hour"]
    if hour < 6 or hour >= 20: factors.append(("Gece", "danger"))
    elif hour in [7,8,9,16,17,18]: factors.append(("Rush hour", "warn"))
    else: factors.append(("Normal saat", "ok"))
    if p["visibility"] < 5: factors.append(("Düşük görüş", "danger"))
    if c_to_f(p["temperature"]) < 32: factors.append(("Don riski", "danger"))
    if p["is_highway"]: factors.append(("Otoyol", "warn"))
    if p["junction"]: factors.append(("Kavşak", "warn"))
    if p["traffic_signal"]: factors.append(("Trafik ışığı", "ok"))
    if p["day_of_week"] >= 5: factors.append(("Hafta sonu", "warn"))
    return factors

def predict_risk(params, threshold=0.40):
    df_feat = build_features(params)
    scaled  = scaler.transform(df_feat)
    probs   = model.predict_proba(scaled)[0]
    pred    = 2 if probs[2] >= threshold else int(np.argmax(probs[:2]))
    risk_pct= (probs[1]*0.5 + probs[2]*1.0) * 100
    return {
        "pred"     : pred,
        "probs"    : probs,
        "risk_pct" : round(risk_pct, 1),
        "label"    : ["DÜŞÜK RİSK", "ORTA RİSK", "YÜKSEK RİSK"][pred],
        "icon"     : ["🟢", "🟡", "🔴"][pred],
        "css_class": ["risk-low","risk-mid","risk-high"][pred],
        "oneri"    : [
            "Yol koşulları güvenli görünüyor. Trafik kurallarına uyun.",
            "Dikkat gerektiren koşullar var. Hızınızı düşürün, mesafeyi artırın.",
            "Yüksek risk tespit edildi. Mümkünse seyahati erteleyin."
        ][pred],
        "factors"  : get_risk_factors(params),
    }

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#fff;margin-bottom:0.2rem">🛣 RiskRadar</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:0.62rem;color:#3a6a8a;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem">Yol Risk Analizi v3.0</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sidebar-section">📍 Konum</div>', unsafe_allow_html=True)
    IL = {
        "Samsun":(41.2867,36.3300),"İstanbul":(41.0082,28.9784),
        "Ankara":(39.9334,32.8597),"İzmir":(38.4189,27.1287),
        "Artvin":(41.1828,41.8183),"Rize":(41.0201,40.5234),
        "Trabzon":(41.0015,39.7178),"Erzurum":(39.9043,41.2679),
        "Bursa":(40.1885,29.0610),"Antalya":(36.8969,30.7133),
        "Manuel":None,
    }
    il_sec = st.selectbox("İl", list(IL.keys()), label_visibility="collapsed")
    if il_sec == "Manuel":
        lat = st.number_input("Enlem", value=41.29, format="%.4f")
        lon = st.number_input("Boylam", value=36.33, format="%.4f")
    else:
        lat, lon = IL[il_sec]

    st.markdown('<div class="sidebar-section">🌤 Hava Durumu</div>', unsafe_allow_html=True)
    otomatik = st.toggle("Otomatik (Open-Meteo)", value=True)
    if otomatik:
        hava = get_weather(lat, lon)
        if hava:
            wg = weather_code_to_group(hava["weather_code"])
            st.markdown(f"""
            <div style="background:#0d1520;border:1px solid #1e2840;border-radius:8px;padding:0.7rem 1rem;margin:0.4rem 0">
                <div style="font-family:Syne,sans-serif;font-size:1.1rem;color:#fff;margin-bottom:0.4rem">
                    {WEATHER_ICONS[wg]} {WEATHER_LABELS[wg]} · {hava['temperature']:.1f}°C
                </div>
                <div style="font-family:DM Mono,monospace;font-size:0.65rem;color:#4a6a8a;line-height:1.8">
                    💧 {hava['precipitation']:.1f} mm &nbsp;|&nbsp;
                    💨 {hava['wind_speed']:.1f} km/h &nbsp;|&nbsp;
                    👁 {hava['visibility']:.1f} m
                </div>
            </div>
            """, unsafe_allow_html=True)
            temperature=hava["temperature"]; precipitation=hava["precipitation"]
            wind_speed=hava["wind_speed"]; visibility=hava["visibility"]
            weather_group=wg
        else:
            st.warning("API erişilemedi — manuel girin")
            otomatik = False
    if not otomatik:
        temperature   = st.slider("Sıcaklık (°C)", -20, 45, 15)
        precipitation = st.slider("Yağış (mm)", 0.0, 50.0, 0.0, 0.5)
        wind_speed    = st.slider("Rüzgar (km/h)", 0, 100, 20)
        visibility    = st.slider("Görüş mesafesi (m)", 0, 20000, 10000, 500)
        w_opt = {"Açık":0,"Bulutlu":1,"Sis":2,"Yağmur":3,"Kar":4}
        weather_group = w_opt[st.selectbox("Hava", list(w_opt.keys()))]

    st.markdown('<div class="sidebar-section">⏰ Zaman</div>', unsafe_allow_html=True)
    tarih = st.date_input("Tarih", datetime.today(), label_visibility="collapsed")
    saat  = st.slider("Saat", 0, 23, datetime.now().hour)
    month = tarih.month; day_of_week = tarih.weekday()

    st.markdown('<div class="sidebar-section">🛣 Yol Özellikleri</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        is_highway     = int(st.checkbox("Otoyol", value=False))
        junction       = int(st.checkbox("Kavşak", value=False))
        crossing       = int(st.checkbox("Yaya geçidi", value=False))
    with c2:
        traffic_signal = int(st.checkbox("Trafik ışığı", value=False))
        stop_sign      = int(st.checkbox("Dur işareti", value=False))

    st.markdown('<div class="sidebar-section">⚙ Diğer</div>', unsafe_allow_html=True)
    humidity = st.slider("Nem (%)", 0, 100, 60)
    pressure = st.slider("Basınç (in Hg)", 28.0, 31.0, 29.92, 0.01)

    st.markdown("<br>", unsafe_allow_html=True)
    tahmin_btn = st.button("▶ ANALİZ YAP", use_container_width=True, type="primary")

# =============================================================================
# PARAMS
# =============================================================================
params = {
    "hour":saat,"month":month,"day_of_week":day_of_week,
    "temperature":temperature,"precipitation":precipitation,
    "wind_speed":wind_speed,"visibility":visibility,
    "weather_group":weather_group,"humidity":humidity,"pressure":pressure,
    "junction":junction,"traffic_signal":traffic_signal,
    "crossing":crossing,"stop":stop_sign,"is_highway":is_highway,
}

# =============================================================================
# ANA SAYFA
# =============================================================================
st.markdown('<div class="main-header">RiskRadar</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">Yol Risk Analiz Sistemi · LightGBM · 7.7M Kayıt · Türkiye Uyarlaması</div>', unsafe_allow_html=True)

col_map, col_res = st.columns([1.6, 1], gap="medium")

# --- SOL: HARİTA ---
with col_map:
    st.markdown('<div class="section-title">Konum Haritası</div>', unsafe_allow_html=True)
    m = folium.Map(location=[lat,lon], zoom_start=11, tiles="CartoDB dark_matter")
    folium.CircleMarker(
        [lat,lon], radius=10,
        color="#4a90a4", fill=True, fill_color="#4a90a4", fill_opacity=0.8,
        popup=f"<b>{il_sec}</b>", tooltip=f"{il_sec}"
    ).add_to(m)
    folium.CircleMarker(
        [lat,lon], radius=25,
        color="#4a90a4", fill=True, fill_color="#4a90a4", fill_opacity=0.15,
    ).add_to(m)
    map_data = st_folium(m, width="100%", height=420, returned_objects=["last_clicked"])
    if map_data and map_data.get("last_clicked"):
        clat = map_data["last_clicked"]["lat"]
        clon = map_data["last_clicked"]["lng"]
        st.markdown(
            f'<div class="oneri-box">📌 Seçilen: {clat:.4f}°N, {clon:.4f}°E — Sidebar\'dan "Manuel" seçerek bu koordinatı girebilirsiniz.</div>',
            unsafe_allow_html=True
        )

# --- SAĞ: TAHMİN ---
with col_res:
    st.markdown('<div class="section-title">Risk Tahmini</div>', unsafe_allow_html=True)

    if tahmin_btn:
        with st.spinner(""):
            sonuc = predict_risk(params, threshold=0.40)

        # Ana risk kutusu
        st.markdown(f"""
        <div class="risk-card {sonuc['css_class']}">
            {sonuc['icon']} {sonuc['label']}
            <div class="risk-subtitle">Risk Skoru: %{sonuc['risk_pct']:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Metrik kartları
        p = sonuc["probs"]
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val" style="color:#4ade80">%{p[0]*100:.0f}</div>
                <div class="metric-lbl">Düşük</div>
            </div>
            <div class="metric-box">
                <div class="metric-val" style="color:#fbbf24">%{p[1]*100:.0f}</div>
                <div class="metric-lbl">Orta</div>
            </div>
            <div class="metric-box">
                <div class="metric-val" style="color:#f87171">%{p[2]*100:.0f}</div>
                <div class="metric-lbl">Yüksek</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Olasılık barları
        st.markdown(f"""
        <div class="prob-bar-container">
            <div class="prob-row">
                <span class="prob-label">Düşük</span>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{p[0]*100:.1f}%;background:#4ade80"></div></div>
                <span class="prob-pct">%{p[0]*100:.1f}</span>
            </div>
            <div class="prob-row">
                <span class="prob-label">Orta</span>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{p[1]*100:.1f}%;background:#fbbf24"></div></div>
                <span class="prob-pct">%{p[1]*100:.1f}</span>
            </div>
            <div class="prob-row">
                <span class="prob-label">Yüksek</span>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{p[2]*100:.1f}%;background:#f87171"></div></div>
                <span class="prob-pct">%{p[2]*100:.1f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk faktörleri
        st.markdown('<div class="section-title">Risk Faktörleri</div>', unsafe_allow_html=True)
        badges = "".join([
            f'<span class="factor-badge factor-{cls}">{lbl}</span>'
            for lbl, cls in sonuc["factors"]
        ])
        st.markdown(f'<div class="factor-row">{badges}</div>', unsafe_allow_html=True)

        # Öneri
        st.markdown(f'<div class="oneri-box">{sonuc["oneri"]}</div>', unsafe_allow_html=True)

        # Detaylar
        with st.expander("Girdi Detayları"):
            st.dataframe(pd.DataFrame({
                "Parametre": ["Sıcaklık","Görüş","Rüzgar","Yağış","Nem","Basınç","Hava","Saat","Otoyol"],
                "Değer"    : [
                    f"{temperature:.1f}°C → {c_to_f(temperature):.1f}°F",
                    f"{visibility:.0f} m",
                    f"{wind_speed:.1f} km/h → {kmh_to_mph(wind_speed):.1f} mph",
                    f"{precipitation:.1f} mm",
                    f"%{humidity}",
                    f"{pressure:.2f} inHg",
                    WEATHER_LABELS[weather_group],
                    f"{saat}:00 {'(Gece)' if (saat<6 or saat>=20) else '(Gündüz)'}",
                    "Evet" if is_highway else "Hayır"
                ]
            }), hide_index=True, use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem">
            <div style="font-size:3rem;margin-bottom:1rem">🛣</div>
            <div style="font-family:Syne,sans-serif;font-size:1.1rem;color:#4a5568;margin-bottom:0.5rem">
                Analiz için hazır
            </div>
            <div style="font-family:DM Mono,monospace;font-size:0.7rem;color:#2d3748;line-height:2">
                1. Sol panelden il seçin<br>
                2. Hava verisi otomatik gelir<br>
                3. Zaman ve yol özelliklerini ayarlayın<br>
                4. ANALİZ YAP butonuna basın
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# ALT: TÜRKİYE HARİTASI
# =============================================================================
st.markdown('<div class="section-title" style="margin-top:2rem">Türkiye Risk Haritası — Anlık Koşullarla</div>', unsafe_allow_html=True)
st.markdown('<div style="font-family:DM Mono,monospace;font-size:0.68rem;color:#2d3748;margin-bottom:1rem">Mevcut hava ve zaman koşullarıyla farklı noktalar için tahmin. Noktalara tıklayın.</div>', unsafe_allow_html=True)

NOKTALAR = [
    {"il":"Artvin — Dağ yolu","lat":41.18,"lon":41.82,"is_highway":0,"junction":0,"traffic_signal":0,"crossing":0,"stop":0},
    {"il":"İstanbul — TEM",   "lat":41.07,"lon":28.78,"is_highway":1,"junction":0,"traffic_signal":1,"crossing":0,"stop":0},
    {"il":"Ankara — Şehir",   "lat":39.93,"lon":32.86,"is_highway":0,"junction":1,"traffic_signal":1,"crossing":1,"stop":0},
    {"il":"Samsun — Sahil",   "lat":41.29,"lon":36.33,"is_highway":0,"junction":0,"traffic_signal":0,"crossing":0,"stop":0},
    {"il":"Erzurum — D100",   "lat":39.90,"lon":41.27,"is_highway":1,"junction":0,"traffic_signal":0,"crossing":0,"stop":0},
    {"il":"Rize — Sahil",     "lat":41.02,"lon":40.52,"is_highway":0,"junction":0,"traffic_signal":1,"crossing":0,"stop":0},
    {"il":"Bursa — O-5",      "lat":40.18,"lon":29.06,"is_highway":1,"junction":0,"traffic_signal":0,"crossing":0,"stop":0},
    {"il":"Antalya — D400",   "lat":36.90,"lon":30.71,"is_highway":0,"junction":1,"traffic_signal":1,"crossing":0,"stop":0},
]

harita = folium.Map(location=[39.5,35.5], zoom_start=6, tiles="CartoDB dark_matter")
RENK   = {0:"#4ade80", 1:"#fbbf24", 2:"#f87171"}

for n in NOKTALAR:
    p = {**params,"junction":n["junction"],"traffic_signal":n["traffic_signal"],
         "crossing":n["crossing"],"stop":n["stop"],"is_highway":n["is_highway"]}
    s = predict_risk(p, threshold=0.40)
    renk = RENK[s["pred"]]
    folium.CircleMarker(
        [n["lat"],n["lon"]], radius=16,
        color=renk, fill=True, fill_color=renk, fill_opacity=0.75,
        popup=folium.Popup(
            f"""<div style="font-family:monospace;font-size:12px;min-width:160px">
                <b>{n['il']}</b><br>
                {s['icon']} {s['label']}<br>
                Risk: %{s['risk_pct']:.0f}<br>
                <span style="color:#888;font-size:10px">
                D:{s['probs'][0]*100:.0f}% O:{s['probs'][1]*100:.0f}% Y:{s['probs'][2]*100:.0f}%
                </span>
            </div>""",
            max_width=200
        ),
        tooltip=f"{n['il']} — %{s['risk_pct']:.0f} risk"
    ).add_to(harita)
    folium.CircleMarker(
        [n["lat"],n["lon"]], radius=28,
        color=renk, fill=True, fill_color=renk, fill_opacity=0.1,
    ).add_to(harita)

st_folium(harita, width="100%", height=380, returned_objects=[])

# Legend
lc1,lc2,lc3,lc4 = st.columns(4)
with lc1: st.markdown('<span style="font-family:DM Mono,monospace;font-size:0.7rem;color:#4ade80">● DÜŞÜK RİSK</span>', unsafe_allow_html=True)
with lc2: st.markdown('<span style="font-family:DM Mono,monospace;font-size:0.7rem;color:#fbbf24">● ORTA RİSK</span>', unsafe_allow_html=True)
with lc3: st.markdown('<span style="font-family:DM Mono,monospace;font-size:0.7rem;color:#f87171">● YÜKSEK RİSK</span>', unsafe_allow_html=True)
with lc4: st.markdown(f'<span style="font-family:DM Mono,monospace;font-size:0.7rem;color:#2d3748">⏱ {datetime.now().strftime("%H:%M")}</span>', unsafe_allow_html=True)