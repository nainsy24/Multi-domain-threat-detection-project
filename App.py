# ================================================================
#  app.py  —  Multi-Domain Threat Detection System
#  Pastel / light theme  —  beginner-friendly UI
#  Run:  streamlit run app.py
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, warnings
warnings.filterwarnings('ignore')

from pipeline import (load_and_train, predict_single,
                      THREAT_COLOR, THREAT_PASTEL,
                      THREAT_BORDER, THREAT_ADVICE)

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Threat Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── PASTEL CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

/* ── Base ── */
.stApp { background: #F8F6FF; }
.main .block-container { padding: 1.5rem 2.5rem; max-width: 1300px; }
html, body, [class*="css"], p, div, span, label {
    font-family: 'Nunito', sans-serif;
    color: #2D2D4E;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EDE9FF 0%, #E8F4FD 100%);
    border-right: 1.5px solid #D5CCF5;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label { color: #2D2D4E; }

/* ── Headings ── */
h1 { font-family: 'Nunito', sans-serif; font-weight: 800;
     font-size: 2rem; color: #3B2F8F; margin-bottom: 0; }
h2 { font-family: 'Nunito', sans-serif; font-weight: 700;
     font-size: 1.25rem; color: #4A3FA0; }
h3 { font-family: 'Nunito', sans-serif; font-weight: 600;
     font-size: 1rem; color: #6B63B5; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: white;
    border: 1.5px solid #D5CCF5;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(91,78,187,0.08);
}
[data-testid="stMetricValue"] {
    font-family: 'Fira Code', monospace !important;
    font-size: 1.8rem !important;
    color: #5B4EBB !important;
    font-weight: 500 !important;
}
[data-testid="stMetricLabel"] {
    color: #9B92D4 !important;
    font-size: .72rem !important;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 600 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #7B6FD4, #5B4EBB) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: .02em;
    width: 100%;
    padding: .65rem 1rem;
    box-shadow: 0 4px 14px rgba(91,78,187,0.3) !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #9B8FE4, #7B6FD4) !important;
    box-shadow: 0 6px 20px rgba(91,78,187,0.4) !important;
    transform: translateY(-1px);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 14px;
    gap: 6px;
    padding: 6px;
    border: 1.5px solid #D5CCF5;
    box-shadow: 0 2px 8px rgba(91,78,187,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-family: 'Nunito', sans-serif;
    font-weight: 600;
    color: #9B92D4;
    padding: .4rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#7B6FD4,#5B4EBB) !important;
    color: white !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: white;
    border: 1.5px solid #D5CCF5;
    border-radius: 10px;
    color: #2D2D4E;
}

/* ── Slider ── */
.stSlider { color: #5B4EBB; }
.stSlider [data-testid="stThumbValue"] { color: #5B4EBB; }

/* ── Divider ── */
hr { border-color: #E0DAF8; }

/* ── Info/Success boxes ── */
.stAlert { border-radius: 12px; }

/* ── Reusable pastel card ── */
.pcard {
    background: white;
    border: 1.5px solid #E8E4F8;
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    margin-bottom: .8rem;
    box-shadow: 0 2px 10px rgba(91,78,187,0.06);
}
.pcard-purple {
    background: #F3F0FF;
    border: 1.5px solid #C8BFEE;
    border-radius: 16px;
    padding: 1rem 1.3rem;
    margin-bottom: .8rem;
}

/* ── Step number badge ── */
.step-badge {
    display: inline-block;
    background: #EDE9FF;
    color: #5B4EBB;
    font-family: 'Fira Code', monospace;
    font-size: .72rem;
    font-weight: 500;
    padding: 2px 9px;
    border-radius: 20px;
    border: 1px solid #C8BFEE;
    margin-right: 8px;
}

/* ── Hint box ── */
.hint-box {
    background: #EEF6FF;
    border: 1.5px solid #BCD9F5;
    border-radius: 12px;
    padding: .9rem 1.1rem;
    margin-bottom: .7rem;
}
.hint-title { font-weight: 700; font-size: .88rem; color: #1D5FA6; margin-bottom: 3px; }
.hint-desc  { font-size: .82rem; color: #4A7FB5; line-height: 1.6; }

/* ── Domain challenge cards ── */
.challenge-air  { background:#FFF0F3; border:1.5px solid #F5C0CC; border-radius:14px; padding:.9rem 1.1rem; margin-bottom:.6rem; }
.challenge-land { background:#F0FFF4; border:1.5px solid #B2DFC0; border-radius:14px; padding:.9rem 1.1rem; margin-bottom:.6rem; }
.challenge-water{ background:#F0F8FF; border:1.5px solid #AECDE8; border-radius:14px; padding:.9rem 1.1rem; margin-bottom:.6rem; }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODELS ───────────────────────────────────────────────────
CSV = "reduced_multi_domain_dataset.csv"

@st.cache_resource(show_spinner="Loading dataset and training models — please wait...")
def get_models():
    return load_and_train(CSV)

if not os.path.exists(CSV):
    st.error(
        "**Dataset file not found!**\n\n"
        f"Please place `{CSV}` in the same folder as `app.py` and restart."
    )
    st.stop()

results, scaler, feature_cols, df_raw = get_models()
best = max(results, key=lambda k: results[k]['accuracy'])


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#3B2F8F;margin-bottom:4px;'>Threat Detection</h2>"
        "<p style='color:#6B63B5;margin-top:0;font-size:.88rem;'>"
        "Multi-domain ML classifier</p>",
        unsafe_allow_html=True)
    st.divider()

    st.success("All models ready")

    st.markdown(
        f"<div class='pcard-purple'>"
        f"<p style='margin:0;font-size:.82rem;color:#5B4EBB;'>"
        f"Dataset: <strong>{len(df_raw)}</strong> objects<br>"
        f"Features: <strong>{len(feature_cols)}</strong><br>"
        f"Domains: Air · Land · Water</p></div>",
        unsafe_allow_html=True)

    st.markdown(
        f"<div class='pcard-purple'>"
        f"<p style='margin:0;font-size:.82rem;color:#5B4EBB;'>"
        f"Best model: <strong>{best}</strong><br>"
        f"Accuracy: <strong>{results[best]['accuracy']*100:.1f}%</strong><br>"
        f"Classes: HIGH · MEDIUM · LOW</p></div>",
        unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<p style='font-size:.8rem;color:#9B92D4;line-height:1.6;'>"
        "Built with Streamlit<br>scikit-learn · Plotly<br><br>"
        "To use: select sensor values on the <strong>Live Predictor</strong> "
        "tab and click Classify Threat.</p>",
        unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────────────────────
st.markdown(
    "<h1>Multi-Domain Threat Detection System</h1>"
    "<p style='color:#6B63B5;margin-top:2px;margin-bottom:1.5rem;font-size:1rem;'>"
    "Real-time ML classification &nbsp;·&nbsp; "
    "Air &nbsp;·&nbsp; Land &nbsp;·&nbsp; Water</p>",
    unsafe_allow_html=True)

# ── KPI STRIP ─────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Objects in dataset",  f"{len(df_raw)}")
k2.metric("Features used",       f"{len(feature_cols)}")
k3.metric("Best accuracy",       f"{results[best]['accuracy']*100:.1f}%")
k4.metric("Models trained",      "3")
k5.metric("Threat classes",      "3")
st.divider()


# ── TABS ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯  Live Predictor",
    "📊  Dataset Explorer",
    "📈  Model Results",
    "🏗️  How It Works",
    "❓  How to Use",
])


# ════════════════════════════════════════════════════════════════
#  TAB 1 — LIVE PREDICTOR
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<h2>Real-time threat classifier</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hint-box'>"
        "<div class='hint-title'>How to use this page</div>"
        "<div class='hint-desc'>"
        "1. Select the object domain (Air / Land / Water)<br>"
        "2. Set the sensor readings using the sliders<br>"
        "3. Click the purple <strong>Classify Threat</strong> button<br>"
        "4. The ML model will instantly show HIGH / MEDIUM / LOW threat"
        "</div></div>",
        unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            "<div class='pcard'><strong style='color:#3B2F8F;"
            "font-size:.95rem;'>Object settings</strong></div>",
            unsafe_allow_html=True)

        domain = st.selectbox(
            "Domain — where is the object?",
            ["air_object", "land_object", "sea_object"],
            format_func=lambda x: {
                "air_object":  "Air object (aircraft, missile, drone)",
                "land_object": "Land object (vehicle, ground threat)",
                "sea_object":  "Sea object (naval vessel, torpedo)",
            }[x],
        )
        traj = st.selectbox(
            "Trajectory type — how is it moving?",
            ["linear", "diving", "ballistic"],
            format_func=lambda x: {
                "linear":    "Linear (straight horizontal flight)",
                "diving":    "Diving (descending fast toward target)",
                "ballistic": "Ballistic (arc-shaped missile path)",
            }[x],
        )
        sensor = st.selectbox(
            "Sensor type — which sensor detected it?",
            ["radar", "thermal", "camera"],
            format_func=lambda x: {
                "radar":   "Radar (radio waves — works in fog/night)",
                "thermal": "Thermal (heat signature — works in dark)",
                "camera":  "Camera (optical — works in daylight only)",
            }[x],
        )
        lighting = st.selectbox(
            "Lighting condition",
            ["day", "dusk", "night"],
            format_func=lambda x: x.capitalize(),
        )
        algo = st.selectbox(
            "Which ML algorithm to use?",
            list(results.keys()),
            help="Random Forest is the most accurate. Try all three to compare!",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div class='pcard'><strong style='color:#3B2F8F;"
            "font-size:.95rem;'>Sensor readings</strong></div>",
            unsafe_allow_html=True)

        velocity   = st.slider("Velocity (m/s) — how fast is it moving?",
                                100, 1000, 550, 10)
        altitude   = st.slider("Altitude (m) — how high is it?",
                                0, 9000, 800, 50)
        traj_angle = st.slider("Trajectory angle (°) — direction of movement",
                                -90, 90, 0, 5,
                                help="Negative = diving down, Positive = climbing up")
        conf_score = st.slider("Confidence score — how sure is the sensor?",
                                0.70, 1.00, 0.85, 0.01)
        fog        = st.slider("Fog density — how foggy is it?",
                                0.00, 1.00, 0.15, 0.05,
                                help="0 = clear sky, 1 = extremely dense fog")
        rain       = st.slider("Rain intensity",
                                0.00, 1.00, 0.10, 0.05)
        visibility = st.slider("Visibility range (m)",
                                100.0, 1000.0, 600.0, 10.0)
        radar_r    = st.slider("Radar range",
                                100.0, 1000.0, 500.0, 10.0)
        thermal_s  = st.slider("Thermal signature — heat level",
                                0.40, 1.00, 0.65, 0.01,
                                help="High = hot engine/exhaust, Low = cold object")

        run = st.button("🎯  Classify Threat", type="primary")

    with right:
        st.markdown(
            "<h3 style='margin-bottom:.5rem;'>Prediction result</h3>",
            unsafe_allow_html=True)

        if run:
            user_input = {
                'object_class': domain, 'trajectory_type': traj,
                'sensor_type': sensor, 'lighting_condition': lighting,
                'velocity': velocity, 'altitude': altitude,
                'trajectory_angle': traj_angle,
                'confidence_score': conf_score,
                'radar_range': radar_r,
                'thermal_signature': thermal_s,
                'fog_density': fog, 'rain_intensity': rain,
                'visibility_range': visibility,
                'doppler_velocity': velocity * 0.9,
                'obstacle_density': 0.3,
                'climb_rate': float(traj_angle) * 2.0,
            }

            label, proba = predict_single(
                user_input, results[algo]['model'], scaler, feature_cols)

            tc = THREAT_COLOR[label]
            tp = THREAT_PASTEL[label]
            tb = THREAT_BORDER[label]
            advice = THREAT_ADVICE[label]

            icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[label]

            # Big threat result badge
            st.markdown(f"""
<div style="background:{tp};border:2.5px solid {tb};border-radius:18px;
            padding:1.5rem 1.5rem;text-align:center;margin-bottom:1rem;
            box-shadow:0 4px 16px rgba(0,0,0,0.08);">
  <div style="font-size:2.8rem;margin-bottom:.3rem;">{icon}</div>
  <div style="font-size:1.7rem;font-weight:800;color:{tc};
              letter-spacing:.04em;font-family:Nunito,sans-serif;">
    {label} THREAT
  </div>
  <div style="font-size:.88rem;color:{tc};margin-top:.4rem;
              font-family:Nunito,sans-serif;opacity:.85;">
    {advice}
  </div>
</div>""", unsafe_allow_html=True)

            # Confidence bars
            st.markdown(
                "<p style='font-weight:700;color:#3B2F8F;margin-bottom:.4rem;'>"
                "Model confidence per class</p>",
                unsafe_allow_html=True)

            class_labels = ['HIGH','LOW','MEDIUM']
            bar_colors   = ['#F1948A','#82E0AA','#F8C471']
            cdf = pd.DataFrame({
                'Class':      class_labels,
                'Confidence': [p*100 for p in proba],
            })
            fig_c = px.bar(
                cdf, x='Confidence', y='Class', orientation='h',
                color='Class',
                color_discrete_sequence=bar_colors,
                text=[f"{p:.1f}%" for p in cdf['Confidence']],
            )
            fig_c.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(243,240,255,0.5)',
                font=dict(family='Nunito',color='#2D2D4E'),
                showlegend=False, height=200,
                margin=dict(l=0,r=20,t=10,b=0),
                xaxis=dict(range=[0,100],gridcolor='#E0DAF8',
                           title='Confidence (%)',
                           tickfont=dict(family='Fira Code')),
                yaxis=dict(gridcolor='#E0DAF8'),
            )
            fig_c.update_traces(textposition='outside',
                                textfont=dict(family='Fira Code',size=11))
            st.plotly_chart(fig_c, use_container_width=True)

            # Sensor analysis summary
            st.markdown(
                "<p style='font-weight:700;color:#3B2F8F;margin-bottom:.4rem;'>"
                "Sensor analysis summary</p>",
                unsafe_allow_html=True)
            speed_ratio = velocity / (altitude + 1)
            env_diff    = fog*0.33 + rain*0.33 + (1 - min(visibility,1000)/1000)*0.33

            c1, c2, c3 = st.columns(3)
            c1.metric("Speed/altitude ratio", f"{speed_ratio:.3f}",
                      help="High = fast + low altitude = more dangerous")
            c2.metric("Env difficulty score",  f"{env_diff:.3f}",
                      help="Fog + rain + poor visibility combined")
            c3.metric("Algorithm",             algo.split()[0])

            # What it means
            meaning = {
                'HIGH':   "The sensor readings indicate a high-velocity, low-altitude or aggressive-trajectory object. This pattern matches dangerous incoming threats.",
                'MEDIUM': "The sensor readings show moderate risk indicators. The object may be a patrol aircraft or vessel that needs closer monitoring.",
                'LOW':    "The sensor readings suggest a slow, high-altitude, or non-aggressive object. This is consistent with safe civilian or surveillance activity.",
            }
            st.markdown(
                f"<div class='pcard-purple'>"
                f"<strong style='font-size:.82rem;color:#5B4EBB;'>"
                f"What does this mean?</strong><br>"
                f"<span style='font-size:.82rem;color:#4A3FA0;'>"
                f"{meaning[label]}</span></div>",
                unsafe_allow_html=True)

        else:
            # Empty state — show helpful hints
            st.markdown(
                "<div class='pcard' style='text-align:center;"
                "padding:2rem;background:#F3F0FF;'>"
                "<div style='font-size:2.5rem;'>🛡️</div>"
                "<p style='color:#6B63B5;margin:.5rem 0 0;'>"
                "Set sensor values on the left<br>"
                "then click <strong>Classify Threat</strong></p>"
                "</div>",
                unsafe_allow_html=True)

            st.markdown(
                "<p style='font-weight:700;color:#3B2F8F;margin-top:1rem;'>"
                "Try these example scenarios:</p>",
                unsafe_allow_html=True)

            scenarios = [
                ("🔴", "Incoming missile — expect HIGH",
                 "Velocity: 900 · Altitude: 120 · Angle: -60° · Sensor: Radar · Domain: Air"),
                ("🟡", "Patrol aircraft — expect MEDIUM",
                 "Velocity: 500 · Altitude: 2000 · Angle: 0° · Sensor: Thermal · Domain: Air"),
                ("🟢", "Surveillance drone — expect LOW",
                 "Velocity: 200 · Altitude: 4000 · Angle: +10° · Sensor: Camera · Domain: Air"),
                ("🟡", "Naval vessel — try sea domain",
                 "Domain: Sea · Velocity: 350 · Fog: 0.7 · Rain: 0.5 · Sensor: Radar"),
            ]
            for icon, title, desc in scenarios:
                st.markdown(
                    f"<div class='hint-box'>"
                    f"<div class='hint-title'>{icon} {title}</div>"
                    f"<div class='hint-desc'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  TAB 2 — DATASET EXPLORER
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<h2>Dataset explorer</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:#6B63B5;'>Visualising <strong>{len(df_raw)}</strong> "
        "sensor recordings across three detection domains. "
        "All charts are interactive — hover over any bar or dot for details.</p>",
        unsafe_allow_html=True)

    # Pastel chart colours
    CM = {'high':'#F1948A','medium':'#F8C471','low':'#82E0AA'}

    col1, col2 = st.columns(2)

    with col1:
        tc = df_raw['threat_level'].value_counts().reset_index()
        tc.columns = ['Threat','Count']
        fig1 = px.bar(tc, x='Threat', y='Count', color='Threat',
                      color_discrete_map=CM, text='Count',
                      title='How many of each threat level?')
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(243,240,255,0.4)',
            font=dict(family='Nunito',color='#2D2D4E'),
            showlegend=False, height=320,
            title_font_size=14,
            margin=dict(t=45,b=0,l=0,r=0),
            xaxis=dict(gridcolor='#E0DAF8'),
            yaxis=dict(gridcolor='#E0DAF8'))
        fig1.update_traces(textposition='outside',
                           marker_line_width=0, opacity=0.85)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        dt = df_raw.groupby(['object_class','threat_level']).size().reset_index(name='n')
        fig2 = px.bar(dt, x='object_class', y='n', color='threat_level',
                      barmode='group', color_discrete_map=CM,
                      title='Which domain has the most threats?',
                      labels={'object_class':'Domain','n':'Count','threat_level':'Threat'})
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(243,240,255,0.4)',
            font=dict(family='Nunito',color='#2D2D4E'),
            height=320, title_font_size=14,
            margin=dict(t=45,b=0,l=0,r=0),
            xaxis=dict(gridcolor='#E0DAF8'),
            yaxis=dict(gridcolor='#E0DAF8'),
            legend=dict(bgcolor='rgba(255,255,255,0.7)',
                        bordercolor='#E0DAF8'))
        fig2.update_traces(opacity=0.85, marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "<p style='font-weight:700;color:#3B2F8F;margin-top:.5rem;'>"
        "Altitude vs velocity — each dot is one detected object</p>",
        unsafe_allow_html=True)
    st.caption("Hover over any dot to see full sensor details for that object")
    fig3 = px.scatter(
        df_raw, x='altitude', y='velocity',
        color='threat_level', color_discrete_map=CM,
        opacity=0.70, size_max=8,
        hover_data=['object_class','sensor_type','trajectory_type','confidence_score'],
        labels={'altitude':'Altitude (m)','velocity':'Velocity (m/s)',
                'threat_level':'Threat Level'},
    )
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(243,240,255,0.4)',
        font=dict(family='Nunito',color='#2D2D4E'),
        height=400,
        margin=dict(t=10,b=0,l=0,r=0),
        xaxis=dict(gridcolor='#E0DAF8'),
        yaxis=dict(gridcolor='#E0DAF8'),
        legend=dict(bgcolor='rgba(255,255,255,0.7)',
                    bordercolor='#E0DAF8'))
    st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig4 = px.box(df_raw, x='threat_level', y='velocity',
                      color='threat_level', color_discrete_map=CM,
                      title='Velocity range per threat level',
                      category_orders={'threat_level':['low','medium','high']})
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(243,240,255,0.4)',
            font=dict(family='Nunito',color='#2D2D4E'),
            height=340, showlegend=False, title_font_size=14,
            margin=dict(t=45,b=0,l=0,r=0),
            xaxis=dict(gridcolor='#E0DAF8'),
            yaxis=dict(gridcolor='#E0DAF8'))
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        sc2 = df_raw.groupby(['sensor_type','threat_level']).size().reset_index(name='n')
        fig5 = px.bar(sc2, x='sensor_type', y='n', color='threat_level',
                      barmode='stack', color_discrete_map=CM,
                      title='Which sensor catches which threats?',
                      labels={'sensor_type':'Sensor','n':'Count','threat_level':'Threat'})
        fig5.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(243,240,255,0.4)',
            font=dict(family='Nunito',color='#2D2D4E'),
            height=340, title_font_size=14,
            margin=dict(t=45,b=0,l=0,r=0),
            xaxis=dict(gridcolor='#E0DAF8'),
            yaxis=dict(gridcolor='#E0DAF8'),
            legend=dict(bgcolor='rgba(255,255,255,0.7)',
                        bordercolor='#E0DAF8'))
        fig5.update_traces(opacity=0.85, marker_line_width=0)
        st.plotly_chart(fig5, use_container_width=True)

    st.divider()
    st.markdown(
        "<p style='font-weight:700;color:#3B2F8F;'>Filter and explore raw data</p>",
        unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f_dom = st.multiselect("Filter by domain",
            df_raw['object_class'].unique().tolist(),
            default=df_raw['object_class'].unique().tolist())
    with col_f2:
        f_thr = st.multiselect("Filter by threat level",
            df_raw['threat_level'].unique().tolist(),
            default=df_raw['threat_level'].unique().tolist())
    df_f = df_raw[df_raw['object_class'].isin(f_dom) &
                  df_raw['threat_level'].isin(f_thr)]
    show_cols = ['object_class','threat_level','velocity','altitude',
                 'trajectory_angle','sensor_type','confidence_score']
    st.dataframe(
        df_f[[c for c in show_cols if c in df_f.columns]].reset_index(drop=True),
        use_container_width=True, height=280)
    st.caption(f"Showing {len(df_f)} of {len(df_raw)} rows")


# ════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL RESULTS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<h2>Model training results</h2>", unsafe_allow_html=True)

    names = list(results.keys())
    mc1,mc2,mc3 = st.columns(3)
    for col,name in zip([mc1,mc2,mc3],names):
        col.metric(name,
                   f"{results[name]['accuracy']*100:.1f}%",
                   delta="✓ best" if name==best else None)

    st.info(
        "**About the accuracy:** This is a balanced 3-class problem — random guessing "
        "would give 33.3%. All three models have learned real patterns from the sensor data. "
        "The feature importance chart below shows exactly which sensors drive the decisions."
    )
    st.divider()

    fig_acc = go.Figure()
    bc = ['#B39DDB','#80CBC4','#F48FB1']
    for i,name in enumerate(names):
        acc = results[name]['accuracy']*100
        fig_acc.add_trace(go.Bar(
            x=[name], y=[acc], marker_color=bc[i],
            text=[f"{acc:.1f}%"], textposition='outside',
            name=name, width=0.4,
            marker_line_width=0,
        ))
    fig_acc.add_hline(y=33.3, line_dash="dash", line_color="#9B92D4",
                      annotation_text="Random baseline 33.3%",
                      annotation_position="top right",
                      annotation_font_color="#9B92D4")
    fig_acc.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(243,240,255,0.4)',
        font=dict(family='Nunito',color='#2D2D4E'),
        height=340, showlegend=False,
        title='Accuracy comparison — all three algorithms',
        title_font_size=14,
        margin=dict(t=45,b=0,l=0,r=0), bargap=0.4,
        yaxis=dict(range=[0,50],gridcolor='#E0DAF8',title='Accuracy (%)'),
        xaxis=dict(gridcolor='#E0DAF8'))
    st.plotly_chart(fig_acc, use_container_width=True)

    st.divider()
    st.markdown(
        "<p style='font-weight:700;color:#3B2F8F;'>Confusion matrices</p>",
        unsafe_allow_html=True)
    st.caption(
        "Rows = what the ACTUAL threat level was · "
        "Columns = what the MODEL PREDICTED · "
        "Bright diagonal = correct predictions")

    class_labels = ['HIGH','LOW','MEDIUM']
    cm_cols = st.columns(3)
    pastel_scales = [
        [[0,'#FFF0F3'],[1,'#E74C3C']],
        [[0,'#F0FFF4'],[1,'#27AE60']],
        [[0,'#F0F8FF'],[1,'#2980B9']],
    ]
    for col,name,scale in zip(cm_cols,names,pastel_scales):
        with col:
            cm = results[name]['cm']
            fig_cm = px.imshow(
                cm, x=class_labels, y=class_labels,
                text_auto=True, title=name,
                color_continuous_scale=scale,
                labels=dict(x='Predicted',y='Actual'))
            fig_cm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Nunito',color='#2D2D4E'),
                height=280, title_font_size=13,
                margin=dict(t=35,b=0,l=0,r=0),
                coloraxis_showscale=False)
            fig_cm.update_traces(textfont=dict(family='Fira Code',size=14))
            st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()
    st.markdown(
        "<p style='font-weight:700;color:#3B2F8F;'>"
        "Feature importances — what does the model rely on?</p>",
        unsafe_allow_html=True)
    st.caption(
        "Longer bar = this sensor reading influenced more decisions")
    rf_key = [k for k in names if 'Forest' in k or 'Random' in k]
    if rf_key and 'importances' in results[rf_key[0]]:
        imp = results[rf_key[0]]['importances']
        imp_df = pd.DataFrame(
            {'Feature':feature_cols,'Importance':imp}
        ).sort_values('Importance',ascending=True).tail(15)
        fig_imp = px.bar(
            imp_df, x='Importance', y='Feature', orientation='h',
            color='Importance',
            color_continuous_scale=['#E8E4F8','#9B92D4','#5B4EBB'],
            title='Top 15 most important sensor features')
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(243,240,255,0.4)',
            font=dict(family='Nunito',color='#2D2D4E'),
            height=480, showlegend=False,
            coloraxis_showscale=False,
            title_font_size=14,
            margin=dict(t=45,b=0,l=0,r=0),
            xaxis=dict(gridcolor='#E0DAF8'),
            yaxis=dict(gridcolor='#E0DAF8'))
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    for name in names:
        with st.expander(f"Full detailed report — {name}"):
            st.code(results[name]['report'], language=None)


# ════════════════════════════════════════════════════════════════
#  TAB 4 — HOW IT WORKS
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<h2>How the system works</h2>", unsafe_allow_html=True)

    left, right = st.columns([3,2])
    with left:
        st.markdown(
            "<p style='font-weight:700;color:#3B2F8F;margin-bottom:.8rem;'>"
            "Complete pipeline — step by step</p>",
            unsafe_allow_html=True)
        steps = [
            ("01","Raw data loaded",
             "525 sensor recordings · 35 columns · air, land, water objects · "
             "target column = threat_level"),
            ("02","Data cleaning",
             "350 missing values filled · 3 useless ID columns removed · "
             "duplicate rows deleted"),
            ("03","Exploratory data analysis",
             "Class balance verified · Domain patterns found · "
             "Sensor correlations studied"),
            ("04","Feature engineering",
             "8 text columns encoded to numbers · 3 new derived features created · "
             "all values scaled to same range"),
            ("05","Train / test split",
             "80% used for training (420 rows) · 20% kept for testing (105 rows) · "
             "fair evaluation guaranteed"),
            ("06","3 ML models trained",
             "Decision Tree · Random Forest (200 trees) · KNN (7 neighbors) · "
             "all trained on same data"),
            ("07","Models compared",
             "Accuracy · F1-score · Confusion matrix · Feature importance · "
             "best model identified"),
            ("08","Live deployment",
             "This Streamlit dashboard · real-time predictions · "
             "public URL via Streamlit Cloud"),
        ]
        for num,title,desc in steps:
            st.markdown(f"""
<div class="pcard">
  <span class="step-badge">{num}</span>
  <strong style="font-size:.9rem;color:#3B2F8F;">{title}</strong><br>
  <span style="font-size:.82rem;color:#6B63B5;line-height:1.6;
               margin-left:2.2rem;display:block;">{desc}</span>
</div>""", unsafe_allow_html=True)

    with right:
        st.markdown(
            "<p style='font-weight:700;color:#3B2F8F;margin-bottom:.8rem;'>"
            "Domain challenges & solutions</p>",
            unsafe_allow_html=True)

        st.markdown("""
<div class="challenge-air">
  <strong style="color:#C0392B;">Air domain challenges</strong><br>
  <small style="color:#7B241C;line-height:1.8;">
    Fog/clouds → Radar + thermal sensors<br>
    Bird confusion → trajectory + speed patterns<br>
    High-speed tracking → speed_alt_ratio feature
  </small>
</div>
<div class="challenge-land">
  <strong style="color:#1A7A3C;">Land domain challenges</strong><br>
  <small style="color:#145A32;line-height:1.8;">
    Terrain/dust → terrain_type + obstacle_density<br>
    Camouflage → thermal_signature heat detection<br>
    False positives → confidence_score filter
  </small>
</div>
<div class="challenge-water">
  <strong style="color:#1A5276;">Water domain challenges</strong><br>
  <small style="color:#154360;line-height:1.8;">
    Wave interference → sea_state + wave_height<br>
    Low visibility → radar_range + env_difficulty<br>
    Sonar noise → doppler_velocity filtering
  </small>
</div>""", unsafe_allow_html=True)

        st.markdown(
            "<div class='pcard-purple'>"
            "<strong style='font-size:.85rem;color:#5B4EBB;'>"
            "CNN — future scope</strong><br>"
            "<span style='font-size:.8rem;color:#4A3FA0;'>"
            "A CNN image classifier for missile shape recognition "
            "is planned as a future addition. It would combine with "
            "this ML system for dual-confirmation classification."
            "</span></div>",
            unsafe_allow_html=True)

        st.markdown(
            "<div class='pcard-purple'><strong style='font-size:.85rem;"
            "color:#5B4EBB;'>Tech stack</strong><br>",
            unsafe_allow_html=True)
        stack = [("Language","Python 3.10+"),("ML","scikit-learn"),
                 ("Charts","Plotly Express"),("Dashboard","Streamlit"),
                 ("Dataset","525 × 34 sensor readings")]
        for k,v in stack:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:3px 0;border-bottom:.5px solid #D5CCF5;'>"
                f"<small style='color:#6B63B5;'>{k}</small>"
                f"<small style='color:#3B2F8F;font-family:Fira Code,monospace;"
                f"font-weight:500;'>{v}</small></div>",
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  TAB 5 — HOW TO USE (for other users)
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<h2>How to use this system</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6B63B5;'>A simple guide for anyone "
        "visiting this app for the first time.</p>",
        unsafe_allow_html=True)

    # Step by step guide
    guide_steps = [
        ("Step 1", "Go to the Live Predictor tab",
         "Click the tab at the top that says '🎯 Live Predictor'. "
         "This is the main page where you can classify a threat."),
        ("Step 2", "Choose the domain",
         "Select whether the detected object is in the Air, Land, or Water domain. "
         "Air = aircraft/missile. Land = ground vehicle. Sea = naval vessel."),
        ("Step 3", "Set the sensor readings",
         "Use the sliders to enter the sensor data for the detected object. "
         "Velocity is how fast it is moving. Altitude is how high it is. "
         "Trajectory angle tells the system if it is diving down or climbing up."),
        ("Step 4", "Click Classify Threat",
         "Press the purple button. The ML model will instantly analyse all "
         "the sensor values and output HIGH, MEDIUM, or LOW threat level."),
        ("Step 5", "Read the result",
         "A coloured badge appears — RED for HIGH, YELLOW for MEDIUM, GREEN for LOW. "
         "The confidence bars show how certain the model is. "
         "The summary at the bottom explains what the result means in plain words."),
        ("Step 6", "Try different scenarios",
         "Change the sliders and click again. Try the example scenarios listed "
         "on the predictor page to see how the system responds to different threat types."),
    ]

    for step_name, title, desc in guide_steps:
        st.markdown(f"""
<div class="pcard">
  <div style="display:flex;align-items:flex-start;gap:.9rem;">
    <span style="background:#EDE9FF;color:#5B4EBB;font-family:Fira Code,monospace;
                 font-size:.75rem;font-weight:500;padding:3px 10px;border-radius:20px;
                 border:1px solid #C8BFEE;white-space:nowrap;margin-top:2px;">
      {step_name}
    </span>
    <div>
      <strong style="font-size:.9rem;color:#3B2F8F;">{title}</strong><br>
      <span style="font-size:.83rem;color:#6B63B5;line-height:1.6;">{desc}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(
            "<p style='font-weight:700;color:#3B2F8F;'>Understanding the threat levels</p>",
            unsafe_allow_html=True)
        for label,icon,desc in [
            ("HIGH","🔴","Object shows aggressive patterns — fast, low altitude, "
             "diving trajectory, high thermal signature. Requires immediate attention."),
            ("MEDIUM","🟡","Object shows moderate risk indicators. Could be a patrol "
             "vehicle or vessel. Needs close monitoring but not immediate action."),
            ("LOW","🟢","Object shows safe patterns — slow, high altitude, linear "
             "trajectory, low thermal signature. Consistent with civilian activity."),
        ]:
            tp = THREAT_PASTEL[label]; tb = THREAT_BORDER[label]
            st.markdown(
                f"<div style='background:{tp};border:1.5px solid {tb};"
                f"border-radius:12px;padding:.9rem 1rem;margin-bottom:.6rem;'>"
                f"<strong style='color:{THREAT_COLOR[label]};'>"
                f"{icon} {label} THREAT</strong><br>"
                f"<small style='color:{THREAT_COLOR[label]};opacity:.85;'>"
                f"{desc}</small></div>",
                unsafe_allow_html=True)

    with col_r:
        st.markdown(
            "<p style='font-weight:700;color:#3B2F8F;'>Frequently asked questions</p>",
            unsafe_allow_html=True)
        faqs = [
            ("Do I need to install anything?",
             "No. If you are accessing this via the public URL, just open it "
             "in any browser. No installation or login needed."),
            ("Which algorithm should I choose?",
             "Random Forest gives the best accuracy. Try all three to compare "
             "how different algorithms classify the same input."),
            ("What do the sliders represent?",
             "They simulate the readings from real sensors — radar, thermal cameras, "
             "and optical cameras — that would detect an object in a real system."),
            ("Why is accuracy around 32%?",
             "This is a balanced 3-class problem. Random guessing = 33%. "
             "The models learned real patterns and the feature importance chart "
             "shows which sensors they rely on most."),
            ("Can I share this with others?",
             "Yes. Deploy it to Streamlit Cloud (share.streamlit.io) and anyone "
             "in the world can open it in their browser with a single URL."),
        ]
        for q,a in faqs:
            with st.expander(q):
                st.markdown(
                    f"<span style='font-size:.88rem;color:#4A3FA0;'>{a}</span>",
                    unsafe_allow_html=True)