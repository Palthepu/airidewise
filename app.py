import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI RideWise | Bike Demand Forecast",
    page_icon="🏍️",
    layout="wide"
)

# --------------------------------------------------
# GLOBAL CSS
# --------------------------------------------------
st.markdown("""
<style>

/* Remove top white gap */
header {visibility: hidden;}
.stApp {
    margin-top: -80px;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Center title */
.center-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #f8fafc;
}
.center-subtitle {
    text-align: center;
    font-size: 16px;
    color: #cbd5f5;
    margin-bottom: 25px;
}

/* Input labels */
.stSelectbox label,
.stSlider label,
.stRadio label {
    color: #e5e7eb !important;
    font-weight: 600;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2563eb, #1e40af);
}
[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(to right, #2563eb, #38bdf8);
    color: white;
    font-size: 18px;
    font-weight: 700;
    border-radius: 12px;
    border: none;
    padding: 0.5em 1.2em;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-size: 32px;
    font-weight: 800;
}
[data-testid="stMetricLabel"] {
    color: #e5e7eb !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

MODEL_PATH = os.path.join(ROOT_DIR, "bike_demand_model.pkl")
IMAGE_PATH = os.path.join(BASE_DIR, "bike.png")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

FEATURES = [
    "season", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"
]

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🏍️ AI RideWise")
if os.path.exists(IMAGE_PATH):
    st.sidebar.image(Image.open(IMAGE_PATH), use_container_width=True)

st.sidebar.markdown("---")
mode = st.sidebar.radio("🔮 Forecast Type", ["Hourly ⏱️", "Daily 📅"])

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="center-title">🚴 Bike Demand Forecasting</div>
<div class="center-subtitle">
Traditional planning strengthened by Machine Learning
</div>
""", unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
st.subheader("🧩 Prediction Inputs")

c1, c2 = st.columns(2)
with c1:
    season = st.selectbox("🍃 Season", [1, 2, 3, 4])
with c2:
    mnth = st.selectbox("📅 Month", list(range(1, 13)))

c3, c4 = st.columns(2)
with c3:
    weekday = st.selectbox("🗓️ Weekday (for Hourly)", list(range(7)))
with c4:
    holiday = st.selectbox("🎉 Holiday", [0, 1])

c5, c6 = st.columns(2)
with c5:
    workingday = st.selectbox("🏢 Working Day", [0, 1])
with c6:
    weathersit = st.selectbox("🌦️ Weather Situation", [1, 2, 3, 4])

c7, c8 = st.columns(2)
with c7:
    temp = st.slider("🌡️ Temperature", 0.0, 1.0, 0.5)
with c8:
    atemp = st.slider("🤒 Feels Like Temperature", 0.0, 1.0, 0.5)

c9, c10 = st.columns(2)
with c9:
    hum = st.slider("💧 Humidity", 0.0, 1.0, 0.6)
with c10:
    windspeed = st.slider("🌬️ Wind Speed", 0.0, 1.0, 0.3)

hr = st.slider("⏰ Hour of Day", 0, 23, 12)

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------
def predict(hour, wd):
    row = pd.DataFrame([[
        season, mnth, hour, holiday, wd,
        workingday, weathersit, temp, atemp, hum, windspeed
    ]], columns=FEATURES)
    return int(model.predict(row)[0])

# --------------------------------------------------
# PREDICT
# --------------------------------------------------
if st.button("🚀 Predict Demand"):

    # ---------- HOURLY ----------
    if "Hourly" in mode:
        hours = list(range(7))
        preds = [predict(h, weekday) for h in hours]

        df = pd.DataFrame({
            "Hour": hours,
            "Predicted Rentals": preds
        })

        st.metric("⏱️ Total Rentals (7 Hours)", sum(preds))

    # ---------- DAILY ----------
    else:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily_values = []

        for wd in range(7):
            day_total = 0
            for h in range(24):
                day_total += predict(h, wd)
            daily_values.append(day_total)

        df = pd.DataFrame({
            "Day": day_names,
            "Predicted Rentals": daily_values
        })

        st.metric("📅 Total Rentals (7 Days)", sum(daily_values))

    # --------------------------------------------------
    # COMPACT TABLE
    # --------------------------------------------------
    st.subheader("📊 Prediction Table")

    tcol1, tcol2, tcol3 = st.columns([1, 2, 1])
    with tcol2:
        st.dataframe(df, height=320)

    # --------------------------------------------------
    # GRAPH WITH FEATURE NAMES
    # --------------------------------------------------
    fig, ax = plt.subplots()

    x_feature = df.columns[0]
    y_feature = df.columns[1]

    ax.plot(df[x_feature], df[y_feature], marker="o", linewidth=2)

    ax.set_xlabel(x_feature, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_feature, fontsize=12, fontweight="bold")
    ax.set_title("Predicted Bike Rental Demand Trend",
                 fontsize=14, fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.6)

    st.pyplot(fig)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    "<hr><p style='text-align:center;'>RideWise AI • Built for Real-World Decision Making</p>",
    unsafe_allow_html=True
)
