import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ------------------ CONFIG ------------------
st.set_page_config(page_title="India Crime Dashboard", layout="wide")
st.markdown("""
<style>
footer {visibility: hidden;}
.stButton>button {background-color: #FF4B4B; color:white;}
.css-18e3th9 {padding: 1rem}
</style>
""", unsafe_allow_html=True)

# ------------------ LOGIN SYSTEM ------------------
USERS = {"admin": "1234", "guest": "pass"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login to Crime Dashboard")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user in USERS and USERS[user] == pwd:
            st.session_state.logged_in = True
            st.session_state.username = user
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ------------------ LOGOUT ------------------
st.sidebar.title(f"Welcome, {st.session_state.username}")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ------------------ LOAD MODELS ------------------
with open("models.pickle", "rb") as f:
    models = pickle.load(f)

with open("label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

# ------------------ LOAD DATASETS ------------------
def load_csv(name):
    try:
        return pd.read_csv(name)
    except:
        return pd.DataFrame()

datasets = {
    "Cyber Crime": load_csv("cyber_crime.csv"),
    "Kidnapping": load_csv("kidnapping.csv"),
    "Murder Motives": load_csv("murder_motives.csv"),
    "Rape Victims": load_csv("rape_victims.csv"),
    "Crime Against Children": load_csv("crime_against_children.csv"),
    "Juvenile Crime": load_csv("juvenile_crime.csv"),
    "Missing Children": load_csv("missing_traced_children.csv"),
    "Trafficking": load_csv("trafficing.csv"),
}

# ------------------ ML PREDICTOR ------------------
st.title("🧠 Crime in India - ML Predictor & Dashboard")

with st.expander("🔮 Predict Crimes in 2016"):
    states = label_encoder.classes_
    state = st.selectbox("📍 Select State", states)
    state_encoded = label_encoder.transform([state])[0]

    col1, col2, col3 = st.columns(3)
    crime_2014 = col1.number_input("Crime in 2014", 0, value=10000)
    crime_2015 = col2.number_input("Crime in 2015", 0, value=10000)
    crime_rate = col3.number_input("Crime Rate", 0.0, value=200.0)
    crime_share = st.number_input("Crime Share % (State)", 0.0, value=3.0)
    model_choice = st.selectbox("🤖 Select ML Model", list(models.keys()))

    if st.button("Predict Now"):
        input_data = np.array([[state_encoded, crime_2014, crime_2015, crime_share, crime_rate]])
        prediction = models[model_choice].predict(input_data)[0]
        st.success(f"🔍 Predicted crimes in 2016 for **{state}**: **{int(prediction):,}**")

st.markdown("---")

# ------------------ PLOT HELPERS ------------------
def plot_bar(df, x, y, title):
    fig = px.bar(df, x=x, y=y, title=title, color=y, height=350)
    fig.update_layout(margin=dict(t=40, l=20, r=20, b=20))
    return fig

def plot_pie(series, title):
    fig = px.pie(names=series.index, values=series.values, title=title, hole=0.4)
    return fig

# ------------------ DASHBOARD ------------------
st.header("📊 Crime Insights Dashboard")

# --- Row 1 ---
col1, col2 = st.columns(2)
with col1:
    cyber = datasets["Cyber Crime"]
    if "CYBER CRIME (2016)" in cyber.columns:
        filtered = cyber[cyber["STATE NAME"] == state]
        if not filtered.empty:
            st.plotly_chart(plot_bar(filtered, "STATE NAME", "CYBER CRIME (2016)", f"Cyber Crime in {state} (2016)"), use_container_width=True)

with col2:
    kid = datasets["Kidnapping"]
    gender_cols = ["KIDNAPPED & ABDUCTED (MALE)", "KIDNAPPED & ABDUCTED (FEMALE)"]
    state_col = "STATE/UT" if "STATE/UT" in kid.columns else "STATE NAME"
    if all(col in kid.columns for col in gender_cols):
        filtered = kid[kid[state_col] == state]
        if not filtered.empty:
            totals = filtered[gender_cols].sum()
            st.plotly_chart(plot_pie(totals, f"Kidnapping Gender Split in {state}"), use_container_width=True)

# --- Row 2 ---
col3, col4 = st.columns(2)
with col3:
    motives = datasets["Murder Motives"]
    cols = ["PERSONAL VENDETTA OR ENMITY", "PROPERTY DISPUTE", "GAIN"]
    state_col = "STATE/UT" if "STATE/UT" in motives.columns else "STATE NAME"
    if all(c in motives.columns for c in cols):
        filtered = motives[motives[state_col] == state]
        if not filtered.empty:
            motive_sum = filtered[cols].sum()
            st.plotly_chart(plot_pie(motive_sum, f"Murder Motives in {state}"), use_container_width=True)

with col4:
    rape = datasets["Rape Victims"]
    if not rape.empty:
        age_ranges = [
            "RAPE VICTIMS BELOW 6 YEARS",
            "RAPE VICTIMS 6-11 YEARS",
            "RAPE VICTIMS 12-15 YEARS",
            "RAPE VICTIMS 16-17 YEARS",
            "RAPE VICTIMS 18-29 YEARS",
            "RAPE VICTIMS 30-44 YEARS",
            "RAPE VICTIMS 45-59 YEARS",
            "RAPE VICTIMS 60 YEARS & ABOVE"
        ]
        if all(col in rape.columns for col in age_ranges):
            melted = pd.melt(rape, id_vars="STATE NAME", value_vars=age_ranges, var_name="Age Group", value_name="Victim Count")
            filtered = melted[melted["STATE NAME"] == state]
            st.plotly_chart(px.bar(filtered, x="Age Group", y="Victim Count", color="Age Group", title=f"Rape Victims by Age Group in {state} (2016)"), use_container_width=True)

# --- Row 3 ---
col5, col6 = st.columns(2)
with col5:
    child = datasets["Crime Against Children"]
    if "STATE/UT" in child.columns and "TOTAL CRIME HEADS" in child.columns:
        filtered = child[child["STATE/UT"] == state]
        if not filtered.empty:
            st.plotly_chart(plot_bar(filtered, "STATE/UT", "TOTAL CRIME HEADS", f"Crimes Against Children in {state}"), use_container_width=True)

with col6:
    juv = datasets["Juvenile Crime"]
    if "STATE/UT" in juv.columns and "TOTAL" in juv.columns:
        filtered = juv[juv["STATE/UT"] == state]
        if not filtered.empty:
            jv = filtered.groupby("STATE/UT")["TOTAL"].sum().reset_index()
            st.plotly_chart(plot_bar(jv, "STATE/UT", "TOTAL", f"Juvenile Crimes in {state}"), use_container_width=True)

# --- Row 4 ---
col7, col8 = st.columns(2)
with col7:
    missing = datasets["Missing Children"]
    if "STATE/UT" in missing.columns and "TOTAL MISSING CHILDREN" in missing.columns:
        filtered = missing[missing["STATE/UT"] == state]
        if not filtered.empty:
            st.plotly_chart(plot_bar(filtered, "STATE/UT", "TOTAL MISSING CHILDREN", f"Missing Children in {state}"), use_container_width=True)

with col8:
    traf = datasets["Trafficking"]
    if "STATE/UT" in traf.columns and "TOTAL TRAFFICKED" in traf.columns:
        filtered = traf[traf["STATE/UT"] == state]
        if not filtered.empty:
            st.plotly_chart(plot_bar(filtered, "STATE/UT", "TOTAL TRAFFICKED", f"Trafficking in {state}"), use_container_width=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("🧠 Built with ❤️ using Streamlit, Plotly & Machine Learning | India Crime Project © Ayushman Kar")
