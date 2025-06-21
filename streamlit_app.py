import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="📱 Google Play ML App", layout="wide")

# -------------------- DATA CLEANING ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("googleplaystore.csv")
    df = df[df['Rating'].notnull() & (df['Rating'] <= 5)]

    df['Installs'] = pd.to_numeric(df['Installs'].str.replace('[+,]', '', regex=True), errors='coerce')
    df['Installs'].fillna(df['Installs'].median(), inplace=True)

    df['Price'] = pd.to_numeric(df['Price'].str.replace('$', '', regex=True), errors='coerce')
    df['Price'] = df['Price'].fillna(0)

    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    df['Reviews'] = df['Reviews'].fillna(df['Reviews'].median() if not df['Reviews'].dropna().empty else 1000)

    def clean_size(val):
        if 'M' in val:
            return float(val.replace('M', '')) * 1_000_000
        elif 'k' in val:
            return float(val.replace('k', '')) * 1_000
        else:
            return np.nan

    df['Size'] = df['Size'].astype(str).apply(clean_size)
    df['Size'].fillna(df['Size'].median(), inplace=True)

    df.drop(['App', 'Last Updated', 'Current Ver', 'Android Ver'], axis=1, inplace=True)

    le = LabelEncoder()
    for col in ['Category', 'Content Rating', 'Genres', 'Type']:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

# -------------------- TRAINING & EVALUATION ------------------------
def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestRegressor(),
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor()
    }
    trained = {name: model.fit(X_train, y_train) for name, model in models.items()}
    return trained

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "MSE": mean_squared_error(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "R²": r2_score(y_test, preds)
    }

# -------------------- DASHBOARD ------------------------
def page_dashboard(df):
    st.markdown("""
        <div style="background-color:#e6f2ff;padding:20px;border-radius:10px;margin-bottom:20px">
            <h1 style="color:#1e40af;text-align:center;">📱 Google Play Store ML Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("🎛️ Controls")
    target_choice = st.sidebar.radio("Prediction Target", ["Rating", "Installs"])
    model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Linear Regression", "Gradient Boosting"])

    st.sidebar.subheader("Input Features")
    input_data = {
        "Category": st.sidebar.slider("Category", 0, int(df['Category'].max()), 5),
        "Reviews": st.sidebar.number_input("Reviews", value=5000),
        "Size": st.sidebar.number_input("Size (bytes)", value=10_000_000),
        "Installs": st.sidebar.number_input("Installs", value=100_000),
        "Type": st.sidebar.selectbox("Type", ['Free', 'Paid']),
        "Price": st.sidebar.number_input("Price", value=0.0),
        "Content Rating": st.sidebar.slider("Content Rating", 0, int(df['Content Rating'].max()), 3),
        "Genres": st.sidebar.slider("Genres", 0, int(df['Genres'].max()), 10)
    }

    X = df[['Category', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres']]
    y = df[target_choice]
    X['Type'] = X['Type'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = train_models(X_train, y_train)
    selected_model = models[model_choice]
    metrics = evaluate_model(selected_model, X_test, y_test)

    input_df = pd.DataFrame([[input_data['Category'], input_data['Reviews'], input_data['Size'],
                              input_data['Installs'], 0 if input_data['Type'] == 'Free' else 1,
                              input_data['Price'], input_data['Content Rating'], input_data['Genres']]],
                            columns=X.columns)

    prediction = selected_model.predict(input_df)[0]
    st.success(f"🎯 Predicted {target_choice}: {prediction:.2f}")
    st.json(metrics)

    st.markdown("### 📊 Visualizations")
    viz_option = st.selectbox("Choose plot", ["Rating Distribution", "Category vs Rating", "Installs vs Rating"])

    if viz_option == "Rating Distribution":
        fig = plt.figure(figsize=(6, 4))
        sns.histplot(df['Rating'], bins=20, kde=True, color='skyblue')
        st.pyplot(fig)
    elif viz_option == "Category vs Rating":
        fig = plt.figure(figsize=(8, 4))
        avg_rating = df.groupby('Category')['Rating'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_rating.index, y=avg_rating.values, palette="viridis")
        plt.xticks(rotation=90)
        st.pyplot(fig)
    elif viz_option == "Installs vs Rating":
        fig = plt.figure(figsize=(7, 4))
        sns.scatterplot(data=df, x='Installs', y='Rating', hue='Category', alpha=0.6)
        st.pyplot(fig)

# -------------------- TOP APPS ------------------------
def page_top_apps(df):
    st.title("🔥 Top Apps Overview")
    search_term = st.text_input("🔍 Search by Category")
    if search_term:
        filtered = df[df['Category'].astype(str).str.contains(search_term, case=False)]
        st.dataframe(filtered[['Category', 'Rating', 'Reviews', 'Installs']].head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⭐ Top Rated Apps")
        top_rated = df.sort_values(by='Rating', ascending=False).head(10)
        st.dataframe(top_rated[['Category', 'Rating', 'Reviews', 'Installs']])
    with col2:
        st.subheader("📈 Trending Apps")
        trending = df.sort_values(by='Installs', ascending=False).head(10)
        st.dataframe(trending[['Category', 'Rating', 'Reviews', 'Installs']])

# -------------------- HOME ------------------------
def page_home():
    st.markdown("""
        <div style="background-color:#e6f2ff;padding:30px;border-radius:10px;margin-bottom:20px">
            <h1 style="text-align:center; color:green;">🏠 Welcome to Google Play ML App</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color:blue; font-size:18px;">Use the sidebar to navigate and explore features.</p>', unsafe_allow_html=True)


# -------------------- MAIN ------------------------
df = load_data()
page = st.sidebar.radio("📂 Pages", ["Home", "ML Dashboard", "Top Apps"])
if page == "Home":
    page_home()
elif page == "ML Dashboard":
    page_dashboard(df)
elif page == "Top Apps":
    page_top_apps(df)
