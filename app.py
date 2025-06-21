# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Google Play Store ML App", layout="wide")

# --- Functions ---

def clean_size(value):
    if 'M' in value:
        return float(value.replace('M', '')) * 1_000_000
    elif 'k' in value:
        return float(value.replace('k', '')) * 1_000
    elif value == 'Varies with device' or value == '0':
        return np.nan
    try:
        return float(value)
    except:
        return np.nan

def load_and_clean_data():
    df = pd.read_csv("googleplaystore.csv")
    df = df[df['Rating'].notnull()]
    df = df[df['Rating'] <= 5]

    df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True)
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
    df['Installs'].fillna(df['Installs'].median(), inplace=True)

    df['Price'] = df['Price'].str.replace('$', '', regex=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Price'].fillna(0, inplace=True)

    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    df['Reviews'].fillna(df['Reviews'].median(), inplace=True)

    df['Size'] = df['Size'].astype(str).apply(clean_size)
    df['Size'].fillna(df['Size'].median(), inplace=True)

    df.drop(['App', 'Last Updated', 'Current Ver', 'Android Ver'], axis=1, inplace=True)

    le = LabelEncoder()
    for col in ['Category', 'Content Rating', 'Genres', 'Type']:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

def train_and_save_model(df):
    X = df[['Category', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres']]
    y = df['Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    with open("app_rating_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return mse, model

def plot_rating_distribution(df):
    fig, ax = plt.subplots()
    sns.histplot(df['Rating'], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title("App Ratings Distribution")
    st.pyplot(fig)

def plot_category_vs_rating(df):
    avg_rating = df.groupby('Category')['Rating'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=avg_rating.index, y=avg_rating.values, palette="viridis", ax=ax)
    plt.xticks(rotation=90)
    ax.set_title("Average Rating per Category")
    st.pyplot(fig)

def plot_installs_vs_rating(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='Installs', y='Rating', hue='Category', alpha=0.6, ax=ax)
    ax.set_title("Installs vs Rating")
    st.pyplot(fig)

# --- UI ---
st.title("📱 Google Play Store ML Dashboard")

df = load_and_clean_data()

with st.expander("📊 View Raw Cleaned Dataset"):
    st.dataframe(df.head(20))

# Plots
st.subheader("🔍 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    plot_rating_distribution(df)

with col2:
    plot_category_vs_rating(df)

st.markdown("---")

plot_installs_vs_rating(df)

# Train and show metrics
st.subheader("🤖 Train Model")

if st.button("Train Random Forest Model"):
    mse, model = train_and_save_model(df)
    st.success(f"Model trained successfully! Mean Squared Error: {mse:.4f}")
    st.success("Model saved as `app_rating_model.pkl` ✅")

# Optional: Prediction input
st.markdown("---")
st.subheader("🎯 Try a Prediction")

with st.form("predict_form"):
    category = st.slider("Category (encoded)", 0, int(df['Category'].max()), 5)
    reviews = st.number_input("Reviews", value=5000)
    size = st.number_input("App Size (bytes)", value=10_000_000)
    installs = st.number_input("Installs", value=100000)
    app_type = st.selectbox("App Type", ['Free', 'Paid'])
    price = st.number_input("Price", value=0.0)
    content_rating = st.slider("Content Rating (encoded)", 0, int(df['Content Rating'].max()), 3)
    genres = st.slider("Genres (encoded)", 0, int(df['Genres'].max()), 10)

    submitted = st.form_submit_button("Predict Rating")

    if submitted:
        try:
            with open("app_rating_model.pkl", "rb") as f:
                model = pickle.load(f)

            app_type_encoded = 0 if app_type == 'Free' else 1

            input_data = pd.DataFrame([[
                category, reviews, size, installs, app_type_encoded,
                price, content_rating, genres
            ]], columns=['Category', 'Reviews', 'Size', 'Installs', 'Type',
                         'Price', 'Content Rating', 'Genres'])

            prediction = model.predict(input_data)[0]
            st.success(f"📈 Predicted App Rating: {prediction:.2f}")
        except FileNotFoundError:
            st.error("❌ Model file not found. Please train the model first.")

