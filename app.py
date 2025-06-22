# app.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load dataset
df = pd.read_csv("crimes.csv")
df = df.drop(columns=["Unnamed: 0"])

# Encode state names
le = LabelEncoder()
df["STATE NAME ENCODED"] = le.fit_transform(df["STATE NAME"])

# Features and target
X = df[["STATE NAME ENCODED", "CRIME (2014)", "CRIME (2015)", "CRIME PER SHARE (STATE)", "CRIME RATE"]]
y = df["CRIME (2016)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "SVR": SVR()
}

# Train and evaluate models
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}:\n  MSE: {mse:.2f}\n  R2 Score: {r2:.2f}\n")
    trained_models[name] = model

# Save all models
with open("models.pickle", "wb") as f:
    pickle.dump(trained_models, f)

# Save label encoder
with open("label_encoder.pickle", "wb") as f:
    pickle.dump(le, f)
