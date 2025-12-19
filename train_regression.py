import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load preprocessed features with risk_score
df = pd.read_csv("old_preprocessed_data.csv")

# Drop non-feature columns
drops_cols = [
    "title", "link", "description", "pub_date", "source",
    "source_url", "extracted_at", "domain",
    "risk_category_High", "risk_category_Medium", "risk_category_Low"
]

# Only drop columns that exist
drops_cols = [col for col in drops_cols if col in df.columns]

X = df.drop(columns=drops_cols + ["risk_score"])
y = df["risk_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/risk_score_model.pkl")
print("MODEL TRAINING COMPLETE")