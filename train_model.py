import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load preprocessed features with risk categories
df = pd.read_csv("old_preprocessed_data.csv")

# Drop non-feature columns
drop_cols = [
    "title", "link", "description", "pub_date", "source",
    "source_url", "extracted_at", "domain",
    "risk_score", "risk_category_Medium", "risk_category_Low"
]

# Only drop columns that exist
drop_cols = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=drop_cols + ["risk_category_High"])
y = df["risk_category_High"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/risk_detection_model.pkl")

print("MODEL TRAINING COMPLETE")
print("Model saved to models/risk_detection_model.pkl")