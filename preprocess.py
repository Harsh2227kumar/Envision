import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent


# =========================
# LOAD DATA
# =========================
def load_data(filepath):
    path = Path(filepath)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    print("Data Loaded:", df.shape)
    return df


# =========================
# PREPROCESSING
# =========================
def preprocess_data(df):
    # Drop critical nulls
    df.dropna(subset=["title", "source", "domain"], inplace=True)
    df.fillna(0, inplace=True)

    # Date conversion
    df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce", dayfirst=True)
    df["extracted_at"] = pd.to_datetime(df["extracted_at"], errors="coerce", dayfirst=True)

    # Boolean columns
    bool_cols = [
        "is_credible_source",
        "has_suspicious_tld",
        "has_suspicious_pattern",
        "has_excessive_caps",
        "has_clickbait_pattern",
        "has_unattributed_claims",
        "has_citation",
        "has_statistics",
    ]

    for col in bool_cols:
        df[col] = df[col].astype(str).str.upper().map(
            {"TRUE": 1, "FALSE": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)

    # Feature engineering
    df["risk_category"] = pd.cut(
        df["risk_score"],
        bins=[-1, 0.3, 0.6, 1],
        labels=["Low", "Medium", "High"]
    )

    df["content_intensity"] = (
        df["emotional_score"] +
        df["sensational_count"] +
        df["bias_count"]
    )

    df["title_length"] = df["title"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len()

    # Time difference
    df["time_to_extract_hrs"] = (
        (df["extracted_at"] - df["pub_date"]).dt.total_seconds() / 3600
    )

    df.drop_duplicates(subset=["title", "source"], inplace=True)

    print("Preprocessing completed:", df.shape)
    return df


# =========================
# EDA
# =========================
def perform_eda(df):
    eda_dir = PROJECT_DIR / "eda_outputs"
    eda_dir.mkdir(exist_ok=True)

    sns.set(style="whitegrid")

    plt.figure()
    sns.countplot(x="is_credible_source", data=df)
    plt.title("Fake vs Real News")
    plt.savefig(eda_dir / "fake_real.png")
    plt.close()

    plt.figure()
    sns.histplot(df["risk_score"], kde=True)
    plt.title("Risk Score Distribution")
    plt.savefig(eda_dir / "risk_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(np.number).corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(eda_dir / "correlation.png")
    plt.close()

    print("EDA completed")


# =========================
# OUTLIER HANDLING (IQR)
# =========================
def handle_outliers(df):
    numeric_cols = df.select_dtypes(np.number).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    print("Outliers handled")
    return df


# =========================
# ENCODING
# =========================
def encode_features(df):
    df = pd.get_dummies(df, columns=["risk_category"], drop_first=True)

    top_domains = df["domain"].value_counts().head(8).index
    df["domain_group"] = df["domain"].apply(
        lambda x: x if x in top_domains else "Other"
    )
    df = pd.get_dummies(df, columns=["domain_group"])

    print("Encoding completed")
    return df


# =========================
# NORMALIZATION
# =========================
def normalize_features(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(np.number).columns

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("Normalization completed")
    return df


# =========================
# SAVE DATA
# =========================
def save_data(df, filename):
    path = PROJECT_DIR / filename
    df.to_csv(path, index=False)
    print("Saved:", path)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(PROJECT_DIR / "fake_news_analysis.csv"),
        help="Input CSV file"
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_DIR / "preprocessed_data.csv"),
        help="Output CSV file"
    )

    args = parser.parse_args()

    df = load_data(args.input)
    df = preprocess_data(df)
    perform_eda(df)
    df = handle_outliers(df)
    df = encode_features(df)
    df = normalize_features(df)
    save_data(df, args.output)