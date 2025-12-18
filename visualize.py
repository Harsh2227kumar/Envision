import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent

# Basic viz defaults
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


def load_data(filepath):
    path = Path(filepath)
    if not path.is_absolute():
        path = PROJECT_DIR / path

    # Fallback: if default CSV doesn't exist, try alternate .ccsv or .csv
    candidate_paths = [path]
    if path.suffix.lower() == ".csv":
        candidate_paths.append(path.with_suffix(".ccsv"))
    elif path.suffix.lower() == ".ccsv":
        candidate_paths.append(path.with_suffix(".csv"))

    for p in candidate_paths:
        if p.exists():
            df = pd.read_csv(p)
            print("Dataset Loaded:", df.shape, "from", p)
            return df

    raise FileNotFoundError(f"Input file not found: {candidate_paths[0]}")
    print("Dataset Loaded:", df.shape)
    return df


def ensure_features(df):
    df.fillna(0, inplace=True)

    bool_cols = [
        "is_credible_source",
        "has_clickbait_pattern",
        "has_excessive_caps",
        "has_citation",
        "has_statistics",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.upper()
                .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
            )
            df[col] = df[col].fillna(0).astype(int)

    # If risk_category is missing (e.g., using raw data), derive it from risk_score.
    if "risk_category" not in df.columns and "risk_score" in df.columns:
        df["risk_category"] = df["risk_score"].apply(
            lambda x: "High Risk" if x >= 0.6 else "Medium Risk" if x >= 0.3 else "Low Risk"
        )
    return df


def generate_visualizations(df):
    # Fake vs Real Distribution
    if "is_credible_source" in df.columns:
        sns.countplot(data=df, x="is_credible_source")
        plt.title("Fake vs Real News Distribution")
        plt.xlabel("0 = Fake | 1 = Real")
        plt.ylabel("Article Count")
        plt.tight_layout()
        plt.show()

    # Credibility Score Comparison
    if {"is_credible_source", "credibility_score"}.issubset(df.columns):
        sns.boxplot(data=df, x="is_credible_source", y="credibility_score")
        plt.title("Credibility Score: Fake vs Real")
        plt.xlabel("0 = Fake | 1 = Real")
        plt.tight_layout()
        plt.show()

    # Emotional Score Analysis
    if {"is_credible_source", "emotional_score"}.issubset(df.columns):
        sns.violinplot(data=df, x="is_credible_source", y="emotional_score")
        plt.title("Emotional Manipulation in Fake vs Real News")
        plt.xlabel("0 = Fake | 1 = Real")
        plt.tight_layout()
        plt.show()

    # Clickbait Pattern Analysis
    if {"has_clickbait_pattern", "is_credible_source"}.issubset(df.columns):
        sns.countplot(data=df, x="has_clickbait_pattern", hue="is_credible_source")
        plt.title("Clickbait Usage Comparison")
        plt.xlabel("Clickbait Present (0 = No, 1 = Yes)")
        plt.ylabel("Count")
        plt.legend(title="News Type", labels=["Fake", "Real"])
        plt.tight_layout()
        plt.show()

    # Capitalization Abuse
    if {"is_credible_source", "caps_word_count"}.issubset(df.columns):
        sns.boxplot(data=df, x="is_credible_source", y="caps_word_count")
        plt.title("Capitalized Word Usage")
        plt.xlabel("0 = Fake | 1 = Real")
        plt.tight_layout()
        plt.show()

    # Risk Score vs Credibility
    if {"credibility_score", "risk_score", "is_credible_source"}.issubset(df.columns):
        sns.scatterplot(
            data=df,
            x="credibility_score",
            y="risk_score",
            hue="is_credible_source",
        )
        plt.title("Risk Score vs Credibility Score")
        plt.xlabel("Credibility Score")
        plt.ylabel("Risk Score")
        plt.legend(title="News Type", labels=["Fake", "Real"])
        plt.tight_layout()
        plt.show()

    # Citation Usage Comparison
    if {"has_citation", "is_credible_source"}.issubset(df.columns):
        sns.countplot(data=df, x="has_citation", hue="is_credible_source")
        plt.title("Citation Usage in Fake vs Real News")
        plt.xlabel("Has Citation (0 = No, 1 = Yes)")
        plt.ylabel("Count")
        plt.legend(title="News Type", labels=["Fake", "Real"])
        plt.tight_layout()
        plt.show()

    # Domain Risk Analysis (Top 10)
    if {"domain", "risk_score"}.issubset(df.columns):
        top_domains = (
            df.groupby("domain")["risk_score"].mean().sort_values(ascending=False).head(10)
        )
        top_domains.plot(kind="bar")
        plt.title("Top 10 Domains by Average Risk Score")
        plt.xlabel("Domain")
        plt.ylabel("Average Risk Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Risk Category Distribution
    if "risk_category" in df.columns:
        plt.figure()
        df["risk_category"].value_counts().plot(kind="bar")
        plt.title("Risk Category Distribution")
        plt.xlabel("Risk Category")
        plt.ylabel("Number of Articles")
        plt.tight_layout()
        plt.show()

    # Credibility vs Emotional Score
    if {"credibility_score", "emotional_score"}.issubset(df.columns):
        plt.figure()
        plt.scatter(df["credibility_score"], df["emotional_score"])
        plt.title("Credibility vs Emotional Score")
        plt.xlabel("Credibility Score")
        plt.ylabel("Emotional Score")
        plt.tight_layout()
        plt.show()

    print("All visualizations generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize fake news dataset (expects preprocessed CSV)."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_DIR / "preprocessed_data.csv"),
        help="Path to the preprocessed CSV file.",
    )

    args = parser.parse_args()

    df = load_data(args.input)
    df = ensure_features(df)
    generate_visualizations(df)
