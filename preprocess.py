import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
# from textblob import TextBlob
from textblob import TextBlob

text = "This is a very good and amazing product"
blob = TextBlob(text)

print(blob.sentiment.polarity)
print(blob.sentiment.subjectivity)

import tldextract
import warnings
import joblib
import re
from datetime import datetime
from typing import Dict, Tuple, Optional
import logging

warnings.filterwarnings("ignore")

# =========================
# LOGGING SETUP
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(_name_)


class NewsDataPreprocessor:
    """
    Production-grade news data preprocessor with:
    - Proper train/test handling
    - Reusable transformers
    - Comprehensive feature engineering
    - Error handling and validation
    """
    
    def _init_(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(
            max_features=self.config.get('tfidf_features', 50),
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.source_encoder = LabelEncoder()
        self.domain_encoder = LabelEncoder()
        
        # Track fitted status
        self.is_fitted = False
        
        # Trusted domains for credibility scoring
        self.trusted_domains = {
            'bbc.com', 'reuters.com', 'apnews.com', 'nytimes.com',
            'theguardian.com', 'wsj.com', 'washingtonpost.com',
            'economist.com', 'ft.com', 'bloomberg.com'
        }
        
        # Clickbait indicators (expanded)
        self.clickbait_patterns = [
            'breaking', 'shocking', 'viral', 'secret', 'exposed',
            'unbelievable', "you won't believe", 'what happens next',
            'doctors hate', 'this one trick', 'blow your mind',
            'gone wrong', 'must see', 'will shock you'
        ]
        
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean input data"""
        required_cols = ['title', 'source', 'source_url', 'scraped_at']
        missing = set(required_cols) - set(df.columns)
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info(f"Initial data shape: {df.shape}")
        
        # Remove completely invalid rows
        initial_count = len(df)
        df = df[df['title'].notna() & (df['title'].str.strip() != '')]
        removed = initial_count - len(df)
        
        if removed > 0:
            logger.warning(f"Removed {removed} rows with missing/empty titles")
        
        return df.copy()
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced temporal feature engineering"""
        df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')
        
        # Basic temporal
        df['scrape_hour'] = df['scraped_at'].dt.hour
        df['scrape_day'] = df['scraped_at'].dt.day
        df['scrape_month'] = df['scraped_at'].dt.month
        df['scrape_weekday'] = df['scraped_at'].dt.weekday
        df['scrape_week_of_year'] = df['scraped_at'].dt.isocalendar().week
        
        # Cyclical encoding for time (preserves circular nature)
        df['hour_sin'] = np.sin(2 * np.pi * df['scrape_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['scrape_hour'] / 24)
        df['weekday_sin'] = np.sin(2 * np.pi * df['scrape_weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['scrape_weekday'] / 7)
        
        # Business insights
        df['is_weekend'] = df['scrape_weekday'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['scrape_hour'].between(9, 17).astype(int)
        df['is_prime_time'] = df['scrape_hour'].between(18, 22).astype(int)
        
        return df
    
    def create_url_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced URL and domain analysis"""
        def extract_domain(url):
            try:
                ext = tldextract.extract(str(url))
                return f"{ext.domain}.{ext.suffix}" if ext.domain else "unknown"
            except:
                return "unknown"
        
        df['domain'] = df['source_url'].apply(extract_domain)
        df['url_length'] = df['source_url'].astype(str).str.len()
        df['has_https'] = df['source_url'].astype(str).str.startswith('https').astype(int)
        
        # Domain credibility
        df['is_trusted_domain'] = df['domain'].isin(self.trusted_domains).astype(int)
        
        # URL structure analysis
        df['url_depth'] = df['source_url'].astype(str).str.count('/')
        df['has_subdomain'] = df['source_url'].astype(str).str.count('\.') > 1
        
        # Suspicious patterns
        suspicious_tlds = ['xyz', 'click', 'buzz', 'info', 'top', 'loan']
        df['has_suspicious_tld'] = (
            df['domain'].str.split('.').str[-1].isin(suspicious_tlds).astype(int)
        )
        
        # URL has numbers (often spam)
        df['url_has_numbers'] = df['source_url'].astype(str).str.contains(r'\d').astype(int)
        
        return df
    
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive text feature engineering"""
        df['title'] = df['title'].fillna('').astype(str)
        
        # Basic metrics
        df['char_count'] = df['title'].str.len()
        df['word_count'] = df['title'].str.split().str.len()
        df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1)
        
        # Punctuation analysis
        df['exclamation_count'] = df['title'].str.count('!')
        df['question_count'] = df['title'].str.count('\?')
        df['ellipsis_count'] = df['title'].str.count('\.\.\.')
        df['quote_count'] = df['title'].str.count('"') + df['title'].str.count("'")
        
        # Capitalization patterns
        df['capital_ratio'] = df['title'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
        )
        df['has_all_caps_word'] = df['title'].str.contains(r'\b[A-Z]{3,}\b').astype(int)
        df['title_case_ratio'] = df['title'].apply(
            lambda x: sum(1 for w in x.split() if w and w[0].isupper()) / max(len(x.split()), 1)
        )
        
        # Clickbait detection (enhanced)
        df['clickbait_score'] = df['title'].apply(
            lambda x: sum(pattern in x.lower() for pattern in self.clickbait_patterns)
        )
        df['has_number_in_title'] = df['title'].str.contains(r'\d').astype(int)
        df['starts_with_number'] = df['title'].str.match(r'^\d').astype(int)
        
        # Sensationalism indicators
        df['has_colon'] = df['title'].str.contains(':').astype(int)
        df['has_dash'] = df['title'].str.contains(' - ').astype(int)
        
        # Readability proxy
        df['complex_word_ratio'] = df['title'].apply(
            lambda x: sum(1 for w in x.split() if len(w) > 8) / max(len(x.split()), 1)
        )
        
        return df
    
    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sentiment and subjectivity analysis"""
        def safe_sentiment(text):
            try:
                blob = TextBlob(str(text))
                return blob.sentiment.polarity, blob.sentiment.subjectivity
            except:
                return 0.0, 0.0
        
        sentiments = df['title'].apply(safe_sentiment)
        df['sentiment_polarity'] = sentiments.apply(lambda x: x[0])
        df['sentiment_subjectivity'] = sentiments.apply(lambda x: x[1])
        
        # Sentiment categories
        df['is_positive'] = (df['sentiment_polarity'] > 0.1).astype(int)
        df['is_negative'] = (df['sentiment_polarity'] < -0.1).astype(int)
        df['is_neutral'] = ((df['sentiment_polarity'] >= -0.1) & 
                           (df['sentiment_polarity'] <= 0.1)).astype(int)
        df['is_subjective'] = (df['sentiment_subjectivity'] > 0.5).astype(int)
        
        return df
    
    def create_source_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Source-based features with aggregations"""
        df['source'] = df['source'].fillna('Unknown')
        
        # Source frequency (popularity proxy)
        source_counts = df['source'].value_counts()
        df['source_frequency'] = df['source'].map(source_counts)
        df['source_frequency_log'] = np.log1p(df['source_frequency'])
        
        # Source consistency metrics (if enough data)
        if len(df) > 100:
            source_stats = df.groupby('source').agg({
                'char_count': ['mean', 'std'],
                'sentiment_polarity': 'mean',
                'clickbait_score': 'mean'
            }).fillna(0)
            
            source_stats.columns = ['source_avg_length', 'source_length_std',
                                   'source_avg_sentiment', 'source_avg_clickbait']
            
            df = df.merge(source_stats, left_on='source', right_index=True, how='left')
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features for better ML performance"""
        # Length × Clickbait
        df['length_clickbait_interaction'] = df['char_count'] * df['clickbait_score']
        
        # Sentiment × Subjectivity
        df['sentiment_subjectivity_interaction'] = (
            df['sentiment_polarity'] * df['sentiment_subjectivity']
        )
        
        # Time × Domain credibility
        df['trusted_weekend_interaction'] = (
            df['is_trusted_domain'] * df['is_weekend']
        )
        
        # Caps × Exclamation (shouting indicator)
        df['shouting_score'] = df['capital_ratio'] * (df['exclamation_count'] + 1)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Fit transformers and transform training data
        Use this ONLY on training data
        """
        logger.info("Starting fit_transform on training data...")
        
        # Validate
        df = self.validate_data(df)
        
        # Create all features
        df = self.create_temporal_features(df)
        df = self.create_url_features(df)
        df = self.create_text_features(df)
        df = self.create_sentiment_features(df)
        df = self.create_source_features(df)
        df = self.create_interaction_features(df)
        
        # Encode categorical features
        df['source_encoded'] = self.source_encoder.fit_transform(df['source'])
        df['domain_encoded'] = self.domain_encoder.fit_transform(df['domain'])
        
        # TF-IDF (fit on training data only)
        tfidf_matrix = self.tfidf.fit_transform(df['title'])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()],
            index=df.index
        )
        df = pd.concat([df, tfidf_df], axis=1)
        
        # Scale numeric features (fit on training data only)
        numeric_cols = [
            'char_count', 'word_count', 'avg_word_length', 'capital_ratio',
            'exclamation_count', 'question_count', 'clickbait_score',
            'url_length', 'sentiment_polarity', 'sentiment_subjectivity',
            'source_frequency_log', 'url_depth', 'complex_word_ratio',
            'length_clickbait_interaction', 'sentiment_subjectivity_interaction',
            'shouting_score'
        ]
        
        # Only scale columns that exist
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        
        df[[f"{c}_scaled" for c in numeric_cols]] = self.scaler.fit_transform(
            df[numeric_cols].fillna(0)
        )
        
        self.is_fitted = True
        logger.info(f"Fit complete. Final shape: {df.shape}")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted transformers
        Use this on validation/test data
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_transform on training data first!")
        
        logger.info("Transforming new data using fitted transformers...")
        
        # Validate
        df = self.validate_data(df)
        
        # Create all features (same as training)
        df = self.create_temporal_features(df)
        df = self.create_url_features(df)
        df = self.create_text_features(df)
        df = self.create_sentiment_features(df)
        df = self.create_source_features(df)
        df = self.create_interaction_features(df)
        
        # Transform categorical using fitted encoders
        df['source_encoded'] = self.source_encoder.transform(
            df['source'].apply(lambda x: x if x in self.source_encoder.classes_ else 'Unknown')
        )
        df['domain_encoded'] = self.domain_encoder.transform(
            df['domain'].apply(lambda x: x if x in self.domain_encoder.classes_ else 'unknown')
        )
        
        # Transform TF-IDF
        tfidf_matrix = self.tfidf.transform(df['title'])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()],
            index=df.index
        )
        df = pd.concat([df, tfidf_df], axis=1)
        
        # Scale using fitted scaler
        numeric_cols = [
            'char_count', 'word_count', 'avg_word_length', 'capital_ratio',
            'exclamation_count', 'question_count', 'clickbait_score',
            'url_length', 'sentiment_polarity', 'sentiment_subjectivity',
            'source_frequency_log', 'url_depth', 'complex_word_ratio',
            'length_clickbait_interaction', 'sentiment_subjectivity_interaction',
            'shouting_score'
        ]
        
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        
        df[[f"{c}_scaled" for c in numeric_cols]] = self.scaler.transform(
            df[numeric_cols].fillna(0)
        )
        
        logger.info(f"Transform complete. Final shape: {df.shape}")
        
        return df
    
    def save(self, path: Path):
        """Save fitted transformers for production use"""
        if not self.is_fitted:
            raise ValueError("Must fit transformers before saving!")
        
        save_dict = {
            'scaler': self.scaler,
            'tfidf': self.tfidf,
            'source_encoder': self.source_encoder,
            'domain_encoder': self.domain_encoder,
            'config': self.config
        }
        
        joblib.dump(save_dict, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load(self, path: Path):
        """Load fitted transformers"""
        save_dict = joblib.load(path)
        
        self.scaler = save_dict['scaler']
        self.tfidf = save_dict['tfidf']
        self.source_encoder = save_dict['source_encoder']
        self.domain_encoder = save_dict['domain_encoder']
        self.config = save_dict['config']
        self.is_fitted = True
        
        logger.info(f"Preprocessor loaded from {path}")


# =========================
# USAGE EXAMPLE
# =========================
def main():
    """
    Example usage with proper train/test split
    """
    # Configuration
    input_path = Path("path/to/your/data.csv")  # CHANGE THIS
    output_dir = Path("path/to/output")  # CHANGE THIS
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(input_path)
    
    # Train/test split (80/20)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Initialize preprocessor
    config = {
        'tfidf_features': 50,
    }
    preprocessor = NewsDataPreprocessor(config)
    
    # Fit and transform training data
    train_processed = preprocessor.fit_transform(train_df)
    
    # Transform test data (using fitted transformers)
    test_processed = preprocessor.transform(test_df)
    
    # Save processed data
    train_processed.to_csv(output_dir / "train_processed.csv", index=False)
    test_processed.to_csv(output_dir / "test_processed.csv", index=False)
    
    # Save preprocessor for production
    preprocessor.save(output_dir / "preprocessor.pkl")
    
    logger.info("=" * 50)
    logger.info("FEATURE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total features created: {train_processed.shape[1]}")
    logger.info(f"\nSample features:\n{train_processed.columns.tolist()[:20]}")
    
    # Feature importance preparation
    feature_cols = [col for col in train_processed.columns 
                   if col not in ['title', 'source', 'source_url', 'scraped_at', 'domain']]
    
    logger.info(f"\nReady for ML with {len(feature_cols)} features")
    logger.info("=" * 50)


if _name_ == "_main_":
    main()