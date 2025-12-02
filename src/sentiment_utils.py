# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-02

"""
Sentiment analysis utility functions for Trump tweets analysis.

This module provides functions to:
- Perform VADER sentiment analysis
- Train word-based sentiment classifiers
- Extract important words for positive/negative sentiment
"""

import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Word lists for weak labeling
POSITIVE_WORDS = set("""
good great amazing fantastic tremendous strong win winning beautiful success successful
happy proud respect love best positive incredible honored grateful huge strongest 
wonderful excellent terrific congratulations proud winner winners achievement 
thank thanks blessed blessing honor wonderful magnificent superb outstanding
""".split())

NEGATIVE_WORDS = set("""
bad terrible horrible weak fail failure disaster sad angry corrupt worst negative 
unfair hate disgrace stupid dishonest illegal failing failed witch hunt hoax
fake news enemy crooked liar lies lying pathetic loser disgraceful shameful
radical dangerous crime criminal criminals border crisis failing disaster
""".split())

# Stopwords to filter out
STOPWORDS = set("""
the a an and of to in is it this that for on with be as by are was were will 
from at have has but not or if so you your my our their they we i he she his 
her him them rt s all t just now amp more very about do what who people word 
should m realdonaldtrump u https http co www com today tomorrow yesterday
going get got much many new been would could can will than been being make
made way know see look well back over only other some such into than then
out up said also even after most through first last still take where when
""".split())


def score_to_label(score: float, pos_threshold: float = 0.05, neg_threshold: float = -0.05) -> str:
    """
    Convert VADER sentiment score to categorical label.
    
    Uses standard VADER thresholds as recommended in Hutto & Gilbert (2014).
    
    Parameters
    ----------
    score : float
        VADER compound score (-1 to 1)
    pos_threshold : float, default=0.05
        Threshold above which sentiment is positive
    neg_threshold : float, default=-0.05
        Threshold below which sentiment is negative
        
    Returns
    -------
    str
        'positive', 'negative', or 'neutral'
        
    Examples
    --------
    >>> score_to_label(0.5)
    'positive'
    >>> score_to_label(-0.3)
    'negative'
    >>> score_to_label(0.02)
    'neutral'
    """
    if score >= pos_threshold:
        return "positive"
    elif score <= neg_threshold:
        return "negative"
    else:
        return "neutral"


def perform_sentiment_analysis(tweets: pd.DataFrame) -> pd.DataFrame:
    """
    Perform VADER sentiment analysis on tweets.
    
    Parameters
    ----------
    tweets : pd.DataFrame
        DataFrame with 'Tweet Text' column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - sentiment_score: VADER compound score
        - Sentiment: categorical label (positive/negative/neutral)
        
    Examples
    --------
    >>> df = perform_sentiment_analysis(tweets)
    >>> 'Sentiment' in df.columns
    True
    """
    tweets = tweets.copy()
    sia = SentimentIntensityAnalyzer()
    
    tweets["Tweet Text"] = tweets["Tweet Text"].fillna("").astype(str)
    tweets["sentiment_score"] = tweets["Tweet Text"].apply(
        lambda t: sia.polarity_scores(t)["compound"]
    )
    tweets["Sentiment"] = tweets["sentiment_score"].apply(score_to_label)
    
    return tweets


def simple_tokenize(text: str) -> list:
    """
    Tokenize text by lowercasing, removing URLs and non-letters.
    
    Parameters
    ----------
    text : str
        Text to tokenize
        
    Returns
    -------
    list
        List of word tokens (length > 2)
        
    Examples
    --------
    >>> simple_tokenize("Hello World! https://t.co/abc")
    ['hello', 'world']
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return [w for w in text.split() if len(w) > 2]


def weak_label(text: str) -> str:
    """
    Create weak sentiment label using lexicon matching.
    
    Parameters
    ----------
    text : str
        Tweet text
        
    Returns
    -------
    str or None
        'positive' if more positive words, 'negative' if more negative words,
        None if tied or no matches
        
    Examples
    --------
    >>> weak_label("This is great and amazing!")
    'positive'
    >>> weak_label("This is terrible and horrible")
    'negative'
    """
    if not isinstance(text, str):
        return None
    tokens = simple_tokenize(text)
    pos_hits = sum(1 for w in tokens if w in POSITIVE_WORDS)
    neg_hits = sum(1 for w in tokens if w in NEGATIVE_WORDS)
    if pos_hits > neg_hits:
        return "positive"
    elif neg_hits > pos_hits:
        return "negative"
    else:
        return None


def train_word_classifier(tweets: pd.DataFrame, max_features: int = 5000, 
                          min_df: int = 5, test_size: float = 0.2) -> tuple:
    """
    Train a logistic regression classifier to identify sentiment-associated words.
    
    Uses weak labeling to create training data, then trains a classifier
    to identify words most predictive of positive/negative sentiment.
    
    Parameters
    ----------
    tweets : pd.DataFrame
        DataFrame with 'Tweet Text' column
    max_features : int, default=5000
        Maximum number of features for CountVectorizer
    min_df : int, default=5
        Minimum document frequency for features
    test_size : float, default=0.2
        Proportion of data for testing
        
    Returns
    -------
    tuple
        (vectorizer, model, accuracy) where:
        - vectorizer: fitted CountVectorizer
        - model: trained LogisticRegression model
        - accuracy: test set accuracy
        
    Examples
    --------
    >>> vectorizer, model, accuracy = train_word_classifier(tweets)
    >>> print(f"Accuracy: {accuracy:.2%}")
    Accuracy: 78.40%
    """
    tweets = tweets.copy()
    tweets["weak_label"] = tweets["Tweet Text"].apply(weak_label)
    
    labeled = tweets.dropna(subset=["weak_label"])
    print(f"Labeled tweets: {len(labeled)} out of {len(tweets)}")
    print(f"  Positive: {(labeled['weak_label'] == 'positive').sum()}")
    print(f"  Negative: {(labeled['weak_label'] == 'negative').sum()}")
    
    vectorizer = CountVectorizer(
        stop_words=list(STOPWORDS),
        max_features=max_features,
        min_df=min_df,
        max_df=0.95
    )
    
    X = vectorizer.fit_transform(labeled["Tweet Text"].fillna(""))
    y = (labeled["weak_label"] == "positive").astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    model = LogisticRegression(max_iter=500, random_state=42, C=1.0)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    return vectorizer, model, accuracy


def get_top_words(vectorizer, model, n_words: int = 50) -> tuple:
    """
    Extract top positive and negative words based on model coefficients.
    
    Parameters
    ----------
    vectorizer : CountVectorizer
        Fitted vectorizer
    model : LogisticRegression
        Trained model
    n_words : int, default=50
        Number of top words to extract for each category
        
    Returns
    -------
    tuple
        (pos_words, neg_words, overlap) where:
        - pos_words: dict of word -> coefficient for positive words
        - neg_words: dict of word -> abs(coefficient) for negative words
        - overlap: set of words appearing in both
        
    Examples
    --------
    >>> pos_words, neg_words, overlap = get_top_words(vectorizer, model)
    >>> print(f"Found {len(overlap)} overlapping words")
    """
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    
    sorted_indices = np.argsort(coefs)
    
    # Top positive words (highest coefficients)
    pos_indices = sorted_indices[-n_words:][::-1]
    pos_words = {feature_names[i]: coefs[i] for i in pos_indices}
    
    # Top negative words (most negative coefficients)
    neg_indices = sorted_indices[:n_words]
    neg_words = {feature_names[i]: abs(coefs[i]) for i in neg_indices}
    
    overlap = set(pos_words.keys()) & set(neg_words.keys())
    
    return pos_words, neg_words, overlap
