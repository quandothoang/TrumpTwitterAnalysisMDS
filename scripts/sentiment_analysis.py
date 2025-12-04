# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-04
# Script: VADER sentiment analysis on Trump tweets (Method 1)

"""
Performs sentiment analysis on Trump tweets using the VADER model.

This script:
1. Applies VADER sentiment analysis to classify tweets
2. Creates sentiment distribution chart
3. Creates sentiment over time chart
4. Saves sentiment-analyzed data and summary tables

Usage:
    python sentiment_vader_method1.py \
        --processed_data data/processed/trump_tweets.csv \
        --write_to data/processed/trump_tweets_with_sentiment.csv \
        --plot_to results/figures \
        --table_to results/tables
"""

import os
import click
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def score_to_label(score: float,
                   pos_threshold: float = 0.05,
                   neg_threshold: float = -0.05) -> str:
    """Map VADER compound score to 'positive' / 'neutral' / 'negative'."""
    if score >= pos_threshold:
        return "positive"
    elif score <= neg_threshold:
        return "negative"
    else:
        return "neutral"


def add_vader_sentiment(tweets: pd.DataFrame,
                        text_col: str = "Tweet Text") -> pd.DataFrame:
    """Add VADER sentiment score and label columns to a tweets DataFrame."""
    sia = SentimentIntensityAnalyzer()

    tweets_out = tweets.copy()
    tweets_out[text_col] = tweets_out[text_col].fillna("").astype(str)

    tweets_out["sentiment_score"] = tweets_out[text_col].apply(
        lambda t: sia.polarity_scores(t)["compound"]
    )
    tweets_out["Sentiment"] = tweets_out["sentiment_score"].apply(score_to_label)

    return tweets_out


def compute_sentiment_counts(tweets_with_sentiment: pd.DataFrame) -> pd.DataFrame:
    """Return a table with counts of tweets per sentiment."""
    sentiment_counts = (
        tweets_with_sentiment
        .groupby("Sentiment")
        .size()
        .reset_index(name="Count")
    )
    # ensure consistent order: positive, neutral, negative
    sentiment_counts["Sentiment"] = pd.Categorical(
        sentiment_counts["Sentiment"],
        categories=["positive", "neutral", "negative"],
        ordered=True,
    )
    sentiment_counts = sentiment_counts.sort_values("Sentiment")
    return sentiment_counts


def create_sentiment_chart(sentiment_counts: pd.DataFrame) -> alt.Chart:
    """Create an Altair bar chart from the sentiment counts table."""
    chart = (
        alt.Chart(sentiment_counts)
        .mark_bar()
        .encode(
            x=alt.X("Sentiment:N",
                    sort=["positive", "neutral", "negative"]),
            y=alt.Y("Count:Q"),
            tooltip=["Sentiment", "Count"],
        )
        .properties(
            title="Number of Positive, Neutral, and Negative Tweets"
        )
    )
    return chart


# ---------- command line interface ----------

@click.command()
@click.option(
    "--processed_data",
    type=str,
    required=True,
    help="Path to processed CSV file containing a 'Tweet Text' column.",
)
@click.option(
    "--write_to",
    type=str,
    required=True,
    help="Path to save CSV with sentiment_score and Sentiment columns.",
)
@click.option(
    "--plot_to",
    type=str,
    required=True,
    help="Directory to save sentiment bar chart PNG.",
)
@click.option(
    "--table_to",
    type=str,
    required=True,
    help="Directory to save sentiment counts table CSV.",
)
def main(processed_data: str, write_to: str, plot_to: str, table_to: str) -> None:
    """Run VADER sentiment analysis and save outputs."""
    # make sure output directories exist
    os.makedirs(os.path.dirname(write_to), exist_ok=True)
    os.makedirs(plot_to, exist_ok=True)
    os.makedirs(table_to, exist_ok=True)

    # 1. load data
    print(f"Loading data from: {processed_data}")
    tweets = pd.read_csv(processed_data)
    print(f"Loaded {tweets.shape[0]} tweets")

    # 2–3. add sentiment columns
    print("\nRunning VADER sentiment analysis (Method 1)...")
    tweets_with_sentiment = add_vader_sentiment(tweets)

    # 4. save tweets with sentiment
    tweets_with_sentiment.to_csv(write_to, index=False)
    print(f"Saved tweets with sentiment to: {write_to}")

    # 5. sentiment counts table
    sentiment_counts = compute_sentiment_counts(tweets_with_sentiment)
    table_path = os.path.join(table_to, "sentiment_counts.csv")
    sentiment_counts.to_csv(table_path, index=False)
    print(f"Saved sentiment counts table to: {table_path}")

    # 6. bar chart
    sentiment_chart = create_sentiment_chart(sentiment_counts)
    plot_path = os.path.join(plot_to, "sentiment_counts.png")
    sentiment_chart.save(plot_path, scale_factor=2)
    print(f"Saved sentiment chart to: {plot_path}")

    print("\nSentiment analysis (Method 1) complete!")


if __name__ == "__main__":
    main()
