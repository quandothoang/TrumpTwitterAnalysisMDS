# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-02

"""Generates word clouds for positive and negative tweets.

This script:
1. Performs sentiment analysis if not already done
2. Trains a logistic regression classifier using weak labels
3. Extracts top positive and negative words based on coefficients
4. Creates word clouds for each sentiment
5. Documents any overlapping words (expected in political text)

Usage: python scripts/wordcloud_analysis.py [OPTIONS]

Options:
--processed_data    Path to processed CSV file (default: data/processed/trump_tweets_processed.csv)
--plot_to           Directory to save word clouds (default: results/figures)
--table_to          Directory to save word tables (default: results/tables)
"""

import click
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sentiment_utils import train_word_classifier, get_top_words, perform_sentiment_analysis
from src.visualization_utils import create_wordcloud


@click.command()
@click.option('--processed_data', type=str, required=False, default="data/processed/trump_tweets_processed.csv",
              help='Path to processed CSV file')
@click.option('--plot_to', type=str, required=False, default="results/figures", help='Directory to save word clouds')
@click.option('--table_to', type=str, required=False, default="results/tables", help='Directory to save word tables')
def main(processed_data: str, plot_to: str, table_to: str):
    """Generate word clouds for positive and negative sentiments."""

    os.makedirs(plot_to, exist_ok=True)
    os.makedirs(table_to, exist_ok=True)

    # Load data
    print(f"Loading data from: {processed_data}")
    tweets = pd.read_csv(processed_data, parse_dates=["Date & Time"], index_col="Date & Time")
    print(f"Loaded {tweets.shape[0]} tweets")

    # Perform sentiment analysis if not already done
    if "Sentiment" not in tweets.columns:
        print("Sentiment column not found - performing sentiment analysis...")
        tweets = perform_sentiment_analysis(tweets)
        print(f"Sentiment analysis complete: {tweets['Sentiment'].value_counts().to_dict()}")

    # Train classifier
    print("\nTraining sentiment classification model...")
    vectorizer, model, accuracy = train_word_classifier(tweets)

    # Save model metrics
    metrics_df = pd.DataFrame({
        'accuracy': [accuracy],
        'model_type': ['LogisticRegression'],
        'n_features': [len(vectorizer.get_feature_names_out())]
    })
    metrics_df.to_csv(os.path.join(table_to, "model_metrics.csv"), index=False)
    print(f"Model metrics saved to: {os.path.join(table_to, 'model_metrics.csv')}")

    # Extract top words
    print("\nExtracting top positive and negative words...")
    pos_words, neg_words, overlap = get_top_words(vectorizer, model, n_words=50)

    # Report overlapping words
    print(f"\nFound {len(overlap)} overlapping words between categories:")
    if overlap:
        print("(This is expected - political language is context-dependent)")
        for word in sorted(overlap):
            pos_coef = pos_words.get(word, 0)
            neg_coef = neg_words.get(word, 0)
            print(f"  '{word}': pos_coef={pos_coef:.4f}, neg_coef={neg_coef:.4f}")

    print("\nTop 10 Positive Words:")
    for i, (word, score) in enumerate(sorted(pos_words.items(), key=lambda x: x[1], reverse=True)[:10], 1):
        print(f"  {i:2d}. {word}: {score:.4f}")

    print("\nTop 10 Negative Words:")
    for i, (word, score) in enumerate(sorted(neg_words.items(), key=lambda x: x[1], reverse=True)[:10], 1):
        print(f"  {i:2d}. {word}: {score:.4f}")

    # Create word clouds
    print("\nCreating word clouds...")

    pos_fig = create_wordcloud(
        pos_words,
        "Words Associated with Positive Sentiment",
        colormap="Greens"
    )
    pos_path = os.path.join(plot_to, "wordcloud_positive.png")
    pos_fig.savefig(pos_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close(pos_fig)
    print(f"Saved: {pos_path}")

    neg_fig = create_wordcloud(
        neg_words,
        "Words Associated with Negative Sentiment",
        colormap="Reds"
    )
    neg_path = os.path.join(plot_to, "wordcloud_negative.png")
    neg_fig.savefig(neg_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close(neg_fig)
    print(f"Saved: {neg_path}")

    # Save word lists with overlap indicator
    pos_df = pd.DataFrame([
        {"word": w, "coefficient": s, "sentiment": "positive", "overlaps_with_negative": w in overlap}
        for w, s in sorted(pos_words.items(), key=lambda x: x[1], reverse=True)
    ])
    pos_df.to_csv(os.path.join(table_to, "top_positive_words.csv"), index=False)

    neg_df = pd.DataFrame([
        {"word": w, "coefficient": s, "sentiment": "negative", "overlaps_with_positive": w in overlap}
        for w, s in sorted(neg_words.items(), key=lambda x: x[1], reverse=True)
    ])
    neg_df.to_csv(os.path.join(table_to, "top_negative_words.csv"), index=False)

    # Save overlap analysis
    if overlap:
        overlap_df = pd.DataFrame([
            {
                "word": w,
                "positive_coefficient": pos_words.get(w, 0),
                "negative_coefficient": neg_words.get(w, 0),
                "interpretation": "Context-dependent word used in both positive and negative tweets"
            }
            for w in sorted(overlap)
        ])
        overlap_df.to_csv(os.path.join(table_to, "overlapping_words.csv"), index=False)
        print(f"Saved: {os.path.join(table_to, 'overlapping_words.csv')}")

    print("\nWord cloud analysis complete!")
    if overlap:
        print(f"Note: {len(overlap)} words appear in both categories - see Discussion in report")


if __name__ == "__main__":
    main()
