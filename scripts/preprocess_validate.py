# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-02

"""Preprocesses and validates raw Trump tweets data.

This script:
1. Parses raw CSV data (handles commas in tweet text)
2. Cleans and validates data using Pandera schema
3. Removes duplicates
4. Creates temporal and text features
5. Detects outliers in tweet length
6. Generates validation figures (correlation matrix, anomaly detection)
7. Saves processed data to data/processed/

Usage: python scripts/preprocess_validate.py [OPTIONS]

Options:
--raw_data    Path to raw CSV file (default: data/raw/realDonaldTrump_in_office.csv)
--write_to    Path to save processed CSV (default: data/processed/trump_tweets_processed.csv)
--plot_to     Directory to save validation figures (default: results/figures)
"""

import click
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import (
    parse_raw_csv,
    clean_tweets,
    create_features,
    detect_outliers_iqr
)


@click.command()
@click.option('--raw_data', type=str, required=False, default="data/raw/realDonaldTrump_in_office.csv", help='Path to raw CSV file')
@click.option('--write_to', type=str, required=False, default="data/processed/trump_tweets_processed.csv", help='Path to save processed CSV file')
@click.option('--plot_to', type=str, required=False, default="results/figures", help='Directory to save validation figures')
def main(raw_data: str, write_to: str, plot_to: str):
    """Preprocess and validate raw Trump tweets data."""

    # Create output directories if they don't exist
    output_dir = os.path.dirname(write_to)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_to, exist_ok=True)

    # Parse raw CSV data
    print("PARSING RAW DATA")
    print(f"Loading from: {raw_data}")
    df = parse_raw_csv(raw_data)
    print(f"Parsed {df.shape[0]} rows")

    # Clean and validate data
    print("CLEANING AND VALIDATING DATA")
    tweets = clean_tweets(df)
    print(f"Cleaned dataset: {tweets.shape[0]} tweets")

    # Print column data types
    print("\nColumn data types:")
    print(tweets.dtypes)

    # Set datetime index
    tweets = tweets.sort_index()

    # Create features
    print("CREATING FEATURES")
    tweets = create_features(tweets)
    print("Created features:")
    print("  - Temporal: hour, weekday, year, month, day, season, time_of_day")
    print("  - Text: length, avg_word_length, word_count, punctuation_count")

    # Check for outliers in tweet length
    print("OUTLIER DETECTION (Tweet Length)")
    print("\nLength descriptive statistics:")
    print(tweets["length"].describe())

    outlier_mask, lower_bound, upper_bound, outlier_count = detect_outliers_iqr(tweets["length"])
    print(f"\nIQR bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")
    print(f"Number of detected outliers: {outlier_count}")
    print("(Outliers are retained for analysis - they represent valid long/short tweets)")

    # Generate validation figures
    # 1. Correlation matrix heatmap
    print("Creating correlation matrix...")
    numeric_cols = ['length', 'hour', 'weekday', 'year', 'month', 'day',
                    'avg_word_length', 'word_count', 'punctuation_count']
    corr_matrix = tweets[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f',
                center=0, square=True, linewidths=0.5)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    corr_path = os.path.join(plot_to, "correlation_matrix.png")
    fig.savefig(corr_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {corr_path}")

    # 2. Outlier/Anomaly detection visualization (boxplots)
    print("Creating anomaly detection visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Tweet length boxplot
    sns.boxplot(x=tweets['length'], ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Tweet Length Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Character Count')

    # Word count boxplot
    sns.boxplot(x=tweets['word_count'], ax=axes[0, 1], color='seagreen')
    axes[0, 1].set_title('Word Count Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Number of Words')

    # Average word length boxplot
    sns.boxplot(x=tweets['avg_word_length'], ax=axes[1, 0], color='coral')
    axes[1, 0].set_title('Average Word Length Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Average Word Length')

    # Punctuation count boxplot
    sns.boxplot(x=tweets['punctuation_count'], ax=axes[1, 1], color='orchid')
    axes[1, 1].set_title('Punctuation Count Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Punctuation Marks')

    plt.suptitle('Anomaly Detection: Feature Distributions with Outliers',
                 fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    anomaly_path = os.path.join(plot_to, "anomaly_detection.png")
    fig.savefig(anomaly_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {anomaly_path}")

    # 3. Feature distributions histogram
    print("Creating feature distributions...")
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(tweets[col], ax=axes[i], kde=True, color='steelblue')
        axes[i].set_title(f'{col}', fontweight='bold')
        axes[i].set_xlabel('')

    plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    dist_path = os.path.join(plot_to, "feature_distributions.png")
    fig.savefig(dist_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {dist_path}")

    # Save processed data
    print("SAVING PROCESSED DATA")
    tweets.to_csv(write_to)
    print(f"Saved to: {write_to}")
    print(f"Final shape: {tweets.shape}")

    # Print summary statistics
    print("DATA SUMMARY")
    print(f"Date range: {tweets['Date & Time'].min()} to {tweets['Date & Time'].max()}")
    print(f"Total tweets: {len(tweets)}")
    print(f"\nTime of day distribution:")
    for tod, count in tweets["time_of_day"].value_counts().items():
        print(f"  {tod}: {count:,} ({count / len(tweets) * 100:.1f}%)")
    print(f"\nSeason distribution:")
    for s, count in tweets["season"].value_counts().items():
        print(f"  {s}: {count:,} ({count / len(tweets) * 100:.1f}%)")

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
