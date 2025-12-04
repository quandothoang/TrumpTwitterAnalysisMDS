# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-02

"""Preprocesses and validates raw Trump tweets data.

This script:
1. Parses raw CSV data (handles commas in tweet text)
2. Cleans and validates data using Pandera schema
3. Removes duplicates
4. Creates temporal and text features
5. Detects outliers in tweet length
6. Saves processed data to data/processed/

Usage: python scripts/preprocess_validate.py [OPTIONS]

Options:
--raw_data    Path to raw CSV file (default: data/raw/realDonaldTrump_in_office.csv)
--write_to    Path to save processed CSV (default: data/processed/trump_tweets_processed.csv)
"""

import click
import os
import sys

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
def main(raw_data: str, write_to: str):
    """Preprocess and validate raw Trump tweets data."""

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(write_to)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
    tweets = tweets.set_index("Date & Time")
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
