# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-02

"""Performs exploratory data analysis and generates visualizations.

This script:
1. Creates time-of-day frequency chart
2. Creates seasonal frequency chart
3. Saves summary tables

Usage: eda.py --processed_data=<processed_data> --plot_to=<plot_to> --table_to=<table_to>

Options:
--processed_data=<processed_data>   Path to the processed CSV file
--plot_to=<plot_to>                 Directory to save figures
--table_to=<table_to>               Directory to save tables
"""

import click
import pandas as pd
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.visualization_utils import create_time_of_day_chart, create_seasonal_chart


@click.command()
@click.option('--processed_data', type=str, required=False, default="../data/cleaned/realDonaldTrump_in_office_cleaned.csv",
    help='Path to raw CSV file to clean')
@click.option('--plot_to', type=str, required=False,  default="~/Documents/", help='Directory to save figures (default=Documents')
@click.option('--table_to', type=str, required=False,  default="~/Documents/", help='Directory to save tables (default=Documents')
def main(processed_data: str, plot_to: str, table_to: str):
    """Generate EDA visualizations and summary tables."""
    
    os.makedirs(plot_to, exist_ok=True)
    os.makedirs(table_to, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {processed_data}")
    tweets = pd.read_csv(processed_data, parse_dates=["Date & Time"], index_col="Date & Time")
    print(f"Loaded {tweets.shape[0]} tweets")
    
    # Create time of day chart
    print("Creating time of day chart...")
    time_chart = create_time_of_day_chart(tweets)
    time_chart_path = os.path.join(plot_to, "tweet_frequency_time_of_day.png")
    time_chart.save(time_chart_path, scale_factor=2)
    print(f"Saved: {time_chart_path}")
    
    # Create seasonal chart
    print("\nCreating seasonal chart...")
    season_chart = create_seasonal_chart(tweets)
    season_chart_path = os.path.join(plot_to, "tweet_frequency_season.png")
    season_chart.save(season_chart_path, scale_factor=2)
    print(f"Saved: {season_chart_path}")
    
    # Create summary tables
    print("Creating summary tables...")

    # Time of day summary
    total = len(tweets)
    time_df = tweets["time_of_day"].value_counts().reset_index()
    time_df.columns = ["Time of Day", "Count"]
    time_df["Percentage"] = (time_df["Count"] / total * 100).round(1)
    time_df["Time Range"] = time_df["Time of Day"].map({
        "daytime": "8:01am – 4:00pm",
        "evening": "4:01pm – 12:00am", 
        "overnight": "12:01am – 8:00am"
    })
    time_df = time_df[["Time of Day", "Time Range", "Count", "Percentage"]]
    time_df.to_csv(os.path.join(table_to, "time_of_day_summary.csv"), index=False)
    print(f"Saved: {os.path.join(table_to, 'time_of_day_summary.csv')}")
    
    # Season summary
    season_df = tweets["season"].value_counts().reset_index()
    season_df.columns = ["Season", "Count"]
    season_df["Season"] = season_df["Season"].str.capitalize()
    season_df["Percentage"] = (season_df["Count"] / total * 100).round(1)
    season_df["Date Range"] = season_df["Season"].map({
        "Spring": "Apr 1 – Jun 30",
        "Summer": "Jul 1 – Sep 30",
        "Autumn": "Oct 1 – Dec 31",
        "Winter": "Jan 1 – Mar 31"
    })
    season_df = season_df[["Season", "Date Range", "Count", "Percentage"]]
    season_df.to_csv(os.path.join(table_to, "season_summary.csv"), index=False)
    print(f"Saved: {os.path.join(table_to, 'season_summary.csv')}")
    
    # Print summaries
    print("TIME OF DAY SUMMARY")
    print(time_df.to_string(index=False))
    
    print("SEASONAL SUMMARY")
    print(season_df.to_string(index=False))
    
    print("\nEDA complete!")


if __name__ == "__main__":
    main()
