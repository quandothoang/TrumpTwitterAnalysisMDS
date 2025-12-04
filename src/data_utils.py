# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-02

# code format guidance and debugging provided by Claude Sonnet 4.5

"""Cleans the Trump tweets dataset from data/raw and saves to data/cleaned folder.

This script:
1. Reads raw CSV data from data/raw/
2. Parses and cleans CSV formatting issues (commas in tweet text, malformed rows)
3. Converts datetime columns to proper datetime format
4. Removes duplicate rows
5. Validates datatypes using Pandera schema
6. Saves cleaned data to data/cleaned/

Usage:
    python clean_trump_tweets.py --read_from=<path> --write_to=<path>

Example:
    python clean_trump_tweets.py 
        --read_from="../data/raw/realDonaldTrump_in_office.csv" 
        --write_to="../data/cleaned/realDonaldTrump_in_office_cleaned.csv"

Options:
    --read_from=<path>      Path to raw CSV file to clean
    --write_to=<path>      Path (including filename) to save cleaned CSV file

This module provides functions to:
- Download CSV data from URLs
- Parse tweets with proper handling of commas in text
- Create categorical and numerical features (time of day, season, etc.)
- Validate data with Pandera schemas
- Detect outliers using IQR method
"""

import click
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema
from pathlib import Path


def parse_raw_csv(file_path):
    """
    Parses raw CSV data and handles common formatting issues.
    
    This function handles cases where tweet text contains commas and other
    malformed CSV rows specific to the Trump tweets dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the raw CSV file
        
    Returns:
    --------
    pd.DataFrame
        Parsed dataframe with columns: ID, Time, Tweet URL, Tweet Text
    """
    # Read the raw file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    rows = []
    
    # Process each line with enumeration starting at 1
    for i, line in enumerate(lines, start=1):
        # Remove trailing newline and carriage return characters
        line = line.rstrip("\n\r")
        
        # Skip completely empty lines (lines with only whitespace)
        if not line.strip():
            continue
        
        # Split the CSV line by commas into a list
        parts = line.split(",")
        
        # Skip the header row (first row)
        if i == 1:
            continue
        
        # Skip malformed rows that have fewer than 4 columns
        if len(parts) < 4:
            continue
        
        # Extract individual columns and strip whitespace
        id_val = parts[0].strip()
        time_val = parts[1].strip()
        url_val = parts[2].strip()
        
        # Handle rows where tweet text contains commas
        # Rejoin parts from index 3 onwards with commas to preserve original text
        tweet_text = ",".join(parts[3:]).strip()
        
        # Add this row as a tuple to the rows list
        rows.append((id_val, time_val, url_val, tweet_text))
    
    # Create DataFrame with proper column names
    df = pd.DataFrame(rows, columns=["ID", "Time", "Tweet URL", "Tweet Text"])
    return df


def clean_tweets(df):
    """
    Cleans and transforms the tweets dataframe.
    
    Performs the following operations:
    1. Strips whitespace from column names
    2. Converts Time column to datetime format
    3. Drops unnecessary columns (ID, Tweet URL, Time)
    4. Removes duplicate rows
    5. Validates data types using Pandera schema
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe from parse_raw_csv()
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with validated datatypes
        
    Raises:
    -------
    pandera.errors.SchemaError
        If dataframe fails schema validation
    """
    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()
    
    # Convert Time column to datetime format
    # errors='coerce' converts invalid dates to NaT (Not a Time)
    df["Date & Time"] = pd.to_datetime(df["Time"], errors="coerce")
    
    # Drop unnecessary columns, keeping only Tweet Text and Date & Time
    df = df.drop(columns=["ID", "Tweet URL", "Time"])
    
    # Remove completely duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    
    if duplicates_removed > 0:
        click.secho(f"\n\nRemoved {duplicates_removed} duplicate rows", fg='blue', bold=True)
    else:
        click.secho("No duplicate rows found", fg='blue', bold=True)
    
    # Define schema for data validation
    schema = DataFrameSchema(
        {
            "Date & Time": Column(pa.DateTime, nullable=False, coerce=True),
            "Tweet Text": Column(pa.String, nullable=False),
        },
        checks=[pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")]
    )
    
    # Validate the dataframe against the schema
    # This will raise an error if validation fails
    df = schema.validate(df)

    click.secho("Data validation passed\n", fg='blue', bold=True)   
    
    return df


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5):
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    Parameters
    ----------
    series : pd.Series
        Numeric series to check for outliers
    multiplier : float, default=1.5
        IQR multiplier for bounds (1.5 is standard)
        
    Returns
    -------
    tuple
        (outlier_mask, lower_bound, upper_bound, outlier_count)
        
    Examples
    --------
    >>> mask, lower, upper, count = detect_outliers_iqr(tweets["length"])
    >>> print(f"Found {count} outliers outside [{lower}, {upper}]")
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_count = outlier_mask.sum()
    
    return outlier_mask, lower_bound, upper_bound, outlier_count


def season(date: pd.Timestamp):
    """
    This function returns the season based on the month.
    
    Parameters
    ----------
    date : pd.Timestamp
        The date we want to find the season from.
        
    Returns
    -------
    str
        Season: 'winter', 'spring', 'summer', or 'autumn'
        
    Examples
    --------
    >>> get_season(1)
    'winter'
    >>> get_season(7)
    'summer'
    """
    if 4 <= date.month <=6:
        return 'spring'
    elif 7 <= date.month <=9:
        return 'spring'
    elif 10 <= date.month <=12:
        return 'autumn'
    else:
        return 'winter'


def daytime(date: pd.Timestamp):
    """
    This function returns the time of day based on the hour.
    
    Parameters
    ----------
    date : pd.Timestamp
        The date we want to find the time of day from.
        
    Returns
    -------
    str
        Time of day: 'overnight' (0-8), 'daytime' (8-16), or 'evening' (16-24)
        
    Examples
    --------
    >>> get_time_of_day(3)
    'overnight'
    >>> get_time_of_day(14)
    'daytime'
    >>> get_time_of_day(20)
    'evening'
    """
    if pd.Timestamp('08:01').time() <= date.time() <= pd.Timestamp('16:00').time():
        return 'daytime'
    elif pd.Timestamp('16:01').time() <= date.time() <= pd.Timestamp('00:00').time():
        return 'evening'
    else:
        return 'overnight'
    
def avg_word_length(text: str):
    """
    This function finds the average word length in a text. 
    
    Parameters
    ----------
    text : str
        A text where each word is separated by spaces.
        
    Returns
    -------
    float
        The average word length rounded to 1 decimal point.
        
    Examples
    --------
    >>> avg_word_length('Donald Trump first presidency began in January 2017, and ended in January 2021.')
    5.2
    """
    average = 0
    for word in text.split() :
        average += len(word)
    return round(average/len(text.split()),1)

def punctuation_count(text):
    """
    This function finds the number of punctuation marks in the text. Here, punctuation marks are considered to be any non-numeric or whitespace character (this includes symbols like & or #).
    
    Parameters
    ----------
    text : str
        A text.
        
    Returns
    -------
    int
        The number of punctuation marks.
        
    Examples
    --------
    >>> punctuation_count('Donald Trump first presidency began in January 2017, and ended in January 2021.')
    2
    """
    count = 0
    for char in text:
        if not char.isalnum() and not char.isspace():
            count+=1
    return count


def create_features(tweets: pd.DataFrame):
    """
    This function creates categorical and numerical features from the existing features.
    
    Parameters
    ----------
    tweets : pd.DataFrame
        DataFrame with Date & Time and Tweet Text columns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional feature columns:
        - length: character count of tweet
        - hour, weekday, year, month, day: temporal components
        - season: winter/spring/summer/autumn
        - time_of_day: overnight/daytime/evening
        - avg_word_length: average word length in tweet
        - word_count: number of words
        - punctuation_count: count of punctuation marks
        
    Examples
    --------
    >>> df = create_features(tweets)
    >>> 'season' in df.columns
    True
    """
    tweets = tweets.reset_index().copy()
    
    # Text features
    tweets["length"] = tweets["Tweet Text"].str.len()
    
    # Temporal features
    tweets["hour"] = tweets['Date & Time'].dt.hour
    tweets["weekday"] = tweets['Date & Time'].dt.weekday
    tweets["year"] = tweets['Date & Time'].dt.year
    tweets["month"] = tweets['Date & Time'].dt.month
    tweets["day"] = tweets['Date & Time'].dt.day
    
    # Derived temporal features
    tweets["season"] = tweets["month"].apply(season)
    tweets["time_of_day"] = tweets["hour"].apply(daytime)
    
    # Additional text features
    tweets["avg_word_length"] = tweets["Tweet Text"].apply(avg_word_length) 
    tweets["word_count"] = tweets["Tweet Text"].apply(lambda x:len(x.split()))
    tweets["punctuation_count"] = tweets["Tweet Text"].apply(punctuation_count)
    
    return tweets


@click.command()
@click.option(
    '--read_from',
    default="../data/raw/realDonaldTrump_in_office.csv",
    help='Path to raw CSV file to clean',
    type=str
)
@click.option(
    '--write_to',
    default="../data/cleaned/realDonaldTrump_in_office_cleaned.csv",
    help='Path (including filename) to save cleaned CSV file',
    type=str
)
def clean_raw_data(read_from, write_to):
    """
    Cleans raw Trump tweets data and saves to cleaned directory.
    
    Reads raw CSV from read_from path, applies cleaning and validation,
    and saves the result to write_to path.
    """
    try:
        click.echo("=" * 70)
        click.echo("Trump Tweets Dataset - Data Cleaning Pipeline")
        click.echo("=" * 70)
        click.echo(f"Reading from: {read_from}")
        click.echo(f"Writing to: {write_to}")
        click.echo("=" * 70)
        
        # Parse raw CSV file 
        click.echo("\nParsing raw CSV data...", nl=False)
        df = parse_raw_csv(read_from)
        click.secho(f"...Successfully loaded {len(df)} rows!",fg='magenta', bold=True)

        # Clean and validate data 
        click.echo("\nCleaning and validating data...", nl=False)
        df = clean_tweets(df)
        click.secho(f"...Final dataset: {len(df)} rows",fg='magenta', bold=True)
        
        # Create the output directory 
        click.echo("\nCreating output directory structure...", nl=False)
        output_path = Path(write_to)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        click.secho("...Directory created!",fg='magenta', bold=True)
        
        # Save the cleaned data to csv 
        click.echo("\nSaving cleaned data to CSV...", nl=False)
        df.to_csv(output_path, index=False)
        click.secho("...File saved!",fg='magenta', bold=True)
       
        click.echo("\n" + "=" * 70)
        click.secho("Data cleaning pipeline complete!", fg='green', bold=True)
        click.echo(f"Saved to: {output_path.resolve()}")
        click.echo("=" * 70)
        
        # Print summary statistics
        click.secho("\nCleaned data summary:\n", fg='cyan', bold=True) 
        click.secho("  Shape: ", fg='blue', bold=True, nl=False)
        click.echo(f"{df.shape[0]} rows by {df.shape[1]} columns")
        click.secho("  Columns: ", fg='blue', bold=True, nl=False)
        click.echo(f"{', '.join(df.columns)}")
        click.secho("  Date range: ", fg='blue', bold=True, nl=False)
        click.echo(f"{df['Date & Time'].min()} to {df['Date & Time'].max()}")
    
        click.secho("\nAll tasks complete! Terminating script.\n", fg='cyan', bold=True)    
        
    except FileNotFoundError:
        click.secho(f"!File not found: {read_from}", fg='red', bold=True)
        raise click.Abort()
    except pd.errors.ParserError as e:
        click.secho(f"!CSV parsing error: {e}", fg='red', bold=True)
        raise click.Abort()
    except pa.errors.SchemaError as e:
        click.secho(f"!Data validation failed: {e}", fg='red', bold=True)
        raise click.Abort()
    except IOError as e:
        click.secho(f"!File I/O error: {e}", fg='red', bold=True)
        raise click.Abort()
    except Exception as e:
        click.secho(f"!Unexpected error: {e}", fg='red', bold=True)
        raise click.Abort()


if __name__ == "__main__":
    clean_raw_data()