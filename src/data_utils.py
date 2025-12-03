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