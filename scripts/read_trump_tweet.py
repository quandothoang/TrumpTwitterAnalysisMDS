# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-02

"""Downloads the Trump tweets dataset from URL and saves to data/raw folder. Bug fixing credited to Claude Opus 4.5

This script ensures reproducibility by:
1. Downloading data programmatically from the source URL (in case the URL is changed)
2. Saving an exact copy to data/raw/ for archival purposes
3. The saved copy ensures analysis remains reproducible even if URL becomes unavailable

Usage: download_data.py --url=<url> --write_to=<write_to>

Options:
--url=<url>             URL to the raw CSV file
--write_to=<write_to>   Path (including filename) to save the CSV file
"""


import click
import os
import sys

from src.data_utils import download_and_parse_csv


@click.command()
@click.option('--url', type=str, required=True, help='URL to the raw CSV file')
@click.option('--write_to', type=str, required=True, help='Path to save the CSV file')
def main(url: str, write_to: str):
    """
    Download Trump tweets data from URL and save to data/raw folder.

    This ensures the analysis is fully reproducible - the data is downloaded
    programmatically and saved locally, so even if the source URL becomes
    unavailable, the exact data used in the analysis is preserved.
    """

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(write_to)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

    # Download and parse data from URL
    print("." * 60)
    print("DOWNLOADING DATA FROM URL")
    tweets = download_and_parse_csv(url)

    print(f"\nParsed {tweets.shape[0]} tweets")
    print(f"Date range: {tweets['Date & Time'].min()} to {tweets['Date & Time'].max()}")

    # Save to data/raw folder
    print("=" * 60)
    print("SAVING DATA TO LOCAL FILE")
    tweets.to_csv(write_to, index=False)
    print(f"Data saved to: {write_to}")



if __name__ == "__main__":
    main()
