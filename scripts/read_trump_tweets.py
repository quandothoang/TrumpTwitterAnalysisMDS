# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-02

# code format guidance and debugging provided by Claude 2.5 sonnet 

"""Downloads the Trump tweets dataset from URL and saves to data/raw folder.
This script ensures reproducibility by:
1. Downloading data programmatically from the source URL
2. Saving an exact copy to data/raw/ for archival purposes
3. The saved copy ensures analysis remains reproducible even if URL becomes unavailable

Usage: 
    python read_trump_tweets.py --url=<url> --write_to=<path>
    
Example:
    python read_trump_tweets.py 
        --url="https://raw.githubusercontent.com/MarkHershey/CompleteTrumpTweetsArchive/master/data/realDonaldTrump_in_office.csv" 
        --write_to="data/raw/realDonaldTrump_in_office.csv"

Options:
    --url=<url>             URL to the raw CSV file
    --write_to=<path>       Path (including filename) to save the CSV file
"""

import click                         # command-line interfaces with click
import requests                     # HTTP requests 
from pathlib import Path           # for cross-platform directory structures


@click.command()                 # deecorater alerts click the functions below are CLI commands 
@click.option(                  # option 1 => pass custom URL 
    '--url',
    default="https://raw.githubusercontent.com/MarkHershey/CompleteTrumpTweetsArchive/master/data/realDonaldTrump_in_office.csv",
    help='URL to the raw CSV file',
    type=str
)
@click.option(              # option 2 => pass custom filepath and filename with defaults
    '--write_to',
    default="../data/raw/realDonaldTrump_in_office.csv",
    help='Path (including filename) to save the CSV file',
    type=str
)
def download_raw_data(url, write_to):
    """
    Downloads raw data from a URL and saves it to a specified file path, 
    otherwise the default download file and path is data/raw/realDonaldTrump_in_office.csv
    
    This script ensures reproducibility by downloading data programmatically
    from the source URL and saving an exact copy for archival purposes.
    """
    try:
        click.echo("=" * 70)
        click.echo("Trump Tweets Dataset - Raw Data Download")
        click.echo("=" * 70)
        click.echo(f"URL: {url}")
        click.echo(f"Save to: {write_to}")
        click.echo("=" * 70)
        
        # Make HTTP request
        click.echo("Downloading data...", nl=False)
        resp = requests.get(url)
        resp.raise_for_status()
        click.secho("...Download Complete! \n", fg='magenta', bold=True)
        
        # Create directory if it doesn't exist
        output_path = Path(write_to)
        click.echo(f"Creating directory structure...", nl=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        click.secho("...Created directories! \n", fg='magenta', bold=True)
        
        # Save raw data to file
        click.echo(f"Writing to file...", nl=False)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(resp.text)
        click.secho("...File successfully created!", fg='magenta', bold=True)
        
        click.echo("=" * 70)
        click.secho("Raw data download complete! \n", fg='green', bold=True)
        click.echo(f"Saved to: {output_path.resolve()}")
        click.echo("=" * 70)
        
    except requests.exceptions.HTTPError as e:
        click.secho(f"!HTTP Error: {e}", fg='red', bold=True)
        raise click.Abort()
    except requests.exceptions.RequestException as e:
        click.secho(f"!Request Error: {e}", fg='red', bold=True)
        raise click.Abort()
    except IOError as e:
        click.secho(f"!File Error: {e}", fg='red', bold=True)
        raise click.Abort()
        
    click.secho("\nAll operations complete! Terminating script. \n", fg='blue', bold=True)

if __name__ == "__main__":
    download_raw_data()
    