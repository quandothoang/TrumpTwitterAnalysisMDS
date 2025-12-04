# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-02

"""Downloads the Trump tweets dataset from URL and saves to data/raw folder.

This script ensures reproducibility by:
1. Downloading data programmatically from the source URL
2. Saving an exact copy to data/raw/ for archival purposes
3. The saved copy ensures analysis remains reproducible even if URL becomes unavailable

Usage: python scripts/read_trump_tweets.py [OPTIONS]

Options:
    --url         URL to the raw CSV file
    --write_to    Path to save the CSV file (default: data/raw/realDonaldTrump_in_office.csv)
"""

import click
import requests
from pathlib import Path


@click.command()
@click.option(
    '--url',
    default="https://raw.githubusercontent.com/MarkHershey/CompleteTrumpTweetsArchive/master/data/realDonaldTrump_in_office.csv",
    help='URL to the raw CSV file',
    type=str
)
@click.option(
    '--write_to',
    default="data/raw/realDonaldTrump_in_office.csv",
    help='Path (including filename) to save the CSV file',
    type=str
)
def main(url, write_to):
    """
    Downloads raw data from a URL and saves it to a specified file path.

    This script ensures reproducibility by downloading data programmatically
    from the source URL and saving an exact copy for archival purposes.
    """
    try:
        click.echo("Trump Tweets Dataset - Raw Data Download")
        click.echo(f"URL: {url}")
        click.echo(f"Save to: {write_to}")

        # Make HTTP request
        click.echo("Downloading data...", nl=False)
        resp = requests.get(url)
        resp.raise_for_status()
        click.secho("...Download Complete!", fg='magenta', bold=True)

        # Create directory if it doesn't exist
        output_path = Path(write_to)
        click.echo("Creating directory structure...", nl=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        click.secho("...Created directories!", fg='magenta', bold=True)

        # Save raw data to file
        click.echo("Writing to file...", nl=False)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(resp.text)
        click.secho("...File successfully created!", fg='magenta', bold=True)

        click.secho("Raw data download complete!", fg='green', bold=True)
        click.echo(f"Saved to: {output_path.resolve()}")

    except requests.exceptions.HTTPError as e:
        click.secho(f"HTTP Error: {e}", fg='red', bold=True)
        raise click.Abort()
    except requests.exceptions.RequestException as e:
        click.secho(f"Request Error: {e}", fg='red', bold=True)
        raise click.Abort()
    except IOError as e:
        click.secho(f"File Error: {e}", fg='red', bold=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
