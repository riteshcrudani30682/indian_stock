"""Command-line interface for Indian Stock Analysis"""

import click
import subprocess
import sys
import os

@click.group()
def cli():
    """Indian Stock Analysis CLI"""
    pass

@cli.command()
def run():
    """Run the Indian Stock Analysis web application"""
    click.echo("Starting Indian Stock Analysis web application...")
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                   os.path.join(os.path.dirname(__file__), "app.py")])

@cli.command()
def version():
    """Show version information"""
    from . import __version__
    click.echo(f"Indian Stock Analysis version {__version__}")

if __name__ == "__main__":
    cli()
