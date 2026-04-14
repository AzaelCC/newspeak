"""Compatibility entry point for running Newspeak from the source tree."""

from newspeak.app import create_app
from newspeak.cli import main

app = create_app()

if __name__ == "__main__":
    main()
