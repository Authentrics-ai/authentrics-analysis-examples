#!/bin/bash

python3 -m venv .venv

# Set the extra index url for the virtual environment to download the Authentrics SDK from the Google Cloud Package Registry
echo -e "[global]\nextra-index-url = https://us-central1-python.pkg.dev/authentrics/authentrics/simple\n" > .venv/pip.conf

# Activate the virtual environment
source .venv/bin/activate

# Install the Google Cloud Package Registry authentication library
pip install keyrings.google-artifactregistry-auth
pip install authentrics