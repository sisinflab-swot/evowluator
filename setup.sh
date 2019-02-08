#!/usr/bin/env bash
cd "$(dirname "${0}")"

# Safeguards
set -o pipefail
set -o errtrace
set -o errexit

# Create virtualenv
if [ ! -d venv ]; then
	echo "Creating venv..."
	python3 -m venv venv
fi

echo "Installing Python package requirements..."
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
deactivate

echo "Done!"
