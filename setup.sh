#!/usr/bin/env bash

# Configuration
PYTHON="${EVOWLUATOR_PYTHON:-python3}"

# Safeguards
set -o pipefail
set -o errtrace
set -o errexit

# Start
cd "$(dirname "${0}")"

if [ ! -f venv/bin/activate ]; then
	echo "Creating venv..."
	rm -rf venv
	"${PYTHON}" -m venv venv
fi

echo "Installing Python package requirements..."
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
deactivate

echo "Done!"
