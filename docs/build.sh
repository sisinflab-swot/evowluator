#!/usr/bin/env bash

# Configuration
ROOT_DIR="$(cd "$(dirname "${0}")"/..; pwd -P)"
SRC_DIR="${ROOT_DIR}/docs"
BUILD_DIR="${SRC_DIR}/_build"
HTML_DIR="${BUILD_DIR}/html"
LOG_DIR="${BUILD_DIR}/log"

# Safeguards
set -o pipefail
set -o errtrace
set -o errexit

# Cleanup
rm -rf "${BUILD_DIR}"
mkdir -p "${LOG_DIR}" "${HTML_DIR}"

# Start
echo "Installing requirements..."
source "${ROOT_DIR}/venv/bin/activate"
pip3 install -r "${SRC_DIR}/requirements.txt" > "${LOG_DIR}/pip.log" 2>&1

echo "Building docs..."
sphinx-build -b html "${SRC_DIR}" "${HTML_DIR}"  > "${LOG_DIR}/sphinx.log" 2>&1

echo "Done!"
open "${HTML_DIR}/index.html"
