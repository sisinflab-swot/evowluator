#!/usr/bin/env bash

# Configuration
PYTHON="${EVOWLUATOR_PYTHON:-python3}"
ROOT_DIR="$(cd "$(dirname "${0}")"; pwd -P)"
OWLTOOL_SRC_DIR="${ROOT_DIR}/lib/owltool"
OWLTOOL_OUT_DIR="${OWLTOOL_SRC_DIR}/build/libs"
OWLTOOL_BIN_DIR="${ROOT_DIR}/bin/OWLTool"

# Safeguards
set -o pipefail
set -o errtrace
set -o errexit

# Start
cd "${ROOT_DIR}"

# Cleanup
rm -rf venv

# Python
echo "Creating venv..."
"${PYTHON}" -m venv venv

echo "Installing Python package requirements..."
source venv/bin/activate
pip3 install --upgrade pip
pip3 install wheel
pip3 install -r requirements.txt
pip3 install -e lib/pyutils
deactivate

# Java
echo "Building Java dependencies..."
cd "${OWLTOOL_SRC_DIR}"
gradle wrapper
./gradlew clean build
cd "${ROOT_DIR}"
mkdir -p "${OWLTOOL_BIN_DIR}"
mv -f "${OWLTOOL_OUT_DIR}"/*.jar "${OWLTOOL_BIN_DIR}/owltool.jar"

echo "Done!"
