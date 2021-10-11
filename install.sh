#!/usr/bin/env bash

# Configuration
ROOT_DIR="$(cd "$(dirname "${0}")"; pwd -P)"
LAUNCHER_BIN="${1:-"${ROOT_DIR}/bin"}/evowluate"

# Safeguards
set -o nounset
set -o pipefail
set -o errtrace
set -o errexit

# Start
{
    echo '#!/usr/bin/env bash'
    echo "cd \"${ROOT_DIR}\""
    echo 'source venv/bin/activate'
    # shellcheck disable=SC2016
    echo 'export EVOWLUATOR_EXE="$(basename "${0}")"'
    echo 'python -m evowluator.main "$@"'
} > "${LAUNCHER_BIN}"

chmod 755 "${LAUNCHER_BIN}"
echo "Launcher created at ${LAUNCHER_BIN}"
