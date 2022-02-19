#!/usr/bin/env bash

# bash strict mode
set -euo pipefail

CURRENT_DIR=$PWD
TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $TEST_DIR
if [ $# -eq 0 ]; then
    python3 -m unittest test_*.py
else
    python3 -m unittest $*
fi
cd $CURRENT_DIR
