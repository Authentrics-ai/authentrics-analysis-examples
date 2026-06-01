#!/bin/bash

wheel_path=$1

if [ -z "$wheel_path" ]; then
    echo "Usage: $0 <wheel_path>"
    exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
pip install $wheel_path
