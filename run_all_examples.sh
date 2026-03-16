#!/usr/bin/env bash

for set in torch hf onnx; do
    for f in examples/$set/*.py; do
        script_name="${f##*/}"
        if [ "$script_name" == "__init__.py" ]; then
            continue
        fi
        echo "Running $script_name"
        python $f
    done
done