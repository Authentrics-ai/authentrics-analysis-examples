#!/usr/bin/env bash

echo "Make sure to login before running the examples"
echo "If you are not logged in cancel this and run: authrx login"
read -p "Press Enter to continue or Ctrl+C to cancel"

for set in torch onnx hf; do
    cd examples/$set

    for f in *.py; do
        echo "Running $f"
        python $f
    
        project_dir=$(ls -d my_*project | head -n 1)
        if [ -z "$project_dir" ]; then
            continue
        fi
    
        authrx project delete $project_dir
        rm -rf $project_dir
    done
    
    cd ../..
done
