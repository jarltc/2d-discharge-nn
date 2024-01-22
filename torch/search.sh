#!/bin/bash

search_string="validation_loss"

# Loop through all the files in the current directory
for file in *; do
    if [ -f "$file" ]; then  # Check if the current item is a file
        # Use grep to search for the string in the file
        if grep -q "$search_string" "$file"; then
            echo "Found '$search_string' in file: $file"
        fi
    fi
done