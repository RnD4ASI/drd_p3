#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Run the weight learning script
python -m src.learn_weights "$@"
