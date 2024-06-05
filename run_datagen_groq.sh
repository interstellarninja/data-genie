#!/bin/bash

# initialize redis vectordb on docker
./run_vectordb.sh

# run data correction pipeline
python scripts/datagen.py --ai_vendor groq --num_results 10 --num_tasks 10 --type function_calling