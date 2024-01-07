#!/bin/bash
# source conda init script
source /home/interstellarninja/anaconda3/etc/profile.d/conda.sh
conda activate data-gen

# initialize redis vectordb on docker
./run_vectordb.sh

# run data correction pipeline
python scripts/reversegen.py --ai_vendor openai --num_results 10 --num_tasks 10