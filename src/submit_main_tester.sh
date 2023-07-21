#!/bin/bash

# Initialize and Load Modules
source /etc/profile
module load anaconda/2020b
module load cuda/10.0

# Run the script
python main_tester.py
