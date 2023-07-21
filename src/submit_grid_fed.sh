#!/bin/bash

# Initialize and Load Modules
source /etc/profile
module load anaconda/2020b


echo $LLSUB_RANK
echo $LLSUB_SIZE

# Run the script
python main.py $LLSUB_RANK $LLSUB_SIZE