#!/bin/bash

# Get the current date and time
DATE=$(date '+%Y-%m-%d-%H-%M-%S')

# Check if the "logs" directory exists, if not, create it
if [ ! -d "logs" ]; then
    mkdir logs
fi

# Create the file with the current date in its name
touch "logs/bench-$DATE.txt"


# Iterate over all text files in the sim_tests directory
for FILE in $(ls sim_tests/*.txt); do
    # Get the case name from the file name by removing the directory and the .txt extension
    CASE_NAME=$(basename $FILE .txt)

    # Run the python command with the case name and log file 5 times
    for i in {1..5}; do
        timeout 1000s python environment.py --method mcts --case $CASE_NAME --log "logs/bench-$DATE.txt"
    done

    for i in {1..5}; do
        timeout 200s python environment.py --method greedy --case $CASE_NAME --log "logs/bench-$DATE.txt"
    done
done