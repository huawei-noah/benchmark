#!/bin/bash

# the directories of input and output files, recieved externally
INPUT_PATH=$1
OUTPUT_PATH=$2

# install all dependencies
pip install numpy

# run algorithm
python runpy.py $INPUT_PATH $OUTPUT_PATH