#!/bin/bash

# the directories of input and output files, recieved externally
INPUT_PATH=$1
OUTPUT_PATH=$2

# install all dependencies

# run algorithm
java -jar BinPacking3D-1.0-SNAPSHOT-jar-with-dependencies.jar $INPUT_PATH $OUTPUT_PATH