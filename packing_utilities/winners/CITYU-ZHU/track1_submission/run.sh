#!/bin/bash

# the directories of input and output files, recieved externally
INPUT_PATH=$1
OUTPUT_PATH=$2

# install all dependencies
#pip install numpy

# run algorithm
#python main.py $INPUT_PATH $OUTPUT_PATH
#java -cp ./track2.jar com.my.vrp.Main2 ./data/inputs/ ./data/outputs/
#java -cp ./track2.jar com.my.vrp.Main2 $INPUT_PATH $OUTPUT_PATH
#java -cp ./track1.jar com.my.vrp.Main $INPUT_PATH $OUTPUT_PATH

#java -cp ./track2_main9_350_18850.jar com.my.vrp.Main9_350_18850 $INPUT_PATH $OUTPUT_PATH
#java -cp ./track1_liu.jar jmetal.metaheuristics.moead.MOEAD_SDVRP $INPUT_PATH $OUTPUT_PATH

#java -cp ./track1_221.jar com.my.vrp.Main9plus1_ $INPUT_PATH $OUTPUT_PATH
java -cp ./EMO2021_Track1_final.jar com.my.vrp.Track1 $INPUT_PATH $OUTPUT_PATH
