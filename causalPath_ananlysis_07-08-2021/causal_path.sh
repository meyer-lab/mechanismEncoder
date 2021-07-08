# Will Yashar
# July 02, 2021
#
# Version: GNU bash, version 3.2.57(1)-release (x86_64-apple-darwin19)
#
# Usage:
# bash causal_path.sh

#!/bin/bash

for grandparent in $(ls results/); do

    for parent in $(ls results/${grandparent}); do

        for child in $(ls results/${grandparent}/${parent}); do

            echo "${grandparent}/${parent}/${child}"
            #java -jar /Users/yashar/gitPackages/causalpath/target/causalpath.jar /Users/yashar/Box/Research\ Data/Will/Projects/mechanism_encoder/aml_data/results/${grandparent}/${parent}/${child}

            java -jar /Users/yashar/gitPackages/causalpath/target/causalpath.jar /Users/yashar/Documents/mechanism_autoencoder_local/results/${grandparent}/${parent}/${child}

        done
    done
done

java -jar C:\Users\wmyas\gitPackages\causalpath\target\causalpath.jar /Users/yashar/Documents/mechanism_autoencoder_local/results/${grandparent}/${parent}/${child}