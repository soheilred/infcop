#!/bin/sh

ID=$(echo $1 | awk '{print $4}')
# echo $ID
DIR="../output/errors/tbst_${ID}.log"
# echo $DIR
grep "In dir:" $DIR | awk '{print $6}'
grep "In dir:" $DIR | awk '{print $6}' | awk -F "/" '{print $11}'