#!/bin/bash

for file in train/*.jpg; do
    echo $file >> train.txt
done
