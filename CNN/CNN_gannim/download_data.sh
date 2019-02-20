#!/bin/bash

wget https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data/rt-polaritydata/rt-polarity.neg
wget https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data/rt-polaritydata/rt-polarity.pos

mkdir data
mv rt-polarity.neg data 
mv rt-polarity.pos data 
