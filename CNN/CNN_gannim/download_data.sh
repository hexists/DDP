#!/bin/bash

wget https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data/rt-polaritydata/rt-polarity.neg
wget https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data/rt-polaritydata/rt-polarity.pos
wget https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip

mkdir data
mv rt-polarity.neg data 
mv rt-polarity.pos data 
mv ted_en-20160408.zip data

unzip data/ted_en-20160408.zip -d data
