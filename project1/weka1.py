#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:40:59 2017

@author: zheyiyi
"""


import subprocess

data_set =[('data/EEGTrainingData_' + str((i + 1) * 10 + 4) + '.arff', 'data/EEGTestingData_' + str((i + 1) * 10 + 4) + '.arff' )for i in range(9)]

weka_class_path = '/r/aiml/ml-software/weka-3-6-11/weka.jar'
classifiers = ['weka.classifiers.lazy.IBk', 'weka.classifiers.trees.J48']

classifier = classifiers[0]

training_sample = './data/EEGTrainingData_24.arff'
testing_sample = './data/EEGTestingData_24.arff'

mypath ='java -classpath ' + weka_class_path + ' ' + classifier + ' -t ' + training_sample + ' -T ' + testing_sample + '| grep Correctly'
result = subprocess.check_output(mypath, shell = True)

