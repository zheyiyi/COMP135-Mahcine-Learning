import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess

accuracy = []
data = [14,24,34,44,54,64,74,84,94]
data_set =[('EEGTrainingData_' + str((i + 1) * 10 + 4) + '.arff', 'EEGTestingData_' + str((i + 1) * 10 + 4) + '.arff' )for i in range(9)]

weka_class_path = '/r/aiml/ml-software/weka-3-6-11/weka.jar'

classifiers = ['weka.classifiers.lazy.IBk', 'weka.classifiers.trees.J48']

classifier = classifiers[0]

for training_sample, testing_sample in data_set:
    mypath ='java -classpath ' + weka_class_path + ' ' + classifier + ' -t ' + training_sample + ' -T ' + testing_sample + '| grep Correctly'

    result = subprocess.check_output(mypath, shell = True)

    accuracy.append(float(result.split()[10])/100)


plt.plot(data,accuracy)

plt.show()
