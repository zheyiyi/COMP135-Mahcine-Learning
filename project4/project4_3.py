# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:11:26 2017

@author: zheyiyi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:56:06 2017

@author: zheyiyi
"""

import numpy as np 
import random 
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_dataset(filename):
    fd = open(filename, "r")
    new_content = ""
    while len(new_content) == 0 or new_content[0].upper() != '@DATA':
        content = fd.readline()
        new_content = content.split()
       
    x = []
    y = []
    
    for line in fd:
        line = line.strip()
        inputs = line.split(",")
        x.append(inputs[:-1])
        y.append(inputs[-1])
    
    x = np.array(x, dtype = "int")
    y = np.array(y, dtype = "int")
    n_x = len(x[0])
    n_y = max(y) - min(y) + 1
    return x, y, n_x, n_y
    
    
def initialize_parameters(parameters):
    n_x, d, w, n_y = parameters[0], parameters[1], parameters[2], parameters[3]
    
    dict1 = {}
    
    n_w = n_x
    
    #random.seed(0)
    for i in range(1, d + 1):
                  
        dict1["w" + str(i)] = np.array([[random.uniform(-0.1,0.1) for x in range(0, n_w)] for i in range(w)])
        n_w = w
      
    dict1["w" + str(d + 1)] =np.array([[random.uniform(-0.1,0.1) for x in range(0, n_w)] for i in range(n_y)])
    
    return dict1


def sigmoid(x):
    x = [max(-50,i)for i in x[0]]
    x = np.array([x])
    return 1 / (1 + np.exp(-x))


def model_forward(x, network,output_dict):
    #output_dict = {}
    for i in range(len(network)):
      
        temp = np.dot(x, network["w" + str(i + 1)].T)
        output_dict["output" + str(i + 1)] = sigmoid(temp)
        x = output_dict["output" + str(i + 1)]
       
    return output_dict
    
def transfer_derivative(output):
    return output * (1 - output) 
    

def backpropagation(output_dict, dict1, label,delta_dict):
    #delta_dict = {}
    for i in reversed(range(len(dict1))):
        if i == len(dict1) - 1:
            error = label - output_dict["output" + str(i + 1)]
          
            delta_dict["delta" + str(i + 1)] =  - error * transfer_derivative(output_dict["output" + str(i + 1)])
            
        else:
            error = np.dot(delta_dict["delta" + str(i + 2)], dict1["w" + str(i + 2)])           
            delta_dict["delta" + str(i + 1)] = error * transfer_derivative(output_dict["output" + str(i + 1)])
                    
       
    return delta_dict


def update(learn_rate, weight_dict, output_dict, delta_dict, input_x):
    n = len(weight_dict)
    for i in range(n):             
        update = learn_rate * np.array([delta * input_x[0] for delta in delta_dict["delta" + str(i + 1)][0]])          
        weight_dict["w" + str(i + 1)] = weight_dict["w" + str(i + 1)] - update
        input_x = output_dict["output" + str(i + 1)]   
        
def onehot(n_x,label):
    result = []
    for value in label:
        result.append([1 if i == value else 0 for i in range(n_x)])
    new_label = np.array(result)
    
    return new_label  

def back_onehot(output):
    max_value = -sys.maxsize
    max_index = -1
    for index, value in enumerate(output[0]):
        if max_value < value:
            max_value = value
            max_index = index
            
    result = []
    for i in range(len(output[0])):
        if i == max_index:
            result.append(1)
        else:
            result.append(0)
            
    return result
 
       
        
def train_data(train, test, weight_dict, iteration,learn_rate, new_label,new_label_test, d):
    output_dict = {}
    delta_dict = {}
    error = []
    iter_numbers = []
    output_dict1 = {}
    error1 = []
  
    for j in range(iteration):
        
        num = 0
        total_num = len(train)
        num1 = 0
        total_num1 = len(test)
        for i in range(len(train)):
            x = np.array([train[i]])
            output_dict = model_forward(x, weight_dict, output_dict)
            delta_dict = backpropagation(output_dict, weight_dict, new_label[i],delta_dict)
            update(learn_rate, weight_dict, output_dict, delta_dict, x)
           
            new_x = back_onehot(output_dict["output" + str(d + 1)])
            new_x = np.array([new_x])
            label = np.array([new_label[i]])
            
            if np.array_equal(new_x, label):
                num += 1                
            if j == iteration - 1:
                new_x = back_onehot(output_dict["output" + str(d + 1)])
                new_x = np.array([new_x])
                print(np.array_equal(new_x, label))
                print(new_x)
                print(label)
                print(output_dict["output" + str(d + 1)])
                
        for i in range(len(test)):
            x = np.array([test[i]])
            output_dict1 = model_forward(x, weight_dict, output_dict1)           
            new_x_1 = back_onehot(output_dict1["output" + str(d + 1)])
            new_x_1 = np.array([new_x_1])
            label_1 = np.array([new_label_test[i]])
            if np.array_equal(new_x_1, label_1):
                num1 += 1    
            
            
        accur = 1 - num / total_num
        accur1 = 1 - num1 / total_num1
        error.append(accur)
        error1.append(accur1)
        iter_numbers.append(j)
   
    return error,error1, iter_numbers
    

error_ratio = []  
train, train_label, n_x, n_y = load_dataset("optdigits_train.arff")
new_label_train = onehot(n_y, train_label)

test, test_label, n_x, n_y = load_dataset("optdigits_test.arff")
new_label_test = onehot(n_y, test_label)
"""
d = 0
w = 10
learn_parameters_train = (n_x, d, w, n_y)
learn_rate = 0.1
weight_dict_train = initialize_parameters(learn_parameters_train)
error_train, error_test, iter_numbers = train_data(train,test, weight_dict_train, 200,learn_rate, new_label_train,new_label_test,d) 
error_ratio.append(error_test[-1])

plt.figure()
plt.plot(iter_numbers, error_train, 'r-', label = "training set error as a function of iterations")
plt.plot(iter_numbers, error_test, 'k-', label = "testing set error as a function of iterations")
plt.xlim(0,200)
plt.ylim((0, 1.1))
plt.legend()
plt.title("Figure9 d = 0 and w = 10")
plt.ylabel("error ratio")
plt.xlabel("iteration number")
plt.savefig('graph9.png')


d = 1
w = 10
learn_parameters_train = (n_x, d, w, n_y)
learn_rate = 0.1
weight_dict_train = initialize_parameters(learn_parameters_train)
error_train, error_test, iter_numbers = train_data(train,test, weight_dict_train, 200,learn_rate, new_label_train,new_label_test,d) 
error_ratio.append(error_test[-1])

plt.figure()
plt.plot(iter_numbers, error_train, 'r-', label = "training set error as a function of iterations")
plt.plot(iter_numbers, error_test, 'k-', label = "testing set error as a function of iterations")
plt.xlim(0,200)
plt.ylim((0, 1.1))
plt.legend()
plt.title("Figure10 d = 1 and w = 10")
plt.ylabel("error ratio")
plt.xlabel("iteration number")
plt.savefig('graph10.png')


d = 2
w = 10
learn_parameters_train = (n_x, d, w, n_y)
learn_rate = 0.1
weight_dict_train = initialize_parameters(learn_parameters_train)
error_train, error_test, iter_numbers = train_data(train,test, weight_dict_train, 200,learn_rate, new_label_train,new_label_test,d) 
error_ratio.append(error_test[-1])

plt.figure()
plt.plot(iter_numbers, error_train, 'r-', label = "training set error as a function of iterations")
plt.plot(iter_numbers, error_test, 'k-', label = "testing set error as a function of iterations")
plt.xlim(0,200)
plt.ylim((0, 1.1))
plt.legend()
plt.title("Figure11 d = 2 and w = 10")
plt.ylabel("error ratio")
plt.xlabel("iteration number")
plt.savefig('graph11.png')


d = 3
w = 10
learn_parameters_train = (n_x, d, w, n_y)
learn_rate = 0.1
weight_dict_train = initialize_parameters(learn_parameters_train)
error_train, error_test, iter_numbers = train_data(train,test, weight_dict_train, 200,learn_rate, new_label_train,new_label_test,d) 
error_ratio.append(error_test[-1])

plt.figure()
plt.plot(iter_numbers, error_train, 'r-', label = "training set error as a function of iterations")
plt.plot(iter_numbers, error_test, 'k-', label = "testing set error as a function of iterations")
plt.xlim(0,200)
plt.ylim((0, 1.1))
plt.legend()
plt.title("Figure12 d = 3 and w = 10")
plt.ylabel("error ratio")
plt.xlabel("iteration number")
plt.savefig('graph12.png')
"""

d = 4
w = 10
learn_parameters_train = (n_x, d, w, n_y)
learn_rate = 0.1
weight_dict_train = initialize_parameters(learn_parameters_train)
error_train, error_test, iter_numbers = train_data(train,test, weight_dict_train, 200,learn_rate, new_label_train,new_label_test,d) 
error_ratio.append(error_test[-1])

plt.figure()
plt.plot(iter_numbers, error_train, 'r-', label = "training set error as a function of iterations")
plt.plot(iter_numbers, error_test, 'k-', label = "testing set error as a function of iterations")
plt.xlim(0,200)
plt.ylim((0.83, 0.95))
plt.legend()
plt.title("Figure13 d = 4 and w = 10")
plt.ylabel("error ratio")
plt.xlabel("iteration number")
plt.savefig('graph13.png')

d = 5
w = 10
learn_parameters_train = (n_x, d, w, n_y)
learn_rate = 0.1
weight_dict_train = initialize_parameters(learn_parameters_train)
error_train, error_test, iter_numbers = train_data(train,test, weight_dict_train, 200,learn_rate, new_label_train,new_label_test,d) 
error_ratio.append(error_test[-1])

plt.figure()
plt.plot(iter_numbers, error_train, 'r-', label = "training set error as a function of iterations")
plt.plot(iter_numbers, error_test, 'k-', label = "testing set error as a function of iterations")
plt.xlim(0,200)
plt.ylim((0.83, 0.95))
plt.legend()
plt.title("Figure14 d = 5 and w = 10")
plt.ylabel("error ratio")
plt.xlabel("iteration number")
plt.savefig('graph14.png')
"""
d_set = [0, 1, 2, 3, 4, 5]
plt.figure()
plt.plot(d_set, error_ratio, 'r-', label = "test set error as a function of w")

plt.xlim(0,6)
plt.ylim((0, 1.1))
plt.legend()
plt.title("Figure15")
plt.ylabel("error ratio")
plt.xlabel("d number")
plt.savefig('graph15.png')
"""