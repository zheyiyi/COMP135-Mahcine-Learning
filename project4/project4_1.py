# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:58:47 2017

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
    
    return x, y, n_x

 
def initialize_parameters(parameters):
    n_x, d, w, n_y = parameters[0], parameters[1], parameters[2], parameters[3]
    
    dict1 = {}
    
    n_w = n_x
    
    random.seed(0)
    for i in range(1, d + 1):
                  
        dict1["w" + str(i)] = np.array([[random.uniform(-0.1,0.1) for x in range(0, n_w)] for i in range(w)])
        n_w = w
      
    dict1["w" + str(d + 1)] =np.array([[random.uniform(-0.1,0.1) for x in range(0, n_w)] for i in range(n_y)])
    
    return dict1



def sigmoid(x):
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
    
""" 
def update(learn_rate, weight_dict, output_dict, delta_dict, input_x):
    input_x = np.array(input_x)
    for i in range(len(weight_dict)):
        
        update = learn_rate * np.array([delta * input_x for delta in delta_dict["delta" + str(i + 1)]])
        weight_dict["w" + str(i + 1)] = weight_dict["w" + str(i + 1)] - update
        input_x = output_dict["output" + str(i + 1)]   
    
    
    return weight_dict

"""

def update(learn_rate, weight_dict, output_dict, delta_dict, input_x):
    n = len(weight_dict)
    for i in range(n):             
        update = learn_rate * np.array([delta * input_x[0] for delta in delta_dict["delta" + str(i + 1)][0]])          
        weight_dict["w" + str(i + 1)] = weight_dict["w" + str(i + 1)] - update
        input_x = output_dict["output" + str(i + 1)]   
        
def onehot(n_x,label):
    result = []
    for value in label:
        result.append([1 if i == value else 0 for i in range(1,n_x + 1)])
    new_label = np.array(result)
    
    return new_label    
    

""" 
x = np.array([[1,1]])

weight_dict = {}
weight_dict["w1"] = np.array([[1, 2],[2,1]])
weight_dict["w2"] = np.array([[1,3],[2,2]])
weight_dict["w3"] = np.array([[2,1]])

new_label = np.array([0])
learn_rate = 0.1
output_dict = model_forward(x, weight_dict)
delta_dict = backpropagation(output_dict, weight_dict, new_label)
update(learn_rate, weight_dict, output_dict, delta_dict, x)

print(weight_dict)


  
"""  

# d is the number of hidden layer
# w is the size(number) of node for each hidden layer
# n_y is the number(size) of node for output layer


  
 
"""

#for i in range(len(train)):
x = np.array([train[0]])
output_dict = model_forward(x, weight_dict)
delta_dict = backpropagation(output_dict, weight_dict, new_label[0])
update(learn_rate, weight_dict, output_dict, delta_dict, x)
       
"""    
      
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
        
def train_data(train, weight_dict, iteration,learn_rate, new_label,d):
    output_dict = {}
    delta_dict = {}
    error = []
    iter_numbers = []
    binary = []
    for j in range(iteration):
        
        num = 0
        total_num = len(train)
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
                #print(np.array_equal(new_x, label))
                print(new_x)
                #print(label)
                
                #print(output_dict["output"+ str(d)])
                binary.append(output_dict["output"+ str(d)])
                #print(output_dict["output" + str(d + 1)])
        
        accur = 1 - num / total_num
  
        error.append(accur)
        iter_numbers.append(j)
    return error, iter_numbers,binary


train, label, n_x = load_dataset("838.arff")
new_label = onehot(n_x, label)
d = 1
w = 3
n_y = len(train[0])
n_x = len(train[0])
learn_parameters = (n_x, d, w, n_y)
learn_rate = 0.1
weight_dict = initialize_parameters(learn_parameters)
error, iter_numbers, binary = train_data(train, weight_dict, 3000,learn_rate,new_label,d) 

binary_pre = []
for value_set in binary:
    a = [1 if i >= 0.5 else 0for i in value_set[0]]
    binary_pre.append(a)

print(binary_pre)
plt.figure()
plt.plot(iter_numbers,error, 'r-', label = " training set error as a function of iterations ")

plt.xlim(0,3000)
plt.ylim((0, 1.1))
plt.legend()
plt.title("Figure1 8-3-8")
plt.ylabel("error ratio")
plt.xlabel("iteration number")
plt.savefig('graph1.png')
