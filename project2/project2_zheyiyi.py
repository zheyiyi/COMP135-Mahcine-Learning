# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 07:07:46 2017

@author: zheyiyi
"""

import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# seperate file into (positive sentence, negative sentence) tuple 
# for 0.1n to n
def word_div_learn(fd):
    total_list = []

    for line in fd.readlines():
        line = line.strip()
        sents = line.split("\t")        
        sent = sents[0]
        sent = sent.strip("!.?")
        sent = sent.split()
        value = sents[1]
        total_list.append((sent,value))
    new_list = []
    for i in range(1,11):
        random.shuffle(total_list)
        n = len(total_list)
        m = int(i * 0.1 * n)
       
        new_total = total_list[:m]
        pos_list = []
        neg_list = []
        for sent, value in new_total:
            if int(value) == 1:
                pos_list.append((sent,value))
            else:
                neg_list.append((sent,value))
        new_list.append((pos_list,neg_list))
        
    return new_list 

# seperate all data into into (positive sentence, negative sentence) tuple 
def word_div(fd):
    total_list = []

    for line in fd.readlines():
        line = line.strip()
        sents = line.split("\t")        
        sent = sents[0]
        sent = sent.strip("!.?")
        sent = sent.split()
        value = sents[1]
        total_list.append((sent,value))
    random.shuffle(total_list)
    pos_list = []
    neg_list = []
    for sent, value in total_list:
         if int(value) == 1:
             pos_list.append((sent,value))          
         else:
             neg_list.append((sent,value))
               
    return (pos_list, neg_list)
    

# no smooth function for training data
def naive_train_nosmooth(train_list):
    pos_sen_total,neg_sen_total = 0, 0
    pos_word_total,neg_word_total = 0, 0
    pos_dish,pos_ratio_dish = {}, {} 
    neg_dish, neg_ratio_dish = {},{}
        
    for sent, value in train_list:
        if int(value) == 1:
            pos_sen_total += 1           
            for word in sent:
                pos_word_total += 1
                if word not in pos_dish:
                    pos_dish[word] = 0
            
                pos_dish[word] += 1              
        else:
            neg_sen_total += 1
            for word in sent:
                neg_word_total += 1
                if word not in neg_dish:
                    neg_dish[word] = 0
                neg_dish[word] += 1
 
    pos_ratio = pos_sen_total / (pos_sen_total + neg_sen_total)
    neg_ratio = neg_sen_total / (pos_sen_total + neg_sen_total)


    for key, value in pos_dish.items(): 
     
        pos_ratio_dish[key] = value / pos_word_total
    

    for key, value in neg_dish.items():        
        neg_ratio_dish[key] = value / neg_word_total
   
    return pos_ratio, neg_ratio, pos_ratio_dish, neg_ratio_dish
    

# no smothing for testing data 
def native_test_nosmooth(test_list, pos_ratio, neg_ratio, pos_ratio_dish, neg_ratio_dish):

    n = len(test_list)
    
    no_smooth_total = 0

    
    for sent,value in test_list:
        pos_value = 0
        neg_value = 0
        pos_value = math.log(pos_ratio)
        neg_value = math.log(neg_ratio)
        for word in sent:
            if word in pos_ratio_dish:
                pos_value += math.log(pos_ratio_dish[word])
                
            
            elif word not in pos_ratio_dish and word in neg_ratio_dish:
                pos_value +=  -math.inf
        for word in sent:       
            if word in neg_ratio_dish:
                neg_value += math.log(neg_ratio_dish[word]) 
               
            elif word not in neg_ratio_dish and word in pos_ratio_dish:
                neg_value +=  -math.inf
        
        if pos_value >= neg_value:
           no_smooth = 1 
        else:
           no_smooth = 0
       
        if no_smooth == int(value):
            no_smooth_total += 1

    return  no_smooth_total / n

# smooth for trainig data
def naive_train_smooth(train_list, m):
    
    pos_sen_total,neg_sen_total = 0, 0
    pos_word_total,neg_word_total = 0, 0
    pos_dish,pos_ratio_dish = {}, {} 
    neg_dish, neg_ratio_dish = {},{}
  
    for sent, value in train_list:
        
        if int(value) == 1:
            pos_sen_total += 1
            for word in sent:
                pos_word_total += 1
                if word not in pos_dish:
                    pos_dish[word] = 0
                pos_dish[word] += 1              
        else:
            neg_sen_total += 1
            for word in sent:
                neg_word_total += 1
                if word not in neg_dish:
                    neg_dish[word] = 0
                neg_dish[word] += 1
    
    vocab = len(list(set(list(pos_dish.keys()) + list(neg_dish.keys()))))
    
    pos_ratio = pos_sen_total / (pos_sen_total + neg_sen_total)
    neg_ratio = neg_sen_total / (pos_sen_total + neg_sen_total)
    
    for key, value in pos_dish.items():
        
        pos_ratio_dish[key] = (int(value) + m) / (m * vocab + pos_word_total)
       
    for key, value in neg_dish.items():
        neg_ratio_dish[key] = (int(value) + m) / (m * vocab + neg_word_total)
    
    return pos_ratio, neg_ratio, pos_ratio_dish, neg_ratio_dish,pos_word_total, neg_word_total,m,vocab
    
# smooth for testing data         
def native_test_smooth(test_list, pos_ratio, neg_ratio, pos_ratio_dish, neg_ratio_dish,pos_word_total, neg_word_total,m,vocab):
    
    smooth = -1
    n = len(test_list)
    smooth_total = 0
    for sent,value in test_list:
        pos_value = 0
        neg_value = 0
        pos_value = math.log(pos_ratio)
        neg_value = math.log(neg_ratio)
        for word in sent:
            if word in pos_ratio_dish:
                pos_value += math.log(pos_ratio_dish[word])
            elif word not in pos_ratio_dish and  word in neg_ratio_dish:
                new_pro1 = m / (m * vocab + pos_word_total)
                if new_pro1 == 0:
                    pos_value += - math.inf
                else:    
                    pos_value += math.log(new_pro1)
         
            if word in neg_ratio_dish:
                neg_value += math.log(neg_ratio_dish[word])
            elif word not in neg_ratio_dish and  word in pos_ratio_dish:
                new_pro2 = m /( m * vocab + neg_word_total)
                if new_pro2 == 0:
                    neg_value += - math.inf
                else:    
                    neg_value += math.log(new_pro2)
                
        if pos_value > neg_value:
           smooth = 1
        else:
           smooth = 0
           
        if smooth == int(value):
           smooth_total += 1
        
    return  smooth_total / n
    

      
# form 10 mixture fold


def cross_validate(content):
   # print(content)
    pos_value = content[0]
   
    pos_n = len(pos_value) // 10
  
    neg_value = content[1]
    neg_n = len(neg_value) // 10
    
    temp_pos = []
    temp_neg = []    
    
    for i in range(10):   
        temp_pos.append(pos_value[i * pos_n: (i + 1) * pos_n])
        temp_neg.append(neg_value[i * neg_n:(i + 1) * neg_n])
    mix_fold = []
    
    for j in range(10):
        mix = temp_pos[j] + temp_neg[j]
        mix_fold.append(mix)
 
    for z in range(10):
        random.shuffle(mix_fold[z])

    return mix_fold
      
# get cross validation result for no smoothing
def cross_validate_nosmooth(mix_list):
    accuracy_list = []
    
    for i in range(10):
        test_set = mix_list[i]
        #print(test_set)
        train_set = mix_list[:i] + mix_list[i + 1:]
        #print(train_set)
        train_set_merge = []
        for i in range(len(train_set)):
            train_set_merge += train_set[i]
        
        pos_ratio, neg_ratio, pos_ratio_dish, neg_ratio_dish = naive_train_nosmooth(train_set_merge)
        
        accuracy_list.append(native_test_nosmooth(test_set,pos_ratio, neg_ratio, pos_ratio_dish, neg_ratio_dish))
          
    return accuracy_list

# get cross validation result for smoothing
def cross_validate_smooth(mix_list,m):
    accuracy_list = []
    for i in range(10):
        test_set = mix_list[i]
        train_set = mix_list[:i] + mix_list[i + 1:]
        train_set_merge = []
        for i in range(len(train_set)):
            train_set_merge += train_set[i]
        pos_ratio, neg_ratio, pos_ratio_dish, neg_ratio_dish,pos_word_total, neg_word_total,m,vocab = naive_train_smooth(train_set_merge,m)
        accuracy_list.append(native_test_smooth(test_set, pos_ratio, neg_ratio, pos_ratio_dish, neg_ratio_dish,pos_word_total, neg_word_total,m,vocab))
    return accuracy_list



accuracy_amazon_nosmooth = []
accuracy_amazon_smooth = []
fd1 = open("amazon_cells_labelled.txt")
total_list = word_div_learn(fd1)

for i in range(10):
    mix1 = cross_validate(total_list[i])
    accuracy_nosmooth = cross_validate_nosmooth(mix1)
    
    accuracy_amazon_nosmooth.append(accuracy_nosmooth)
    accuracy_smooth = cross_validate_smooth(mix1,1)   
    accuracy_amazon_smooth.append(accuracy_smooth)
fd1.close()


fd1 = open("amazon_cells_labelled.txt")
accuracy_amazon_m1 = []
accuracy_amazon_m2 = []

total_list_a_m = word_div(fd1)
mix1_a_m = cross_validate(total_list_a_m)

for j in range(10):
    accuracy_m_a_1 = cross_validate_smooth(mix1_a_m, j * 0.1)
    accuracy_amazon_m1.append(accuracy_m_a_1)
    
for j in range(1,11):
    accuracy_m_a_2 = cross_validate_smooth(mix1_a_m, j)
    accuracy_amazon_m2.append(accuracy_m_a_2)


accuracy_yelp_nosmooth = []
accuracy_yelp_smooth = []
fd2 = open("yelp_labelled.txt", encoding='utf-8')
total_list = word_div_learn(fd2)

for i in range(10):
    mix1 = cross_validate(total_list[i])
    accuracy_nosmooth = cross_validate_nosmooth(mix1)
    accuracy_yelp_nosmooth.append(accuracy_nosmooth)
    
    accuracy_smooth = cross_validate_smooth(mix1,1)   
    accuracy_yelp_smooth.append(accuracy_smooth)

fd2.close()


fd2 = open("yelp_labelled.txt", encoding='utf-8')
accuracy_yelp_m1 = []
accuracy_yelp_m2 = []

total_list_y_m = word_div(fd2)
mix1_y_m = cross_validate(total_list_y_m)

for j in range(10):
    accuracy_y_1 = cross_validate_smooth(mix1_y_m, j * 0.1)
    accuracy_yelp_m1.append(accuracy_y_1)
    
for j in range(1,11):
    accuracy_y_2 = cross_validate_smooth(mix1_y_m, j)
    accuracy_yelp_m2.append(accuracy_y_2)

accuracy_imdb_nosmooth = []
accuracy_imdb_smooth = []
fd3 = open("imdb_labelled.txt",encoding='utf-8')
total_list = word_div_learn(fd3)

for i in range(10):
    mix1 = cross_validate(total_list[i])
    accuracy_nosmooth = cross_validate_nosmooth(mix1)
    accuracy_imdb_nosmooth.append(accuracy_nosmooth)
    accuracy_smooth = cross_validate_smooth(mix1,1)   
    accuracy_imdb_smooth.append(accuracy_smooth)
fd3.close()


fd3 = open("imdb_labelled.txt", encoding='utf-8')
accuracy_imdb_m1 = []
accuracy_imdb_m2 = []

total_list_i_m = word_div(fd3)
mix1_i_m = cross_validate(total_list_i_m)

for j in range(10):
    accuracy_i_1 = cross_validate_smooth(mix1_i_m, j * 0.1)
    accuracy_imdb_m1.append(accuracy_i_1)
    
for j in range(1,11):
    accuracy_i_2 = cross_validate_smooth(mix1_i_m, j)
    accuracy_imdb_m2.append(accuracy_i_2)
    

#draw learning graph

#print(accuracy_amazon_nosmooth)
#print(accuracy_amazon_smooth)
mean_amazon_no = []
std_amazon_no = []
for i in range(len(accuracy_amazon_nosmooth)):
    accuracy_amazon_nosmooth1 = np.array(accuracy_amazon_nosmooth[i]).astype(np.float)
    mean_amazon_no.append(np.mean(accuracy_amazon_nosmooth1, axis = 0))
    std_amazon_no.append(np.std(accuracy_amazon_nosmooth1, axis = 0))

mean_amazon = []
std_amazon = []
for i in range(len(accuracy_amazon_smooth)):
    accuracy_amazon_smooth1 = np.array(accuracy_amazon_smooth[i]).astype(np.float)
    mean_amazon.append(np.mean(accuracy_amazon_smooth1, axis = 0))
    std_amazon.append(np.std(accuracy_amazon_smooth1, axis = 0))



#print(accuracy_yelp_nosmooth)
#print(accuracy_yelp_smooth)
mean_yelp_no = []
std_yelp_no = []
for i in range(len(accuracy_yelp_nosmooth)):
    accuracy_yelp_nosmooth1 = np.array(accuracy_yelp_nosmooth[i]).astype(np.float)
    mean_yelp_no.append(np.mean(accuracy_yelp_nosmooth1, axis = 0))
    std_yelp_no.append(np.std(accuracy_yelp_nosmooth1, axis = 0))

mean_yelp = []
std_yelp = []
for i in range(len(accuracy_yelp_smooth)):
    accuracy_yelp_smooth1 = np.array(accuracy_yelp_smooth[i]).astype(np.float)
    mean_yelp.append(np.mean(accuracy_yelp_smooth1, axis = 0))
    std_yelp.append(np.std(accuracy_yelp_smooth1, axis = 0))


#print(accuracy_imdb_nosmooth)
#print(accuracy_imdb_smooth)
mean_imdb_no = []
std_imdb_no = []
for i in range(len(accuracy_imdb_nosmooth)):
    accuracy_imdb_nosmooth1 = np.array(accuracy_imdb_nosmooth[i]).astype(np.float)
    mean_imdb_no.append(np.mean(accuracy_imdb_nosmooth1, axis = 0))
    std_imdb_no.append(np.std(accuracy_imdb_nosmooth1, axis = 0))

mean_imdb = []
std_imdb = []
for i in range(len(accuracy_imdb_smooth)):
    accuracy_imdb_smooth1 = np.array(accuracy_imdb_smooth[i]).astype(np.float)
    mean_imdb.append(np.mean(accuracy_imdb_smooth1, axis = 0))
    std_imdb.append(np.std(accuracy_imdb_smooth1, axis = 0))
 

print(accuracy_amazon_m1)
print(accuracy_amazon_m2)

mean_amazon_m1 = []
std_amazon_m1 = []
for i in range(len(accuracy_amazon_m1)):
    accuracy_amazon_1 = np.array(accuracy_amazon_m1[i]).astype(np.float)
    mean_amazon_m1.append(np.mean(accuracy_amazon_1 , axis = 0))
    std_amazon_m1.append(np.std(accuracy_amazon_1, axis = 0))

mean_amazon_m2 = []
std_amazon_m2 = []
for i in range(len(accuracy_amazon_m2)):
    accuracy_amazon_2 = np.array(accuracy_amazon_m2[i]).astype(np.float)
    mean_amazon_m2.append(np.mean(accuracy_amazon_2, axis = 0))
    std_amazon_m2.append(np.std(accuracy_amazon_2, axis = 0))
    
    
print(accuracy_yelp_m1)
print(accuracy_yelp_m2)

mean_yelp_m1 = []
std_yelp_m1 = []
for i in range(len(accuracy_yelp_m1)):
    accuracy_yelp_1 = np.array(accuracy_yelp_m1[i]).astype(np.float)
    mean_yelp_m1.append(np.mean(accuracy_yelp_1 , axis = 0))
    std_yelp_m1.append(np.std(accuracy_yelp_1, axis = 0))

mean_yelp_m2 = []
std_yelp_m2 = []
for i in range(len(accuracy_yelp_m2)):
    accuracy_yelp_2 = np.array(accuracy_yelp_m2[i]).astype(np.float)
    mean_yelp_m2.append(np.mean(accuracy_yelp_2, axis = 0))
    std_yelp_m2.append(np.std(accuracy_yelp_2, axis = 0))  
    
    

print(accuracy_imdb_m1)
print(accuracy_imdb_m2)

mean_imdb_m1 = []
std_imdb_m1 = []
for i in range(len(accuracy_imdb_m1)):
    accuracy_imdb_1 = np.array(accuracy_imdb_m1[i]).astype(np.float)
    mean_imdb_m1.append(np.mean(accuracy_imdb_1 , axis = 0))
    std_imdb_m1.append(np.std(accuracy_imdb_1, axis = 0))

mean_imdb_m2 = []
std_imdb_m2 = []
for i in range(len(accuracy_imdb_m2)):
    accuracy_imdb_2 = np.array(accuracy_imdb_m2[i]).astype(np.float)
    mean_imdb_m2.append(np.mean(accuracy_imdb_2, axis = 0))
    std_imdb_m2.append(np.std(accuracy_imdb_2, axis = 0))


data = [0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9,1.0]
plt.figure()
plt.plot(data,  mean_amazon_no, 'r-', label = "nosmooth accuracy as a function of data size of amazon")
plt.errorbar(data, mean_amazon_no, yerr = std_amazon_no, linestyle = "None")
plt.plot(data,  mean_amazon,'k-', label = "smooth accuracy as a function of data size of amazon")
plt.errorbar(data, mean_amazon, yerr = std_amazon, linestyle = "None")
plt.xlim(0.0,1.2)
plt.ylim((0.0, 1.1))
plt.legend()
plt.title("learning Rate Curve")
plt.ylabel("Accuracy")
plt.xlabel("data size")
plt.savefig('graph1.png')

plt.figure()
plt.plot(data,  mean_yelp_no, 'r-', label = "nosmooth accuracy as a function of data size of yelp")
plt.errorbar(data, mean_yelp_no, yerr = std_yelp_no, linestyle = "None")
plt.plot(data,  mean_yelp,'k-', label = "smooth accuracy as a function of data size of yelp")
plt.errorbar(data, mean_yelp, yerr = std_yelp, linestyle = "None")
plt.xlim(0.0,1.2)
plt.ylim((0.0, 1.1))
plt.legend()
plt.title("learning Rate Curve")
plt.ylabel("Accuracy")
plt.xlabel("data size")
plt.savefig('graph2.png')


plt.figure()
plt.plot(data,  mean_imdb_no, 'r-', label = "nosmooth accuracy as a function of data size of imdb")
plt.errorbar(data, mean_imdb_no, yerr = std_imdb_no, linestyle = "None")
plt.plot(data,  mean_imdb,'k-', label = "smooth accuracy as a function of data size of imdb")
plt.errorbar(data, mean_imdb, yerr = std_imdb, linestyle = "None")
plt.xlim(0.0,1.2)
plt.ylim((0.0, 1.1))
plt.legend()
plt.title("learning Rate Curve")
plt.ylabel("Accuracy")
plt.xlabel("data size")
plt.savefig('graph3.png')



data1 = [0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
plt.figure()
plt.plot(data1,  mean_amazon_m1, 'r-', label = "accuracy as a function of change of m of amazon")
plt.errorbar(data1, mean_amazon_m1, yerr = std_amazon_m1, linestyle = "None")
plt.plot(data1,  mean_yelp_m1, 'g-', label = "accuracy as a function of change of m of yelp")
plt.errorbar(data1, mean_yelp_m1, yerr = std_yelp_m1, linestyle = "None")
plt.plot(data1,  mean_imdb_m1, 'k-', label = "accuracy as a function of change of m of imdb")
plt.errorbar(data1, mean_imdb_m1, yerr = std_imdb_m1, linestyle = "None")


plt.xlim(0.0,1.0)
plt.ylim((0.60, 0.9))
plt.legend()
plt.title("Cross Validation Performance")
plt.ylabel("Accuracy")
plt.xlabel("the number of m")
plt.savefig('graph4.png')

data2 = [1,2,3,4,5,6,7,8,9,10]
plt.figure()
plt.plot(data2,  mean_amazon_m2,'r-', label = "accuracy as a function of change of m of amazon")
plt.errorbar(data2, mean_amazon_m2, yerr = std_amazon_m2, linestyle = "None")
plt.plot(data2,  mean_yelp_m2,'g-', label = "accuracy as a function of change of m of yelp")
plt.errorbar(data2, mean_yelp_m2, yerr = std_yelp_m2, linestyle = "None")


plt.plot(data2,  mean_imdb_m2,'k-', label = "accuracy as a function of change of m of imdb")
plt.errorbar(data2, mean_imdb_m2, yerr = std_imdb_m2, linestyle = "None")
plt.xlim(0,11)
plt.ylim((0.65, 0.90))
plt.legend()
plt.title("Cross Validation Performance")
plt.ylabel("Accuracy")
plt.xlabel("the number of m")
plt.savefig('graph5.png')
