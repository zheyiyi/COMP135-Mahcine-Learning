# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:16:48 2017

@author: zheyiyi
"""
import math
import random
import collections
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# read_data can get all data to be stored in dict and the number of class
#dict[number] = data, the output is  

def read_data(file_name):
    fd = open(file_name, "r")
    new_content = ""
    k = 0
    dict_t = {}
   
    while len(new_content) == 0 or (new_content[0].upper() != '@DATA'):
       content = fd.readline()
       new_content= content.split()
       if len(new_content) == 3:
           if new_content[1] == "class":
               new_content_1 = new_content[2][1:-1]
       
               class_list = new_content_1.split(",")
           
               k = len(class_list)
               
               if not class_list[1].isdigit():
                   for num,elm in enumerate(class_list):
                       dict_t[elm] = num
                 
    #print(dict_t)
    
    dict = {}
    dict_real = collections.defaultdict(list)

    number = 0
    for line in fd:
        line = line.strip()
        words = line.split(",")
        
        
        dict[number] = list(map(float, words[:-1]))
        if words[-1].isdigit():
            
            dict_real[int(words[-1]) - 1].append(list(map(float, words[:-1])))
        else:
            
            dict_real[dict_t[words[-1]]].append(list(map(float, words[:-1])))
        
            
        number += 1

    length = len(dict)
    fd.close()
    return k, dict, length, dict_real




# choose k random data and store them into list
def random_class(data_hash):
    l = data_hash[2]
    d = data_hash[1]
    #result = []
    k = data_hash[0]
    #sequence = [i for i in range(l)]
    #num_list= random.sample(sequence, k)
    a = set()
    while len(a) != k:        
        num = random.randrange(0,l)
        a.add(tuple(d[num]))
    return list(a)
    

def smart_class(data_hash):
    max_num = - sys.maxsize
    max_value = - sys.maxsize
    k = data_hash[0]
    result = []
    new_data_hash = data_hash[1]
    new_data_set = list(new_data_hash.values())
    random.shuffle(new_data_set)
    initial_data = new_data_set[0]
    result.append(initial_data)
    semi_result = []    
    sum = 0
    
    set_1 = new_data_set[1:11]
    for i in range(len(set_1)):
        for j in range(len(set_1[i])):
           sum += (set_1[i][j] - initial_data[j]) ** 2   
        semi_result.append((sum, set_1[i]))
    
    for num, value in semi_result:
        if max_num < num:
            max_num = num
            max_value = value
            
    result.append(max_value)
    
    n = k - 2
    for i in range(n):
        a = (i + 1) * 10 + 1
        b = (i + 2) * 10 + 1
        new_data = cal_center(result, new_data_set[a:b])
        result.append(new_data)
    
    return result
    

    
def cal_center(list_data, data_set):
    max_distance = - sys.maxsize
    max_data = - sys.maxsize
    result = []
    for data in data_set:
        cloest = []   
        for point in list_data:
            sum = 0
            for i in range(len(point)):
                sum += (point[i] - data[i]) ** 2 
            cloest.append(sum)
        result.append((min(cloest),data))
        
    for c, d in result:
        if max_distance < c:
            max_distance = c
            max_data = d
    
    return max_data
        
    
                   


    
"""
calculate and cal_distance 
"""     
# seprate all datas into the clusters their belong to respectively
def calculate(clusters, data_hash):
    cluster = collections.defaultdict(list)
    l = len(data_hash)
    for data in data_hash.values():
        
        cluster[tuple(cal_distance(data, clusters))].append(data)
    #print(set(clusters))
            
    temp = []
    for c in clusters:
        temp.append(tuple(c))

    new_cluster ={}
    for c in clusters:
        #print(tuple(c))
        if tuple(c) not in cluster:
           # print(tuple(c))
            num = random.randrange(0,l)
            new_cluster[tuple(c)] = data_hash[num]
    #print(len(cluster))
    #print(len(new_cluster))
    return cluster

# calculaten the smallest distance between one point and all centers
def cal_distance(point, clusters):
    result = []
    min_num = sys.maxsize
    min_value = sys.maxsize
    for cluster in clusters:
        sum = 0
        for i in range(len(point)):
            sum += (point[i] - cluster[i]) ** 2   
        result.append((sum, cluster))
    
    for num, value in result:
        if min_num > num:
            min_num = num
            min_value = value
        
    return  min_value
  
  
def calculation_cd(dict): 
    sum_total = 0
    for center in dict.keys():
       new_center = np.array(center)
       sum_c = 0
       for x in dict[center]:
           new_x = np.array(x)
           sum_c += np.sum((new_x - new_center) ** 2)
       sum_total += sum_c
    return sum_total
           
# update center point based on new clusters
def update_center(dict):
    result = []
    for center in dict.keys():
       c = len(dict[center])
       x = np.array(dict[center])
       #print(x)
       sum_x = np.sum(x,axis = 0)
       new_center = sum_x / c
       result.append(new_center)
    
    return result 

def cal_nml(dict_cal, dict_real):
   # print(dict_cal.keys())
    new_dict_u = {}
    # from list as key to number(1,2,3,4,5) as key
    new_dict_cal = collections.defaultdict(list)
    for i, key in enumerate(dict_cal):
        new_dict_cal[i] = dict_cal[key]
    
    for i in range(len(new_dict_cal)):
       u = new_dict_cal[i]
       for j in dict_real:
          v = dict_real[j]
          count = 0
          for elm in u:         
              if elm in v:
                 count += 1
          new_dict_u[(i,j)] = count
   # print(new_dict_u)
    #print(new_dict_u.keys())
    
  
    a = [0] * len(dict_cal) 
   # print(len(dict_cal))
    for i in range(len(new_dict_cal)):
        a[i] = sum([new_dict_u[(x,y)] for x,y in new_dict_u if x == i]) 
    
    b = [0] * len(dict_real) 
    for i in range(len(dict_real)):
        b[i] = sum([new_dict_u[(x,y)] for x,y, in new_dict_u if y == i ])        
    
    N = (sum(a) + sum(b)) / 2
    
    H_u = sum([- (elm / N) * math.log(elm / N) if elm != 0 else 0 for elm in a ])
    #H_uv = sum([- (new_dict_u[x,y] / N) * math.log(new_dict_u[x,y] / N) if new_dict_u[x,y] != 0 else 0 for x,y in new_dict_u ] )
    
    #H_u_v = sum([- (new_dict_u[x,y] / N) * math.log((new_dict_u[x,y] / N)/(b[y]/ N)) if new_dict_u[x,y] != 0 and b[y] != 0 else 0 for x,y in new_dict_u ])
    I_u_v = sum([(new_dict_u[x,y] / N) * math.log((new_dict_u[x,y] * N)/(b[y] * a[x])) if new_dict_u[x,y] != 0 and b[y] * a[x] != 0 else 0 for x,y in new_dict_u ])
    
 
    H_v = sum([- (elm / N) * math.log(elm / N) if elm != 0 else 0 for elm in b ])
    nmi = 2 * I_u_v / (H_u + H_v)
    return nmi
   
 
def cal_nml_total(filename):
    result = []
    data_hash = read_data(filename)

    for i in range(10):
       
        center_point = random_class(data_hash)       
        cluster_data = calculate(center_point, data_hash[1])
        new_cluster_data = update_center(cluster_data)
        count = 0
    
        while new_cluster_data == cluster_data != True:
           if count % 2 == 0:
              cluster_data = update_center(new_cluster_data)
              count += 1
           else:
              new_cluster_data = update_center(cluster_data)
              count += 1
    
        clusters_f = calculate(new_cluster_data, data_hash[1])
        #print(clusters_f.keys())
        cd_value = calculation_cd(clusters_f)
        nml_value = cal_nml(clusters_f,data_hash[3])        
        
        result.append((cd_value, nml_value))
    return result

def smart_cal_nml_total(filename):
    data_hash = read_data(filename)
    center_point = smart_class(data_hash)
    cluster_data = calculate(center_point, data_hash[1])
    new_cluster_data = update_center(cluster_data)
    count = 0
    
    while new_cluster_data == cluster_data != True:
           if count % 2 == 0:
              cluster_data = update_center(new_cluster_data)
              count += 1
           else:
              new_cluster_data = update_center(cluster_data)
              count += 1
    
    clusters_f = calculate(new_cluster_data, data_hash[1])
        #print(clusters_f.keys())
    cd_value = calculation_cd(clusters_f)
    nml_value = cal_nml(clusters_f,data_hash[3])
    return cd_value, nml_value
    


random_data_0 = cal_nml_total("artdata0.5.arff") 
smart_data_0 = smart_cal_nml_total("artdata0.5.arff")   

random_data_0_cs = [x for x,y in random_data_0]
random_data_0_cs.append(smart_data_0[0])
random_data_0_nml = [y for x,y in random_data_0]
random_data_0_nml.append(smart_data_0[1])

plt.figure()
n = 11
X = np.arange(n)+ 1
y1 = random_data_0_cs
y2 = random_data_0_nml
plt.bar(X,y1,width = 0.3,facecolor = 'lightskyblue',edgecolor = 'white')
plt.ylim(0,4000)
plt.xlim(0,12)
plt.legend(("the cs of artdata0.5.arff",))
plt.savefig('graph1.png')
plt.show()

plt.figure()
n = 11
X = np.arange(n)+ 1

y2 = random_data_0_nml
plt.bar(X,y2,width = 0.3,facecolor = 'yellowgreen',edgecolor = 'white')
plt.ylim(0,1.2)
plt.xlim(0,12)
plt.legend(("the nmi of artdata0.5.arff",))
plt.savefig('graph2.png')
plt.show()


random_data_1 = cal_nml_total("artdata1.arff") 
smart_data_1 = smart_cal_nml_total("artdata1.arff")  

random_data_1_cs = [x for x,y in random_data_1]
random_data_1_cs.append(smart_data_1[0])
random_data_1_nml = [y for x,y in random_data_1]
random_data_1_nml.append(smart_data_1[1])

plt.figure()
n = 11
X = np.arange(n)+ 1
y1 = random_data_1_cs

plt.bar(X,y1,width = 0.3,facecolor = 'lightskyblue',edgecolor = 'white')
plt.ylim(0,5000)
plt.xlim(0,12)
plt.legend(("the cs of artdata1.arff",))
plt.savefig('graph3.png')
plt.show()

plt.figure()
n = 11
X = np.arange(n)+ 1

y2 = random_data_1_nml
plt.bar(X,y2,width = 0.3,facecolor = 'yellowgreen',edgecolor = 'white')
plt.ylim(0,1.2)
plt.xlim(0,12)
plt.legend(("the nmi of artdata1.arff",))
plt.savefig('graph4.png')
plt.show()

random_data_2 = cal_nml_total("artdata2.arff") 
smart_data_2 = smart_cal_nml_total("artdata2.arff")  


random_data_2_cs = [x for x,y in random_data_2]
random_data_2_cs.append(smart_data_2[0])
random_data_2_nml = [y for x,y in random_data_2]
random_data_2_nml.append(smart_data_2[1])

plt.figure()
n = 11
X = np.arange(n)+ 1
y1 = random_data_2_cs

plt.bar(X,y1,width = 0.3,facecolor = 'lightskyblue',edgecolor = 'white')
plt.ylim(0,10000)
plt.xlim(0,12)
plt.legend(("the cs of artdata2.arff",))
plt.savefig('graph5.png')
plt.show()

plt.figure()
n = 11
X = np.arange(n)+ 1

y2 = random_data_2_nml
plt.bar(X,y2,width = 0.3,facecolor = 'yellowgreen',edgecolor = 'white')
plt.ylim(0,1.2)
plt.xlim(0,12)
plt.legend(("the nmi of artdata2.arff",))
plt.savefig('graph6.png')
plt.show()

random_data_3 = cal_nml_total("artdata3.arff") 
smart_data_3 = smart_cal_nml_total("artdata3.arff") 


random_data_3_cs = [x for x,y in random_data_3]
random_data_3_cs.append(smart_data_3[0])
random_data_3_nml = [y for x,y in random_data_3]
random_data_3_nml.append(smart_data_3[1])

plt.figure()
n = 11
X = np.arange(n)+ 1
y1 = random_data_3_cs

plt.bar(X,y1,width = 0.3,facecolor = 'lightskyblue',edgecolor = 'white')
plt.ylim(0,20000)
plt.xlim(0,12)
plt.legend(("the cs of artdata3.arff",))
plt.savefig('graph7.png')
plt.show()

plt.figure()
n = 11
X = np.arange(n)+ 1

y2 = random_data_3_nml
plt.bar(X,y2,width = 0.3,facecolor = 'yellowgreen',edgecolor = 'white')
plt.ylim(0,1.2)
plt.xlim(0,12)
plt.legend(("the nmi of artdata3.arff",))
plt.savefig('graph8.png')
plt.show() 

random_data_4 = cal_nml_total("artdata4.arff") 
smart_data_4 = smart_cal_nml_total("artdata4.arff")


random_data_4_cs = [x for x,y in random_data_4]
random_data_4_cs.append(smart_data_4[0])
random_data_4_nml = [y for x,y in random_data_4]
random_data_4_nml.append(smart_data_4[1])

plt.figure()
n = 11
X = np.arange(n)+ 1
y1 = random_data_4_cs

plt.bar(X,y1,width = 0.3,facecolor = 'lightskyblue',edgecolor = 'white')
plt.ylim(0,25000)
plt.xlim(0,12)
plt.legend(("the cs of artdata4.arff",))
plt.savefig('graph9.png')
plt.show()

plt.figure()
n = 11
X = np.arange(n)+ 1

y2 = random_data_4_nml
plt.bar(X,y2,width = 0.3,facecolor = 'yellowgreen',edgecolor = 'white')
plt.ylim(0,1.2)
plt.xlim(0,12)
plt.legend(("the cs of artdata4.arff",))
plt.savefig('graph10.png')
plt.show()  

random_data_io = cal_nml_total("ionosphere.arff") 
smart_data_io = smart_cal_nml_total("ionosphere.arff") 


random_data_io_cs = [x for x,y in random_data_io]
random_data_io_cs.append(smart_data_io[0])
random_data_io_nml = [y for x,y in random_data_io]
random_data_io_nml.append(smart_data_io[1])

plt.figure()
n = 11
X = np.arange(n)+ 1
y1 = random_data_io_cs

plt.bar(X,y1,width = 0.3,facecolor = 'lightskyblue',edgecolor = 'white')
plt.ylim(0,20000)
plt.xlim(0,12)
plt.legend(("the cs of ionosphere.arff",))
plt.savefig('graph11.png')
plt.show()

plt.figure()
n = 11
X = np.arange(n)+ 1

y2 = random_data_io_nml
plt.bar(X,y2,width = 0.3,facecolor = 'yellowgreen',edgecolor = 'white')
plt.ylim(0,0.5)
plt.xlim(0,12)
plt.legend(("the nmi of ionosphere.arff",))
plt.savefig('graph12.png')
plt.show()   

random_data_ir = cal_nml_total("iris.arff") 
smart_data_ir = smart_cal_nml_total("iris.arff")  

random_data_ir_cs = [x for x,y in random_data_ir]
random_data_ir_cs.append(smart_data_ir[0])
random_data_ir_nml = [y for x,y in random_data_ir]
random_data_ir_nml.append(smart_data_ir[1])

plt.figure()
n = 11
X = np.arange(n)+ 1
y1 = random_data_ir_cs

plt.bar(X,y1,width = 0.3,facecolor = 'lightskyblue',edgecolor = 'white')
plt.ylim(0,1000)
plt.xlim(0,12)
plt.legend(("the cs of iris.arff",))
plt.savefig('graph13.png')
plt.show()

plt.figure()
n = 11
X = np.arange(n)+ 1

y2 = random_data_ir_nml
plt.bar(X,y2,width = 0.3,facecolor = 'yellowgreen',edgecolor = 'white')
plt.ylim(0,1.2)
plt.xlim(0,12)
plt.legend(("the nmi of iris.arff",))
plt.savefig('graph14.png')
plt.show()   

random_data_so = cal_nml_total("soybean-processed.arff") 
smart_data_so = smart_cal_nml_total("soybean-processed.arff")  

random_data_so_cs = [x for x,y in random_data_so]
random_data_so_cs.append(smart_data_so[0])
random_data_so_nml = [y for x,y in random_data_so]
random_data_so_nml.append(smart_data_so[1])

plt.figure()
n = 11
X = np.arange(n)+ 1
y1 = random_data_so_cs

plt.bar(X,y1,width = 0.3,facecolor = 'lightskyblue',edgecolor = 'white')
plt.ylim(0,5000)
plt.xlim(0,12)
plt.legend(("the cs of soybean-processed.arff",))
plt.savefig('graph15.png')
plt.show()

plt.figure()
n = 11
X = np.arange(n)+ 1

y2 = random_data_so_nml
plt.bar(X,y2,width = 0.3,facecolor = 'yellowgreen',edgecolor = 'white')
plt.ylim(0,1.2)
plt.xlim(0,12)
plt.legend(("the nmi of soybean-processed.arff",))
plt.savefig('graph16.png')
plt.show()   





#data_set = [, 1, 2, 3, 4, io, ir, ]


"""

choose k  [2...22]
    
"""



def read_data_k(file_name):
    fd = open(file_name, "r")
    new_content = ""

    dict_t = {}
    while len(new_content) == 0 or (new_content[0].upper() != '@DATA'):
       content = fd.readline()
       new_content= content.split()
       if len(new_content) == 3:
           if new_content[1] == "class":
               new_content_1 = new_content[2][1:-1]
       
               class_list = new_content_1.split(",")
                         
               if not class_list[1].isdigit():
                   for num,elm in enumerate(class_list):
                       dict_t[elm] = num                
  
    dict = {}
    dict_real = collections.defaultdict(list)
    
    number = 0
    for line in fd:
        line = line.strip()
        words = line.split(",")
        
        dict[number] = list(map(float, words[:-1]))
        if words[-1].isdigit():
            
            dict_real[int(words[-1]) - 1].append(list(map(float, words[:-1])))
        else:
            
            dict_real[dict_t[words[-1]]].append(list(map(float, words[:-1])))
        
            
        number += 1
    length = len(dict)
    fd.close()
    return dict, length, dict_real
    
def random_class_k(k,data_hash):
    l = data_hash[1]
    d = data_hash[0] 
    a = set()
    while len(a) != k:        
        num = random.randrange(0,l)
        a.add(tuple(d[num]))
    
    return list(a)

def cal_nml_k(dict_cal, dict_real):
    
    new_dict_u = {}
    # from list as key to number(1,2,3,4,5) as key
    new_dict_cal = collections.defaultdict(list)
    for i, key in enumerate(dict_cal):
        new_dict_cal[i] = dict_cal[key]
    
    for i in range(len(new_dict_cal)):
       u = new_dict_cal[i]
       for j in dict_real:
          v = dict_real[j]
          count = 0
          for elm in u:         
              if elm in v:
                 count += 1
          new_dict_u[(i,j)] = count
    #print(new_dict_u.keys())
    
  
    a = [0] * len(dict_cal) 
   # print(len(dict_cal))
    for i in range(len(new_dict_cal)):
        a[i] = sum([new_dict_u[(x,y)] for x,y in new_dict_u if x == i]) 
         
    b = [0] * len(dict_real) 
    for i in range(len(dict_real)):
        b[i] = sum([new_dict_u[(x,y)] for x,y, in new_dict_u if y == i])
 
        
    
    N = sum(a) + sum(b)
    H_u = sum([- (elm / N) * math.log(elm / N) if elm != 0 else 0 for elm in a ])
    H_v = sum([- (elm / N) * math.log(elm / N) if elm != 0 else 0 for elm in b ])
    #H_uv = sum([- (new_dict_u[x,y] / N) * math.log(new_dict_u[x,y] / N) if new_dict_u[x,y] != 0 else 0 for x,y in new_dict_u ] )
    
    #H_u_v = sum([- (new_dict_u[x,y] / N) * math.log((new_dict_u[x,y] / N)/(b[y]/ N)) if new_dict_u[x,y] != 0 and b[y] != 0 else 0 for x,y in new_dict_u ])
    I_u_v = sum([(new_dict_u[(x,y)] / N) * math.log((new_dict_u[(x,y)] / N)/((b[y] * a[x]/ N) / N)) if new_dict_u[x,y] != 0 and b[y] * a[x] != 0 else 0 for x,y in new_dict_u ])
   
    nmi = 2 * I_u_v / (H_u + H_v)
    return nmi   
#  calculate k from 2 to 23， for each k,  use random_class_k to get k center point
# cluster all points based on k center point, and then update k point, until all 
#k point doesn;t change, and clculate cs value and get the smallest value
def k_cal_cd_nml(filename):
    result = []
    data_hash = read_data_k(filename)

    for k in range(2,23):
        cd_min_value = sys.maxsize
        for i in range(10):
           
            center_point = random_class_k(k,data_hash)
            cluster_data = calculate(center_point, data_hash[0])
            new_cluster_data = update_center(cluster_data)
            count = 0
        
            while new_cluster_data == cluster_data != True:
               if count % 2 == 0:
                  cluster_data = update_center(new_cluster_data)
                  count += 1
               else:
                  new_cluster_data = update_center(cluster_data)
                  count += 1
        
            clusters_f = calculate(new_cluster_data, data_hash[0])
            #print(clusters_f.keys())
            
            cd_value = calculation_cd(clusters_f)
            
            
           # nml_value = cal_nml_k(clusters_f,data_hash[2])
           
            
            #nml_min_value = sys.maxsize
            
            if cd_min_value > cd_value:
                cd_min_value = cd_value
               # nml_min_value = nml_value
            
            
        result.append(cd_min_value)
    return result
  

random_data_0_k = k_cal_cd_nml("artdata0.5.arff") 

random_data_1_k = k_cal_cd_nml("artdata1.arff")

random_data_2_k = k_cal_cd_nml("artdata2.arff") 
 
random_data_3_k = k_cal_cd_nml("artdata3.arff") 

random_data_4_k = k_cal_cd_nml("artdata4.arff") 

random_data_io_k = k_cal_cd_nml("ionosphere.arff") 

random_data_ir_k = k_cal_cd_nml("iris.arff") 
random_data_so_k = k_cal_cd_nml("soybean-processed.arff") 

data = [i for i in range(2,23)]


plt.figure()
plt.plot(data,  random_data_0_k, 'r-', label = "the cs of artdata0.5.arff")
plt.xlim(1,23)
plt.ylim((1000, 3000))
plt.legend()
plt.title("selection for k")
plt.ylabel("the value of cs")
plt.xlabel("the value of k")
plt.savefig('graph17.png')


plt.figure()
plt.plot(data, random_data_1_k, 'r-', label = "the cs of artdata1.arff")
plt.xlim(1,23)
plt.ylim((1000, 4000))
plt.legend()
plt.title("selection for k")
plt.ylabel("the value of cs")
plt.xlabel("the value of k")
plt.savefig('graph18.png')



plt.figure()
plt.plot(data, random_data_2_k, 'r-', label = "the cs of artdata2.arff")
plt.xlim(1,23)
plt.ylim((1000, 8000))
plt.legend()
plt.title("selection for k")
plt.ylabel("the value of cs")
plt.xlabel("the value of k")
plt.savefig('graph19.png')



plt.figure()
plt.plot(data, random_data_3_k, 'r-', label = "the cs of artdata3.arff")
plt.xlim(1,23)
plt.ylim((1000, 15000))
plt.legend()
plt.title("selection for k")
plt.ylabel("the value of cs")
plt.xlabel("the value of k")
plt.savefig('graph20.png')




plt.figure()
plt.plot(data, random_data_4_k, 'r-', label = "the cs of artdata4.arff")
plt.xlim(1,23)
plt.ylim((1000, 25000))
plt.legend()
plt.title("selection for k")
plt.ylabel("the value of cs")
plt.xlabel("the value of k")
plt.savefig('graph21.png')



plt.figure()
plt.plot(data, random_data_io_k, 'r-', label = "the cs of ionosphere.arff")
plt.xlim(1,23)
plt.ylim((5000, 10000))
plt.legend()
plt.title("selection for k")
plt.ylabel("the value of cs")
plt.xlabel("the value of k")
plt.savefig('graph22.png')

plt.figure()
plt.plot(data, random_data_ir_k, 'r-', label = "the cs of iris.arff")
plt.xlim(1,23)
plt.ylim((0, 1000))
plt.legend()
plt.title("selection for k")
plt.ylabel("the value of cs")
plt.xlabel("the value of k")
plt.savefig('graph24.png')


plt.figure()
plt.plot(data, random_data_so_k, 'r-', label = "the cs of soybean-processed.arff")
plt.xlim(1,23)
plt.ylim((4000, 7500))
plt.legend()
plt.title("selection for k")
plt.ylabel("the value of cs")
plt.xlabel("the value of k")
plt.savefig('graph24.png')
