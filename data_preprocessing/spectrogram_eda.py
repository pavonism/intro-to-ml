import torch
import scipy.ndimage
import os
import numpy as np 
import matplotlib.pyplot as plt 


def get_labels():
    labels = os.listdir(f'../data/tsrc_spectrograms/train')
    return labels
def per_class_count(dataset_path , set):
    labels = os.listdir(f'{dataset_path}/{set}')

    res = {}
    for word in labels:
        tmp = os.listdir(f'{dataset_path}/{set}/{word}')
        res.update({word : len(tmp)})
    
    return res

def per_class_count_y_true(y_true):
    res = [0 for i in range(30)]

    for i in range(len(y_true)):
        res[y_true[i]] += 1

    return res


def per_class_acc(y_true , y_pred):
    tab = [[0 for i in range(2)] for j in range(30)]
    for i in range(0 , len(y_true)):
    
        true = y_true[i]
        pred = y_pred[i]
        if true == pred:
            tab[true][0] = tab[true][0] + 1
        else:
            tab[true][1] = tab[true][1] + 1


    return tab

def count_acc(y_true , y_pred):
    tab = per_class_acc(y_true , y_pred)

    res = [0 for i in range(30)]
    for i in range(len(tab)):
        sum = tab[i][0] + tab[i][1]
        res[i] = tab[i][0] / sum

    return res

def class_to_class(y_true , y_pred , class1 , class2):
    classes = get_labels()
    in1 = classes.index(class1)
    in2 = classes.index(class2)

    res = [0 for i in range(4)]

    for i in range(len(y_true)):
        if(y_true[i] == in1):
            if(y_pred[i] == in1):
                res[0] += 1
            if(y_pred[i] == in2):
                res[1] += 1
        elif(y_true[i] == in2):
            if(y_pred[i] == in2):
                res[2] += 1
            if(y_pred[i] == in1):
                res[3] += 1

    return(res)

def plot_acc_per_class(y_true , y_pred , y_true_clean , y_pred_clean):
    acc = count_acc(y_true , y_pred)

    acc_clean = count_acc(y_true_clean , y_pred_clean)

    barWidth = 0.25
    fig = plt.subplots(figsize =(20, 8)) 

    br1 = np.arange(len(acc)) 
    br2 = [x + barWidth for x in br1] 

    plt.bar(br1, acc, color ='darkred', width = barWidth, 
            edgecolor ='grey', label ='Normal') 
    plt.bar(br2, acc_clean, color ='darkblue', width = barWidth, 
            edgecolor ='grey', label ='Clean') 

    plt.xlabel('Word', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Accurency', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + barWidth/2 for r in range(len(acc))], 
            get_labels())

    plt.legend()
    plt.show()

def class_to_class_acc(y_true , y_pred , class1 , class2):
    res = class_to_class(y_true , y_pred , class1 , class2)

    res_acc = [0 for i in range(4)]

    classes = get_labels()
    in1 = classes.index(class1)
    in2 = classes.index(class2)

    count = per_class_count_y_true(y_true)

    sum1 = count[in1]
    sum2 = count[in2]
                 
    for i in range(2):
        res_acc[i] = res[i] / sum1
    for i in range(2, 4):
        res_acc[i] = res[i] / sum2

                
    return(res_acc)

def plot_clc_acc_per_class(y_true , y_pred , y_true_clean , y_pred_clean, class1 , class2):
    res_acc = class_to_class_acc(y_true , y_pred , class1 , class2)
    res_clean_acc = class_to_class_acc(y_true_clean , y_pred_clean , class1 , class2)


    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8)) 

    br1 = np.arange(len(res_acc)) 
    br2 = [x + barWidth for x in br1] 

    plt.bar(br1, res_acc, color ='darkred', width = barWidth, 
            edgecolor ='grey', label ='Normal') 
    plt.bar(br2, res_clean_acc, color ='darkblue', width = barWidth, 
            edgecolor ='grey', label ='Clean') 

    plt.xlabel('Word', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Accurency', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + barWidth/2 for r in range(len(res_acc))], 
            [f"{class1}->{class1}" , f"{class1}->{class2}" , f"{class2}->{class2}" , f"{class2}->{class1}"])

    plt.legend()
    plt.show()

def plot_count(y_true , y_true_clean):
    count = per_class_count_y_true(y_true)

    count_clean = per_class_count_y_true(y_true_clean)

    barWidth = 0.25
    fig = plt.subplots(figsize =(20, 8)) 

    br1 = np.arange(len(count)) 
    br2 = [x + barWidth for x in br1] 

    plt.bar(br1, count, color ='darkred', width = barWidth, 
            edgecolor ='grey', label ='Normal') 
    plt.bar(br2, count_clean, color ='darkblue', width = barWidth, 
            edgecolor ='grey', label ='Clean') 

    plt.xlabel('Word', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Accurency', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + barWidth/2 for r in range(len(count_clean))], 
            get_labels())

    plt.legend()
    plt.show()

