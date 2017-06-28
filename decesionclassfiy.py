#coding :utf-8
'''
2017.6.25 author :Erin 
          function: "decesion tree" ID3
          
'''
import numpy as np
import pandas as pd
from math import log
import operator  
import random
def load_data():
    red = [line.strip().split(';') for line in open('e:/a/winequality-red.csv')]
    white = [line.strip().split(';') for line in open('e:/a/winequality-white.csv')]
    data=red+white
    random.shuffle(data)  #����data
    x_train=data[:800]
    x_test=data[800:]
    
    features=['fixed','volatile','citric','residual','chlorides','free','total','density','pH','sulphates','alcohol','quality']
    return x_train,x_test,features

def cal_entropy(dataSet):
 
    
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        label = featVec[-1]
        if label not in labelCounts.keys():
            labelCounts[label] = 0
        labelCounts[label] += 1
    entropy = 0.0
    for key in labelCounts.keys():
        p_i = float(labelCounts[key]/numEntries)
        entropy -= p_i * log(p_i,2)#log(x,10)��ʾ��10 Ϊ�׵Ķ���
    return entropy

def split_data(data,feature_index,value):
    '''
    �������ݼ�
    feature_index�����ڻ������������������硰���䡱
    value:���ֺ������ֵ�����硰�����ꡱ
    '''
    data_split=[]#���ֺ�����ݼ�
    for feature in data:
        if feature[feature_index]==value:
            reFeature=feature[:feature_index]
            reFeature.extend(feature[feature_index+1:])
            data_split.append(reFeature)
    return data_split
def choose_best_to_split(data):
    
    '''
    ����ÿ����������Ϣ���棬ѡ�����Ļ������ݼ�����������
    '''
    
    count_feature=len(data[0])-1#��������4
    #print(count_feature)#4
    entropy=cal_entropy(data)#ԭ�����ܵ���Ϣ��
    #print(entropy)#0.9402859586706309
    
    max_info_gain=0.0#��Ϣ�������
    split_fea_index = -1#��Ϣ������󣬶�Ӧ��������

    for i in range(count_feature):
        
        feature_list=[fe_index[i] for fe_index in data]#��ȡ������������ֵ
        #######################################
        
       # print(feature_list)
        unqval=set(feature_list)#ȥ���ظ�
        Pro_entropy=0.0#��������
        for value in unqval:#�����������µ���������
            sub_data=split_data(data,i,value)
            pro=len(sub_data)/float(len(data))
            Pro_entropy+=pro*cal_entropy(sub_data)
            #print(Pro_entropy)
            
        info_gain=entropy-Pro_entropy
        if(info_gain>max_info_gain):
            max_info_gain=info_gain
            split_fea_index=i
    return split_fea_index
        
        
##################################################
def most_occur_label(labels):
    #sorted_label_count[0][0]  �����������ǩ
    label_count={}
    for label in labels:
        if label not in label_count.keys():
            label_count[label]=0
        else:
            label_count[label]+=1
        sorted_label_count = sorted(label_count.items(),key = operator.itemgetter(1),reverse = True)
    return sorted_label_count[0][0]
def build_decesion_tree(dataSet,featnames):
    '''
    �ֵ�ļ���Žڵ���Ϣ����֧��Ҷ�ӽڵ���ֵ
    '''
    featname = featnames[:]              ################
    classlist = [featvec[-1] for featvec in dataSet]  #�˽ڵ�ķ������
    if classlist.count(classlist[0]) == len(classlist):  #ȫ������һ��
        return classlist[0]
    if len(dataSet[0]) == 1:         #������,û��������
        return Vote(classlist)       #�������Ӷ���
    # ѡ��һ�������������л���
    bestFeat = choose_best_to_split(dataSet)
    bestFeatname = featname[bestFeat]
    del(featname[bestFeat])     #��ֹ�±겻׼
    DecisionTree = {bestFeatname:{}}
    # ������֧,���ҳ���������ֵ,����֧��
    allvalue = [vec[bestFeat] for vec in dataSet]
    specvalue = sorted(list(set(allvalue)))  #ʹ��һ��˳��
    for v in specvalue:
        copyfeatname = featname[:]
        DecisionTree[bestFeatname][v] =  build_decesion_tree(split_data(dataSet,bestFeat,v),copyfeatname)
    return DecisionTree

def classify(Tree, featnames, X):
    classLabel=''
    root = list(Tree.keys())[0]
    firstDict = Tree[root]
    featindex = featnames.index(root)  #���ڵ�������±�
    #classLabel='0'
    for key in firstDict.keys():   #�����Ե�ȡֵ,ȡ�ĸ��������Ŀ�����
        if X[featindex] == key:
            if type(firstDict[key]) == type({}):
                classLabel = classify(firstDict[key],featnames,X)
            else:
                classLabel = firstDict[key]
    return classLabel



            
    
if __name__ == '__main__':
    x_train,x_test,features=load_data()
    split_fea_index=choose_best_to_split(x_train)
    newtree=build_decesion_tree(x_train,features)
    
    count=0
    for test in x_test:
        label=classify(newtree, features,test)
        
        if(label==test[-1]):
            count=count+1
    acucy=float(count/len(x_test))
    print(acucy)
    
    
