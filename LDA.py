#!/usr/bin/env python3
# -- coding: utf-8 --

import os
import random
import copy 
import csv
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter
import sys

def readFile(fname):
    file = open("20newsgroups/" + fname , 'r');
    for line in file:
        temp = line.split(" ")
    return temp

def readDirectory(directory):
    data = []
    files = []
    for file in os.listdir(directory):
        if file != "index.csv" :
            data.append(readFile(file))
            files.append(file)
    return data, files
            
def preprocess(data):
    words  = []
    for doc in data:
        for word in doc:
            words.append(word)
    cntWords = Counter(words)
    uniqueWords = []
    for uniqueWord in cntWords:
        uniqueWords.append(uniqueWord)
        
    return words ,uniqueWords

def createCSV(lists):
    with open('topicwords.csv', 'w', newline='') as file:
     write = csv.writer(file)
     for i in range(K):
         write.writerow([i ,lists[i]])
         
def optimization(w,r,d,phiTrain):
    phiT = phiTrain.T
    a = 0.01
    I = np.eye(len(phiTrain[0]), dtype = float)
    x = phiT.dot(r)
    x = -x.dot(phiTrain) - I.dot(a)
    H = np.linalg.inv(x)
    g = phiT.dot(d) - w.dot(a)
    Wn = w - H.dot(g)
    return Wn

def LogisticRegression(phiTrain,phiTarget):
    wList = []
    position = []
    timee = []
    w = [0] * int(np.shape(phiTrain[0])[1])
    w = np.array(w)
    for i in range(len(phiTrain)):
        start = time.time()
        for j in range(500):
            phi = phiTrain[i]
            phi = np.array(phi)
            x = phi.dot(w)
            y = 1/(1 + np.exp(-x))
            t = phiTarget[i]
            d = t - y
            r = y.dot(1-y)
            Wn = optimization(w,r,d,phi) 
            temp = w
            w = Wn
            wCon = np.sqrt(np.sum(np.square(w - temp)))/np.sqrt(np.sum(np.square(temp)))
            if wCon <= 0.001 or j == 499: 
                position.append(j)
                end = time.time()
                duration = end - start
                timee.append(duration)
                break
        wList.append(w)
    return wList, position, timee

def generateData(phiTrainData):
    PhiTrainData = []
    PhiTargetData = []
    phiTrain = phiTrainData[:,:-1]
    phiTarget = phiTrainData[:,-1]
    for i in np.arange(0.1,1.1,0.1):
        PhiTrainData.append(phiTrain[:int(len(phiTrain)*i)])
        PhiTargetData.append(phiTarget[:int(len(phiTarget)*i)])
    
    return PhiTrainData, PhiTargetData

def calMeanStd(listt):
    mean = np.mean(np.array(listt),axis=0)
    std = np.std(np.array(listt),axis=0)
    
    return mean,std

def LogisticRegressionTest(phiTestData, wList):   
    error = []
    phiTest = phiTestData[:,:-1]
    phiTestLabel = phiTestData[:,-1]
    for i in range(len(wList)):
        err = 0
        x = phiTest.dot(wList[i])
        y = 1/(1+np.exp(-x))
        yFinal = []
        for j in range(len(y)):
            if y[j] >= 0.5:
                yFinal.append(1)
            else:
                yFinal.append(0)
        for k in range(len(y)):
            if yFinal[k] != phiTestLabel[k]:
                err = err + 1
        error.append(1 - (err/len(phiTestLabel)))
    return error

def logistic(CD ,labels, alpha, method):   
    data = []
    if method == 1: 
        for i in range(D):
            temp = [0] * K
            data.append(temp)
        for i in range(D):
            for j in range(K):
                num = CD[i][j] + alpha
                den = K*alpha + sum(CD[i])
                data[i][j] = num/den
    else : 
        data = CD
    data = np.array(data)
    labels = np.array(labels).reshape(len(labels),1)
    errorList = []
    for i in range(30):
        index = np.random.permutation(len(labels))
        data, labels =  data[index], labels[index]
        data1 = np.concatenate((data, labels), axis = 1)
        phiTrainData = data1[:int(len(data1)*(0.67))]
        phiTestData = data1[int(len(data)*(0.67)):]

        PhiTrainData, PhiTargetData = generateData(phiTrainData)
        wList, position, timee  = LogisticRegression(PhiTrainData, PhiTargetData)
        error = LogisticRegressionTest(phiTestData, wList)
        errorList.append(error)
    
    mean, std = calMeanStd(errorList)
    return mean, std
    
def calCT(K, V, words, uniqueWords, tags):    
    ct = []
    for i in range(K):
        temp = [0] * V
        ct.append(temp)

    for i in range(len(words)):
        for j in range(V):
            if words[i] == uniqueWords[j]:
                for k in range(len(tags)):
                    if topicList[i] == tags[k]: 
                        ct[k][j] += 1
                        break
    return ct

def calCD(D, K, topics, tags): 
    cd = []    
    for i in range(D):
        temp = [0] * K
        cd.append(temp)
    
    for i in range(len(topics)):
        for j in range(len(topics[i])):
            for k in range(len(tags)):
                if topics[i][j] == tags[k]: 
                    cd[i][k] += 1
                    break
    return cd
    
def gibbsSampling(CD, CT, words, uniqueWords, pi, K, doc, topicList):
    iterations = 500
    beta = 0.01
    alpha = 5/K
    prob = [0] * K
    
    for i in range(iterations):
        #print(i)
        for j in range(len(words)):
            index = pi[j]
            currWord = words[index]
            currWordIndex = uniqueWords.index(currWord)
            document = doc[pi[j]]
            topic = topicList[pi[j]]
            CD[document][topic] -= 1
            CT[topic][currWordIndex] -= 1
            
            for k in range(K):
                numerator = (CT[k][currWordIndex] + beta) * (CD[document][k] + alpha)
                temp1 = sum(CT[k])
                temp2 = sum(CD[document])
                denominator = (V*beta + temp1) * (K*alpha + temp2)
                prob[k] = numerator/denominator
                
            temp3 = sum(prob)
            prob = list(np.array(prob)/temp3)
            ran = random.uniform(0,1)
            temp4 = 0  
            for x in range(len(prob)):
                temp4 = sum(prob[:x+1])
                if ran <= temp4 : 
                    ind = x
                    break
                
            topic = ind                    
            topicList[pi[j]] = topic 
            CD[document][topic] += 1
            CT[topic][currWordIndex] += 1
            
    return CD, CT, topicList
        
def alternateMethod(D, V, data, uniqueWords):
    altMethod = []
    for i in range(D):
        temp = [0] * V
        altMethod.append(temp)

    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(uniqueWords)): 
                if data[i][j] == uniqueWords[k]: 
                    altMethod[i][k] += 1
                    break
    return altMethod

    
def toCSV(uniqueWords, CT):
    temp = copy.deepcopy(CT)
    lists = []
    for i in range(len(CT)):
        
        temp[i].sort(reverse = True)
        temp1 = temp[i]
        max1 = temp1[0]
        ind1 = CT[i].index(max1)
        tags1 = uniqueWords[ind1]
        max2 = temp1[1]
        ind2 = CT[i].index(max2)
        tags2 = uniqueWords[ind2]
        max3 = temp1[2]
        ind3 = CT[i].index(max3)
        tags3 = uniqueWords[ind3]
        max4 = temp1[3]
        ind4 = CT[i].index(max4)
        tags4 = uniqueWords[ind4]
        max5 = temp1[4]
        ind5 = CT[i].index(max5)
        tags5 = uniqueWords[ind5]
        temp2 = [tags1, tags2,tags3,tags4,tags5]
        lists.append(temp2) 
    createCSV(lists)

def plotGraph(mean1, std1, method1, mean2, std2, method2):
    x = [i for i in np.arange(0.1,1.1,0.1)]
    plt.errorbar(x, mean1, std1, label = method1)
    plt.errorbar(x , mean2, std2, label = method2)
    plt.xlabel("DataSize")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.show()

#if __name__ == "__main__":

#directory = "C:/Users/Viral Prajapati/Machine Learning/Programming Project 4/20newsgroups"
directory = os.path.dirname(os.path.abspath(__file__))
data , files = readDirectory(directory+"/20newsgroups")
words , uniqueWords = preprocess(data)
V = len(uniqueWords)
D = 200
K = 20
pi = []
for i in range(len(words)):
    pi.append(i)
random.shuffle(pi)

doc = []
for i in range(len(data)):
    for j in range(len(data[i])):
        doc.append(i)
        
alpha = 5/K        

topicList = [] 
topics = []
for i in range(len(data)):
    temp = []
    for j in range(len(data[i])):
        r = random.randint(0,19)
        topicList.append(r)
        temp.append(r)
    topics.append(temp)

tags = [i for i in range(0,20)]
CD = calCD(D, K, topics, tags)
CT = calCT(K, V, words, uniqueWords, tags)
CD, CT, topicList = gibbsSampling(CD, CT, words, uniqueWords, pi, K, doc, topicList)

toCSV(uniqueWords , CT)

label = np.genfromtxt(directory + "20newsgroups/index.csv", delimiter = ",")

labels = []
for i in range(len(files)):
    labels.append(label[int(files[i]) - 1][1])
    
mean1 , std1 = logistic(CD , labels, alpha, 1)

data2 = alternateMethod(D, V, data, uniqueWords)

mean2 , std2 = logistic(data2 , labels, alpha, 0)

method1 = "LDA"
method2= "Bag of Words"
plotGraph(mean1 , std1, method1, mean2, std2, method2)