import os
import re
from math import log
test_dir = 'test'
train_dir = 'train'
spam_dir = 'spam'
ham_dir = 'ham'

def get_test_data(path, swords):
    word_list = []
    dictio = {}
    for file in os.listdir(path):
        f = open(path+"/"+file, encoding="utf8", errors='ignore')
        wlist=[]
        for line in f:
            wlist.extend(line.strip("\n").split())
        new = []
        for x in wlist:
            if x.lower() not in swords:
                new.append(x)
        dictio[file] = wlist
        for word in new:
            word_list.append(word)
    return word_list,dictio

def get_data(path, swords):
    word_list = []
    files_count = len(os.listdir(path))
    for file in os.listdir(path):
        f = open(path+"/"+file, encoding="utf8", errors='ignore')
        wlist=[]
        for line in f:
            wlist.extend(line.strip("\n").split())
        new = []
        for x in wlist:
            if x.lower() not in swords:
                new.append(x)
        for word in new:
            word_list.append(word)
    return word_list,files_count

def class_probab(SFD,S_len,comb_list):
    j = {}
    for k in comb_list:
        if k in SFD:
            i = SFD[k]
        else:
            i=0
        prob = float(i + 1)/(S_len + len(comb_list))
        j[k] = prob
    return j

def NBTraining(S_freq, H_freq, SL, HL,k):
    spam_probab=class_probab(S_freq,len(SL),k)
    ham_probab=class_probab(H_freq,len(HL),k)
    return spam_probab,ham_probab

def NBTesting(S, SD, HD, key):
    Dic = [SD,HD]
    j=0
    for i in range(len(Dic)):
        for x in Dic[i]:
            spam_val = log(S)
            ham_val = log(1-S)
            for word in Dic[i][x]:
                if word in key:
                    spam_val += log(spam_probab[word])
                    ham_val += log(ham_probab[word])
            if spam_val >= ham_val and i == 0:
                j +=1
            elif spam_val <= ham_val and i == 1:
                j +=1
        k=float(j)/(len(SD) + len(HD))
    return k

SL_test, SD_test = get_test_data(test_dir + '/' + spam_dir, [])
HL_test, HD_test = get_test_data(test_dir + '/' + ham_dir, [])

SL_train, SD_train = get_data(train_dir + '/' + spam_dir,[])
HL_train, HD_train = get_data(train_dir + '/' + ham_dir,[])

S_prior = SD_train/(SD_train+HD_train)
S_freq={}
for i in SL_train:
    if i in S_freq.keys():
        S_freq[i]+=1
    else:
        S_freq[i]=1
H_freq={}
for i in HL_train:
    if i in H_freq.keys():
        H_freq[i]+=1
    else:
        H_freq[i]=1

k_list = set(SL_train).union(set(HL_train))
spam_probab, ham_probab = NBTraining(S_freq, H_freq, SL_train, HL_train,k_list)
print("Accuracy of Naive Bayes before removing stop words", NBTesting(S_prior, SD_test, HD_test, k_list))


stop_words = []
f = open(r'./stopwords.txt', 'r')
for line in f:
    for word in line.split():
        stop_words.append(word)


SL_train, SD_train = get_data(train_dir + '/' + spam_dir, stop_words)
HL_train, HD_train = get_data(train_dir + '/' + ham_dir, stop_words)

S_prior = SD_train/(SD_train+HD_train)

S_freq={}
for i in SL_train:
    if i in S_freq.keys():
        S_freq[i]+=1
    else:
        S_freq[i]=1
H_freq={}
for i in HL_train:
    if i in H_freq.keys():
        H_freq[i]+=1
    else:
        H_freq[i]=1
k_list = set(SL_train).union(set(HL_train))

CP1, CP2 = NBTraining(S_freq, H_freq, SL_train, HL_train,k_list)
print("Accuracy of Naive Bayes after removing stop words is", NBTesting(S_prior, SD_test, HD_test, k_list))


