import math
import os
'''
def File_Features( filename):
    features = {'bias': 1.0}
    words = Extract_Words(fileDir + "/" + filename)
    return features
'''

'''
def Sum_Weights(features, wts):
    weightedSum = 0.0
    for feature, value in features.items():
        if feature in wts:
            weightedSum += value * wts[feature]
    return weightedSum'''
    
def Extract_Words(filePath):
    words = []
    try:
       with open(filePath) as file:
           words = [word for line in file for word in line.split()]
    except OSError as ex:
        print(ex.message)
    finally:
        return words


def building_vocab(data, words_list):
    vocab = []
    for classType in data:
        for item in data[classType]:
            for word in item:
                if word not in vocab and word.lower() not in words_list:
                    vocab.append(word)
    return vocab

'''
def Features( filename):
    features = {'bias': 1.0}
    words = Extract_Words(fileDir + "/" + filename)
    return words
'''

def Features(doc):
    features = {'bias': 1.0}
    for word in doc:
        features[word] = doc.count(word)
    return features

def File_Features(fileDir, filename):
    features = {'bias': 1.0}
    words = Extract_Words(fileDir + "/" + filename)
    for word in words:
        features[word] = words.count(word)
    return features

'''
def Weights(features, wts):
    weighted = 0.0
    for feature, value in features.items():
        if feature in wts:
            weightedSum += value * wts[feature]
    return weighted
'''
 
def Sum_Weights(features, wts):
    weightedSum = 0.0
    for feature, value in features.items():
        if feature in wts:
            weightedSum += value * wts[feature]
    return weightedSum

def Calc_ClassProb(features, wts):
    weightedSum = Sum_Weights(features, wts)
    try:
        value = math.exp(weightedSum) * 1.0
    except OverflowError as ex:
        return 1
    return round((value) / (1.0 + value), 5)
 
def train_LR(data, vocab, I, l_rate, l):
    wts = {'bias': 0.0}
    for word in vocab:
        wts[word] = 0.0
    for i in range(0, I):
        print('Iteration no. %d' % (i))
        errorSum = {}
        for classType in data:
            for item in data[classType]:
                features = Features(item)
                classError = class_val[classType] - Calc_ClassProb(features, wts)
                if classError != 0:
                    for feature in features:
                        if(feature in errorSum):
                            errorSum[feature] += (features[feature] * classError)
                        else:
                            errorSum[feature] = (features[feature] * classError)
        for wt in wts:
            if wt in errorSum:
                wts[wt] = wts[wt] + (l_rate * errorSum[wt]) - (l_rate * l * wts[wt])
    return wts

def test_LR(wts):
    accuracy = {1: 0.0, 0: 0.0}
    for filename in os.listdir(os.getcwd() + "/test/" + 'ham'):
        features = File_Features(os.getcwd() +  "/test/" + 'ham', filename)
        classWeightedSum = Sum_Weights(features, wts)
        if(classWeightedSum >= 0):
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0
    for filename in os.listdir(os.getcwd() + "/test/" + 'spam'):
        features = File_Features(os.getcwd() + "/test/"+ 'spam', filename)
        classWeightedSum = Sum_Weights(features, wts)
        if(classWeightedSum < 0):
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0
    return (accuracy[1] * 100) / sum(accuracy.values())
    
'''   
def Data():
    data = {'ham': [], 'spam': []}
    for fileName in os.listdir(os.getcwd() + "/train/spam"):
        words = Extract_Words(os.getcwd() + "/train/spam/" + fileName)
        if len(words) > 0:
            data['spam'].append(words)
        docs['spam'] += 1.0
    return data
  '''
def Store_Data():
    data = {'ham': [], 'spam': []}
    for fileName in os.listdir(os.getcwd() + "/train/ham" ):
        words = Extract_Words(os.getcwd() +  "/train/ham/" + fileName)
        if len(words) > 0:
            data['ham'].append(words)
        docs['ham'] += 1.0
    for fileName in os.listdir(os.getcwd() + "/train/spam"):
        words = Extract_Words(os.getcwd() + "/train/spam/" + fileName)
        if len(words) > 0:
            data['spam'].append(words)
        docs['spam'] += 1.0
    return data

class_val = {'ham': 1.0, 'spam': 0.0}
docs= {'ham': 0.0, 'spam': 0.0}
stop_file_path = os.getcwd() + '/' + 'stopwords.txt'
stopWords = Extract_Words(stop_file_path)
data = Store_Data()
vocab = building_vocab(data, [])
vocab_without_stopwords = building_vocab(data, stopWords)

l_rate = 0.001 
lam = [0.2,0.1,0.05,0.01]
i = 1000
for l in lam:
    print("lambda value is:", l)
    wts = train_LR(data, vocab, i, l_rate, l)
    print("LR Accuracy including Stop Words : " + str(test_LR(wts)))
    wts = train_LR(data, vocab_without_stopwords, i, l_rate, l)
    print("LR Accuracy removing Stop Words : " + str(test_LR(wts)))
print("END")
