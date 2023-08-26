from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input,Flatten,Masking,BatchNormalization
import json
from argparse import ArgumentParser
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from tensorflow.keras import optimizers
from sklearn.metrics import classification_report


parser = ArgumentParser()
parser.add_argument("-p", "--phase", dest='phaseNum', default='4')
args = parser.parse_args()

def generateKmers(seq):
    N = len(seq)
    return [seq[i:i+3] for i in range(N - 2)]

def binaryDnn(trainPath, testPath):
    f1 = open(trainPath)
    trainDataset = json.load(f1)
    trainData, trainLabel = list(trainDataset.keys()), list(trainDataset.values())
    f2 = open(testPath)
    testDataset = json.load(f2)
    testData, testLabel = list(testDataset.keys()), list(testDataset.values())
    
    encoded = []
    docModel = Doc2Vec.load('../model/Doc2Vec_model/AFP_doc2vec_DS2.model')
    for seq in trainData:
        kmers = generateKmers(seq)
        encoded.append(docModel.infer_vector(kmers))
    trainData = np.array(encoded)
    trainLabel = np.array(trainLabel)
    print(trainLabel.shape)

    encoded = []
    for seq in testData:
        kmers = generateKmers(seq)
        encoded.append(docModel.infer_vector(kmers))
    testData = np.array(encoded)
    testLabel = np.array(testLabel)
    

    input_ = Input(shape=(400,))
    model = Sequential()
    model.add(input_)
    model.add(Dense(400, activation='sigmoid'))
    model.add(Dense(400, activation='sigmoid'))
    model.add(Dense(400, activation='sigmoid'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    history = model.fit(trainData, trainLabel, shuffle=True, epochs=50, batch_size=512)
    history.model.save('./models/linearModel.h5')
    _, accuracy = model.evaluate(testData, testLabel)
    predicted = model.predict(testData)
    predicted = np.argmax(predicted, axis=1)
            
    print(classification_report(testLabel, predicted, target_names=['class 0', 'class 1']))
    print('accuracy = ', accuracy)


def multiDnn(trainPath, testPath, phase):
    f1 = open(trainPath)
    trainDataset = json.load(f1)
    trainData, trainLabel = list(trainDataset.keys()), list(trainDataset.values())
    f2 = open(testPath)
    testDataset = json.load(f2)
    testData, testLabel = list(testDataset.keys()), list(testDataset.values())

    encoded = []
    docModel = Doc2Vec.load('../model/Doc2Vec_model/AFP_doc2vec_DS2.model')
    for seq in trainData:
        kmers = generateKmers(seq)
        encoded.append(docModel.infer_vector(kmers))
    trainData = np.array(encoded)
    trainLabel = np.array(trainLabel)
    #print(trainLabel.shape)

    encoded = []
    for seq in testData:
        kmers = generateKmers(seq)
        encoded.append(docModel.infer_vector(kmers))
    testData = np.array(encoded)
    testLabel = np.array(testLabel)
    

    input_ = Input(shape=(400,))
    model = Sequential()
    model.add(input_)
    model.add(Dense(400, activation='sigmoid'))
    model.add(Dense(400, activation='sigmoid'))
    #model.add(Dense(50, activation='sigmoid'))
    if phase == 3:
        model.add(Dense(9, activation='sigmoid'))
    else:
        model.add(Dense(20, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    history = model.fit(trainData, trainLabel, shuffle=True, epochs=200, batch_size=512)
    history.model.save('./models/linearModel.h5')
    _, accuracy = model.evaluate(testData, testLabel)
    predicted = model.predict(testData)
    predicted = np.argmax(predicted, axis=1)
            
    print(classification_report(testLabel, predicted, target_names=['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8']))
    print('accuracy = ', accuracy)







if int(args.phaseNum) == 1:
    binaryDnn('../data/trueData/p1train.json', '../data/trueData/p1test.json')
elif int(args.phaseNum) == 2:
    binaryDnn('../data/trueData/p2train.json', '../data/trueData/p2test.json')
elif int(args.phaseNum) == 3:
    multiDnn('../data/trueData/p3train.json', '../data/trueData/p3test.json', 3)
elif int(args.phaseNum) == 4:
    multiDnn('../data/trueData/p4train.json', '../data/trueData/p4test.json', 4)
    


