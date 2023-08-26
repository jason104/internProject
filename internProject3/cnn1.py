from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Add
from tensorflow.keras.layers import Input,Flatten,Masking,BatchNormalization
from tensorflow.keras.layers import LSTM,Conv1D, Conv2D, MaxPool1D, MaxPool2D
import json
from argparse import ArgumentParser
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from tensorflow.keras import optimizers, Model
from sklearn.metrics import classification_report


parser = ArgumentParser()
parser.add_argument("-p", "--phase", dest='phaseNum', default='4')
args = parser.parse_args()

def generateKmers(seq):
    N = len(seq)
    return [seq[i:i+3] for i in range(N - 2)]

def getEncoding(trainPath, testPath):
    f1 = open(trainPath)
    trainDataset = json.load(f1)
    trainData, trainLabel = trainDataset.keys(), list(trainDataset.values())
    f2 = open(testPath)
    testDataset = json.load(f2)
    testData, testLabel = testDataset.keys(), list(testDataset.values())

    encoded = []
    docModel = Doc2Vec.load('../model/Doc2Vec_model/AFP_doc2vec_DS2.model')
    for seq in trainData:
        kmers = generateKmers(seq)
        encoded.append(docModel.infer_vector(kmers))
    trainData = np.array(encoded)
    L = len(trainData)
    trainData = np.reshape(trainData, (L, 20, 20))
    trainLabel = np.array(trainLabel)

    encoded = []
    for seq in testData:
        kmers = generateKmers(seq)
        encoded.append(docModel.infer_vector(kmers))
    testData = np.array(encoded)
    L = len(testData)
    testData = np.reshape(testData, (L, 20, 20))
    testLabel = np.array(testLabel)

    return trainData, trainLabel, testData, testLabel

def binaryDnn(trainPath, testPath):
    trainData, trainLabel, testData, testLabel = getEncoding(trainPath, testPath)

    input_ = Input(shape=(20, 20, 1))
    cnn1 = Conv2D(64, 4, activation = 'relu', padding="same", input_shape=(20, 20, 1))(input_)
    norm = BatchNormalization()(cnn1)
    pool = MaxPool2D(5)
    cnn2 = Conv2D(64, 4, activation = 'relu', padding="same")(norm)
    norm = BatchNormalization()(cnn2)
    d = Dropout(0.5)(norm)
    f = Flatten()(d)
    dense1 = Dense(512, activation = "relu")(f)
    d = Dropout(0.5)(dense1)
    dense2 = Dense(256, activation = "relu")(d)
    d = Dropout(0.5)(dense2)
    result = Dense(2, activation = "softmax")(d)
    model = Model(inputs=input_, outputs=result)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    history = model.fit(trainData, trainLabel, shuffle=True, epochs=15, batch_size=512)
    history.model.save('./models/linearModel.h5')
    _, accuracy = model.evaluate(testData, testLabel)
    print('accuracy = ', accuracy)


def multiDnn(trainPath, testPath, phase):
    trainData, trainLabel, testData, testLabel = getEncoding(trainPath, testPath)

    input_ = Input(shape=(20, 20, 1))
    cnn1 = Conv2D(64, 4, activation = 'relu', padding="same", input_shape=(20, 20, 1))(input_)
    norm = BatchNormalization()(cnn1)
    pool = MaxPool2D(5)
    cnn2 = Conv2D(64, 4, activation = 'relu', padding="same")(norm)
    norm = BatchNormalization()(cnn2)
    d = Dropout(0.5)(norm)
    f = Flatten()(d)
    dense1 = Dense(512, activation = "relu")(f)
    d = Dropout(0.5)(dense1)
    dense2 = Dense(256, activation = "relu")(d)
    d = Dropout(0.5)(dense2)

    if phase == 3:
    	result = Dense(5, activation = "sigmoid")(f)
    else:		
    	result = Dense(20, activation = "sigmoid")(f)
    model = Model(inputs=input_,outputs=result)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])


    history = model.fit(trainData, trainLabel, shuffle=True, epochs=35, batch_size=512)
    history.model.save('./models/linearModel.h5')
    _, accuracy = model.evaluate(testData, testLabel)
    predicted = model.predict(testData)
    predicted = np.argmax(predicted, axis=1)

    if phase == 3:
        print(classification_report(testLabel, predicted, target_names=['class 0', 'class 1', 'class 2', 'class 3', 'class 4']))
    else:
        print(classification_report(testLabel, predicted, target_names=['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9', 'class 10', 'class 11', 'class 12', 'class 13', 'class 14', 'class 15', 'class 16', 'class 17', 'class 18', 'class 19']))
    print('accuracy = ', accuracy)







if int(args.phaseNum) == 1:
    binaryDnn('../data/trueData/p1train.json', '../data/trueData/p1test.json')
elif int(args.phaseNum) == 2:
    binaryDnn('../data/trueData/p2train.json', '../data/trueData/p2test.json')
elif int(args.phaseNum) == 3:
    multiDnn('../data/trueData/p3train.json', '../data/trueData/p3test.json', 3)
elif int(args.phaseNum) == 4:
    multiDnn('../data/trueData/p4train.txt', '../data/trueData/p4test.txt', 4)
