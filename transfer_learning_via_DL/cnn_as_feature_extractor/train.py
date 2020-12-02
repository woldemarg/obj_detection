import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from transfer_learning_via_DL.cnn_as_feature_extractor.scripts import config


# %%

def load_data_split(splitPath):
    data = []
    labels = []

    for row in open(splitPath):
        row = row.strip().split(',')
        label = row[0]
        features = np.array(row[1:], dtype='float')
        data.append(features)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    return (data, labels)


# %%

trainingPath = os.path.sep.join([config.BASE_CSV_PATH,
                                 '{}.csv'.format(config.TRAIN)])

testingPath = os.path.sep.join([config.BASE_CSV_PATH,
                                '{}.csv'.format(config.TEST)])

trainX, trainY = load_data_split(trainingPath)
testX, testY = load_data_split(testingPath)

le = pickle.loads(open(config.LE_PATH, 'rb').read())

# %%

model = LogisticRegression(max_iter=150)
model.fit(trainX, trainY)

preds = model.predict(testX)
print(classification_report(testY, preds, target_names=le.classes_))

# %%

f = open(config.MODEL_PATH, 'wb')
f.write(pickle.dumps(model))
f.close()
