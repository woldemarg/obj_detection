import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import numpy as np
from transfer_learning_via_DL.cnn_as_feature_extractor.scripts import config

# %%

def csv_feature_generator(inputPath, bs, numClasses, mode="train"):
    f = open(inputPath, "r")
    while True:
        data = []
        labels = []
        while len(data) < bs:
            row = f.readline()
            if row == "":
                f.seek(0)
                row = f.readline()
                if mode == "eval":
                    break
            row = row.strip().split(",")
            label = row[0]
            label = to_categorical(label, num_classes=numClasses)
            features = np.array(row[1:], dtype="float")
            data.append(features)
            labels.append(label)
        # https://stackoverflow.com/a/60131716/6025592
        yield (np.array(data), np.array(labels))


# %%

le = pickle.loads(open(config.LE_PATH, 'rb').read())

trainPath = os.path.sep.join([config.BASE_CSV_PATH,
                              '{}.csv'.format(config.TRAIN)])
valPath = os.path.sep.join([config.BASE_CSV_PATH,
                            '{}.csv'.format(config.VAL)])
testPath = os.path.sep.join([config.BASE_CSV_PATH,
                            '{}.csv'.format(config.TEST)])

totalTrain = sum([1 for ln in open(trainPath)])
totalVal = sum([1 for ln in open(valPath)])

testLabels = [int(row.split(',')[0]) for row in open(testPath)]
totalTest = len(testLabels)

# %%

trainGen = csv_feature_generator(trainPath,
                                 config.BATCH_SIZE,
                                 len(config.CLASSES),
                                 mode='train')

valGen = csv_feature_generator(valPath,
                               config.BATCH_SIZE,
                               len(config.CLASSES),
                               mode='eval')

testGen = csv_feature_generator(testPath,
                                config.BATCH_SIZE,
                                len(config.CLASSES),
                                mode='eval')

# %%

# rule of thumb is to take the square root of the previous number of nodes
# in the layer and then find the closest power of 2
model = Sequential()
model.add(Dense(128, input_shape=(7 * 7 * 512,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(config.CLASSES), activation='softmax'))

# %%

opt = SGD(lr=1e-3, momentum=0.9, decay=1e-3 / 25)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# %%

# https://github.com/frankkramer-lab/MIScnn/issues/11
H = model.fit(
    x=trainGen,
    steps_per_epoch=totalTrain // config.BATCH_SIZE,
    validation_data=valGen,
    validation_steps=totalVal // config.BATCH_SIZE,
    epochs=25)

predIdxs = model.predict(
    x=testGen,
    steps=(totalTest // config.BATCH_SIZE) + 1)

predIdxs_arr = np.argmax(predIdxs, axis=1)

print(classification_report(testLabels, predIdxs_arr,
                            target_names=le.classes_))

# %%

f = open(config.MODEL_PATH, 'wb')
f.write(pickle.dumps(model))
f.close()
