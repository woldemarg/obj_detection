import pickle
import random
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from imutils import paths
from transfer_learning_via_DL.cnn_as_feature_extractor.scripts import config

# %%

model = VGG16(weights='imagenet', include_top=False)
le = None

# %%

split = 'evaluation'
for split in (config.TRAIN, config.TEST, config.VAL):
    p = os.path.sep.join([config.BASE_PATH, split])
    imagePaths = list(paths.list_images(p))
    random.shuffle(imagePaths)
    labels = [p.split(os.path.sep)[-2] for p in imagePaths]
    if le is None:
        le = LabelEncoder()
        le.fit(labels)
    csvPath = os.path.sep.join([config.BASE_CSV_PATH,
                                '{}.csv'.format(split)])
    csv = open(csvPath, 'w')
    for b, i in enumerate(range(0, len(imagePaths), config.BATCH_SIZE)):
        print("[INFO] processing batch {}/{}"
              .format(b + 1,
                      int(np.ceil(len(imagePaths) / float(config.BATCH_SIZE)))
                      ))
        batchPaths = imagePaths[i: i + config.BATCH_SIZE]
        batchLabels = le.transform(labels[i: i + config.BATCH_SIZE])
        batchImages = []
        for imagePath in batchPaths:
            image = load_img(imagePath, target_size=config.INPUT_SIZE)
            image_arr = img_to_array(image)
            image_exp = np.expand_dims(image_arr, axis=0)
            image_pre = preprocess_input(image_exp)
            batchImages.append(image_pre)
        batchImages_arr = np.vstack(batchImages)
        features = model.predict(batchImages_arr, batch_size=config.BATCH_SIZE)
        features_res = features.reshape((features.shape[0], 7 * 7 * 512))
        for label, vec in zip(batchLabels, features_res):
            vec = ','.join([str(v) for v in vec])
            csv.write('{},{}\n'.format(label, vec))
    csv.close()

# %%

f = open(config.LE_PATH, 'wb')
f.write(pickle.dumps(le))
f.close()
