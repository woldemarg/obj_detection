import time
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression
from tensorflow.keras.applications import ResNet50, imagenet_utils
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from cnn_classifier_to_obj_detector.part_1.detection_helpers import sliding_window, image_pyramid

# %%

WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (250, 250)
INPUT_SIZE = (224, 224)

# %%

model = ResNet50(weights='imagenet', include_top=True)
orig = cv2.imread('images/hummingbird.jpg')
orig_res = imutils.resize(orig, width=WIDTH)
H, W = orig_res.shape[:2]

# %%

pyramid = image_pyramid(orig_res, scale=PYR_SCALE, minSize=ROI_SIZE)
rois = []
locs = []

# %%

for image in pyramid:
    scale = W / float(image.shape[1])
    for (x, y, roi_orig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)
        roi = cv2.resize(roi_orig, INPUT_SIZE)
        roi_arr = img_to_array(roi)
        roi_pre = preprocess_input(roi_arr)
        rois.append(roi_pre)
        locs.append((x, y, x + w, y + h))

# %%

# https://stackoverflow.com/a/47143492/6025592
rois_arr = np.array(rois, dtype='float32')

# %%

start = time.time()
preds = model.predict(rois_arr)
end = time.time()

print('[INFO] Classifying rois took {:.5f} seconds'.format(end - start))

# %%

preds_top = imagenet_utils.decode_predictions(preds, top=1)
labels = {}

for i, p in enumerate(preds_top):
    imagenetID, label, prob = p[0]
    if prob >= 0.95:
        box = locs[i]
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

# %%

for label in labels:
    clone = orig_res.copy()
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes_nms = non_max_suppression(boxes, proba)
    for (startX, startY, endX, endY) in boxes_nms:
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone,
                    label,
                    (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    2)
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
