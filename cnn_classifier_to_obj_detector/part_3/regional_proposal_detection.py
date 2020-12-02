from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt
import numpy as np
import cv2

# %%

def selective_search(image, method='fast'):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    if method == 'fast':
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    return rects


# %%

# IMG_PATH = 'images/beagle.jpg'
IMG_PATH = 'images/table.jpg'
# IMG_PATH = 'images/hummingbird.jpg'
# IMG_PATH = '/content/gdrive/MyDrive/ssh_files/hummingbird.jpg'
INPUT_SIZE = (224, 224)

# %%

model = ResNet50(weights='imagenet')
image = cv2.imread(IMG_PATH)
H, W = image.shape[:2]

# %%

rects = selective_search(image)
proposals = []
boxes = []

# %%

for (x, y, w, h) in rects:
    if w / float(W) < 0.1 or h / float(H) < 0.1:
        continue
    roi = image[y: y + h, x: x + w]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_res = cv2.resize(roi_rgb, INPUT_SIZE)
    roi_arr = img_to_array(roi_res)
    roi_pre = preprocess_input(roi_arr)

    proposals.append(roi_pre)
    boxes.append((x, y, w, h))

# %%

proposals_arr = np.array(proposals)
preds = model.predict(proposals_arr)
preds_decoded = imagenet_utils.decode_predictions(preds, top=1)

# %%

labels = {}

for i, p in enumerate(preds_decoded):
    imagenetID, label, prob = p[0]
    if prob >= 0.95:
        x, y, w, h = boxes[i]
        box = (x, y, x + w, y + h)
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

# %%

COLORS = np.random.uniform(0, 255, size=(len(labels.keys()), 3))

fig, ax = plt.subplots(2, 1, figsize=(20, 10))
before = image.copy()
after = image.copy()

for i, label in enumerate(labels.keys()):

    for (box, prob) in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(before,
                      (startX, startY),
                      (endX, endY),
                      COLORS[i],
                      2)

    boxes_lb = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes_nms = non_max_suppression(boxes_lb, proba)
    for (startX, startY, endX, endY) in boxes_nms:
        cv2.rectangle(after,
                      (startX, startY),
                      (endX, endY),
                      COLORS[i],
                      2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(after,
                    label,
                    (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    COLORS[i],
                    2)
ax[0].set_title('Before NMS')
ax[1].set_title('After NMS')
ax[0].imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
fig.tight_layout()
