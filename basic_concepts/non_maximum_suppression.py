# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt


# %%

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")

# %%

image = np.ones((250, 180, 3), dtype=np.uint8)
orig = image.copy()

bounding_boxes = np.array([
       (12, 84, 140, 212),
       (24, 84, 152, 212),
       (36, 84, 164, 212),
       (12, 96, 140, 224),
       (24, 96, 152, 224),
       (24, 108, 152, 236)])

np.random.shuffle(bounding_boxes)

COLORS = np.random.uniform(0, 255, size=(bounding_boxes.shape[0], 3))

# %%

for j, (startX, startY, endX, endY) in enumerate(bounding_boxes):
    cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[j], 2)

boxes_nms = non_max_suppression(bounding_boxes)[0]
cv2.rectangle(orig,
              (boxes_nms[0], boxes_nms[1]),
              (boxes_nms[2], boxes_nms[3]),
              (255, 255, 255),
              3)

orig_RGB = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
plt.imshow(orig_RGB)
