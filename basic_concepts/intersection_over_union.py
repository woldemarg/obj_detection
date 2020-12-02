from collections import namedtuple
import cv2

# %%

Detection = namedtuple('Detection',
                       ['image_path', 'gt', 'pred'])

# %%

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = inter_area / float(boxA_area + boxB_area - inter_area)

    return iou


# %%

examples = [
    Detection('images/image_0002.jpg', [39, 63, 203, 112], [54, 66, 198, 114]),
    Detection('images/image_0016.jpg', [49, 75, 203, 125], [42, 78, 186, 126]),
    Detection('images/image_0075.jpg', [31, 69, 201, 125], [18, 63, 235, 135]),
    Detection('images/image_0090.jpg', [50, 72, 197, 121], [54, 72, 198, 120]),
    Detection('images/image_0120.jpg', [35, 51, 196, 110], [36, 60, 180, 108])]

# %%

for detection in examples:
    image = cv2.imread(detection.image_path)
    cv2.rectangle(image,
                  tuple(detection.gt[:2]),
                  tuple(detection.gt[2:]),
                  (0, 255, 0),
                  2)
    cv2.rectangle(image,
                  tuple(detection.pred[:2]),
                  tuple(detection.pred[2:]),
                  (0, 0, 255),
                  2)
    iou = bb_intersection_over_union(detection.gt, detection.pred)
    cv2.putText(image,
                'IoU: {:.4f}'.format(iou),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2)
    print("{}: {:.4f}".format(detection.image_path, iou))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
