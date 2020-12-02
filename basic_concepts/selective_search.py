import random
import time
import cv2

# %%

# IMG_PATH = 'images/lemons.jpg'
IMG_PATH = 'images/table.jpg'

# %%

image = cv2.imread(IMG_PATH)

# %%

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

# %%

ss.switchToSelectiveSearchFast()
start = time.time()
rect_fast = ss.process()
end = time.time()

print("[INFO] selective search took {:.4f} seconds".format(end - start))

# %%

ss.switchToSelectiveSearchQuality()
start = time.time()
rect_qual = ss.process()
end = time.time()

print("[INFO] selective search took {:.4f} seconds".format(end - start))

# %%

for i in range(0, len(rect_fast), 100):
    output = image.copy()
    for (x, y, w, h) in rect_fast[i:i + 100]:
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    cv2.imshow("output", output)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
