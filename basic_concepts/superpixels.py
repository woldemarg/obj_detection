from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%

IMG_PATH = 'images/lemons.jpg'

# %%

image = img_as_float(io.imread(IMG_PATH))

segments = slic(image, n_segments=200, sigma=5)

plt.figure(figsize=(10, 10))
plt.imshow(mark_boundaries(image, segments))

# %%

num_of_segments = np.unique(segments)

image_cv = cv2.imread(IMG_PATH)

mask = np.zeros(image.shape[:2], dtype=np.uint8)
mask[segments == 130] = 255

cv2.imshow('applied', cv2.bitwise_and(image_cv, image_cv, mask=mask))
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%

mask_inv = np.zeros(image.shape[:2], dtype=np.uint8)
mask_inv[segments != 130] = 255
mask_inv = cv2.merge([mask_inv, mask_inv, mask_inv])

cv2.imshow('applied', cv2.bitwise_and(image_cv, mask_inv))
cv2.waitKey(0)
cv2.destroyAllWindows()
