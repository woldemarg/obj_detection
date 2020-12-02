import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%

# image = cv2.imread('images/footballplayer.jpg')
image = cv2.imread('images/table.jpg')
# image = cv2.imread('images/lemons.jpg')
# image = cv2.imread('images/barcelona.jpg')

# %%

saliency_sr = cv2.saliency.StaticSaliencySpectralResidual_create()
success, saliency_map_sr = saliency_sr.computeSaliency(image)
saliency_map_scaled = (saliency_map_sr * 255).astype(np.uint8)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(saliency_map_scaled, cv2.COLOR_BGR2RGB))
fig.tight_layout()

# %%

saliency_fg = cv2.saliency.StaticSaliencyFineGrained_create()
success, saliency_map_fg = saliency_fg.computeSaliency(image)
saliency_map_fg_scaled = (saliency_map_fg * 255).astype(np.uint8)
saliency_map_fg_bin = cv2.threshold(saliency_map_fg_scaled, 0, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(saliency_map_fg_bin, cv2.COLOR_BGR2RGB))
fig.tight_layout()

# %%

saliency_bing = cv2.saliency.ObjectnessBING_create()
# https://github.com/opencv/opencv_contrib/tree/master/modules/saliency/samples/ObjectnessTrainedModel
saliency_bing.setTrainingPath('basic_concepts/bing_pretrained_model')
success, saliency_map_bing = saliency_bing.computeSaliency(image)
num_detections = saliency_map_bing.shape[0]


for i in range(0, min(num_detections, 10)):
    output = image.copy()
    (startX, startY, endX, endY) = saliency_map_bing[i].flatten()
    color = np.random.randint(0, 255, size=(3,))
    color = [int(c) for c in color]
    cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
    cv2.imshow('image', output)
    cv2.waitKey(0)

cv2.destroyAllWindows()
