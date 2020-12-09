import os
import random
# from shutil import copyfile
import lxml.etree as et
import cv2
from imutils import paths
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from tbl_detection import config

# %%

image_paths = list(paths.list_images(config.ORIG_IMAGES))
sample_ids = sorted(random.sample(range(0, len(image_paths)), 25))
sample_data = {}

for i, image_path in enumerate(image_paths):
    filename = image_path.split(os.path.sep)[-1]
    filename = filename[:filename.rfind('.')]
    ANNOT_PATH = os.path.sep.join([config.ORIG_ANNOTS,
                                  "{}.xml".format(filename)])

    if not os.path.exists(ANNOT_PATH):
        if i in sample_ids:
            rand_num = random.choice(list(set(range(i + 1,
                                              len(image_paths)))
                                          - set(sample_ids)))
            sample_ids.append(rand_num)
        continue

    CONTENTS = str(open(ANNOT_PATH).read())
    soup = BeautifulSoup(CONTENTS, 'xml')

    # remove cells as objects from annotations
    for o in soup.find_all("object"):
        label = o.find("name").string
        if label == 'cell':
            o.decompose()

    # output from BS without extraneous newlines
    # https://stackoverflow.com/a/58346786/6025592
    xml_string = et.fromstring(soup.decode_contents())
    xml_styles = et.fromstring(str(open(config.XML_STYLE).read()))

    transformer = et.XSLT(xml_styles)
    xml_prettified = transformer(xml_string)

    if not os.path.exists(config.TBLS_ANNOTS):
        os.makedirs(config.TBLS_ANNOTS)

    with open(os.path.sep.join([config.TBLS_ANNOTS,
                                '{}.xml'.format(filename)]), 'w') as f:
        f.write(str(xml_prettified))

    if not os.path.exists(config.TBLS_IMAGES):
        os.makedirs(config.TBLS_IMAGES)

    img = cv2.imread(image_path)
    cv2.imwrite(os.path.sep.join([config.TBLS_IMAGES,
                                  '{}.jpg'.format(filename)]),
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # copyfile(image_path,
    #          os.path.sep.join([config.TBLS_IMAGES,
    #                         '{}.jpg'.format(filename)]))

    if i in sample_ids:
        tbl_bboxes = []
        for o in soup.find_all("object"):
            xMin = int(o.find("xmin").string)
            yMin = int(o.find("ymin").string)
            xMax = int(o.find("xmax").string)
            yMax = int(o.find("ymax").string)
            tbl_bboxes.append((xMin, yMin, xMax, yMax))
        sample_data[filename] = tbl_bboxes

# %%

fig, ax = plt.subplots(5, 5, figsize=(25, 25))
for i, (k, v) in enumerate(sample_data.items()):
    img = cv2.imread(os.path.sep.join([config.TBLS_IMAGES,
                                       '{}.jpg'.format(k)])).copy()
    for bbox in v:
        (xMin, yMin, xMax, yMax) = bbox
        cv2.rectangle(img,
                      (xMin, yMin),
                      (xMax, yMax),
                      (0, 0, 255),
                      2)
    ax.flat[i].axis('off')
    ax.flat[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.tight_layout()
