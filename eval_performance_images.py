import cv2
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score

'''
evaulate performance directly from images
'''

images = []
images_y = []
n = 10
for i in range(n):
    images.append(cv2.imread(f"./experiments/m_prediction_{i}.png"))
    images_y.append(cv2.imread(f"./experiments/m_groundtruth_{i}.png"))

images = np.array(images).reshape(n*736*973,3)
images_y = np.array(images_y).reshape(n*736*973,3)
images = np.array(list(map(lambda x: str(x[0]) + str(x[1]) + str(x[2]), images)))
images_y = np.array(list(map(lambda x: str(x[0]) + str(x[1]) + str(x[2]), images_y)))
print(images.shape)
print(images_y.shape)

'''
0165255 -> orange -> Arbuscules
25500 -> blue -> Vesicles 
255255255 -> white -> Background
511530 -> green -> Hyphae
'''

print(classification_report(images_y,images))
print(jaccard_score(images_y,images, average = None))
print(np.mean(jaccard_score(images_y,images, average = None)))