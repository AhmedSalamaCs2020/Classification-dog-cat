import os
import pickle

import pandas as pd
from skimage.transform import resize
from skimage.feature import hog
import cv2

# DATA
from utils.constants import Constants

images = []


def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = resize(gray_image, (64, 128))
        fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=False)
        images.append(fd)


load_images_from_folder(Constants.testingPath)
# load the model from disk
loaded_model = pickle.load(open(Constants.filename, 'rb'))
data=pd.read_csv(Constants.sampleSubmission)
result = loaded_model.score(images, data['label'])

print(result)
