# IMPORTS
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skimage.transform import resize
from skimage.feature import hog
import pickle
from utils.constants import Constants

lables = []
images = []


# FUNCTIONS
def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        if (filename.find("cat")):
            lables.append(0)
        elif (filename.find("dog")):
            lables.append(1)
        img = cv2.imread(os.path.join(folder, filename))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = resize(gray_image, (64, 128))
        fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=False)
        images.append(fd)
        # print(hog_image)


# CALLS
load_images_from_folder(Constants.trainingPath)
X_train, X_test, y_train, y_test = train_test_split(images, lables, test_size=0.1, random_state=0)
###
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train, y_train)
#
y_pred = logreg.predict(X_test)
# save the model to disk
r2_score = logreg.score(X_test, y_test)
print(r2_score * 100, '%')
###Save Model ######################################
pickle.dump(logreg, open(Constants.filename, 'wb'))
