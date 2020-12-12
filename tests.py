import os
import numpy as np
#from keras.preprocessing import image
import cv2
import dlib
from matplotlib import pyplot as plt

global basedir, image_paths, target_size
basedir = './Datasets'
images_dir = os.path.join(basedir,'celeba/img')
labels_filename = 'labels.csv'

test_image = os.path.join(images_dir, '100.jpg')
print(test_image)

gray = cv2.imread(test_image)
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(gray) #cmap='gray'
plt.show()

# Histogram of Oriented Gradients + Linear SVM method - for detecting rectangular face region - dlib implementation
detector = dlib.get_frontal_face_detector()

# One Millisecond Face Alignment with an Ensemble of Regression Trees - for detecting facial landmarks - dlib implementation
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Helper function 1 - from dlib 'rect' to tuple
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

# Helper function 2 - from dlib 'shape' to list of (x-y) coords
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# Detecting the bounding box of faces in our image, and num of faces detected
rects = detector(gray, 1)

# Looping over face detections (in this case every image only has 1 face)
for (i, rect) in enumerate(rects):
    print(i)
    (x, y, w, h) = rect_to_bb(rect)

    # drawing the rectangle on the image
    cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 3)

    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)

    for (x, y) in shape:
        cv2.circle(gray, (x, y), 1, (255, 0, 0), -1)

plt.figure(figsize=(10,5))
plt.imshow(gray, cmap='gray')
plt.show()