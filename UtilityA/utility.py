# ======================================================================================================================
# This file contains utility functions used to extract features/data from the dlib face detector for Task A.
# It also contains the global paths to the celebA image dataset directory. Adapted from lab.
# ======================================================================================================================
import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib

# PATH TO ALL IMAGES AND PRE-TRAINED FACE DETECTOR
# ====================================================
global basedir, image_paths, target_size
basedir = './Datasets'

celeba_set = os.path.join(basedir,'celeba')
celeba_test_set = os.path.join(basedir,'celeba_test')

labels_filename = 'labels.csv'
images_dir = 'img'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# ====================================================

# how to find frontal human faces in an image using 68 landmarks.
# These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

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

def printProgressBar(iteration, total, prefix = 'Progress:', suffix = ' ', decimals = 1, length = 50, fill = '█', printEnd = ''):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print("Complete")


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_features_labels(data_set, smiles=False):
    """
    This function extracts the landmarks features for all images in the folder specified by 'data_set' parameter.
    It also extracts the gender label or smile label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        labels:      an array containing the gender/smile label (male=0 and female=1) for each image in
                            which a face was detected
    """

    # Set up directory paths
    images_dir = os.path.join(data_set, 'img')
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(data_set, labels_filename), 'r')

    # strip lines of \n
    lines = [line.rstrip('\n') for line in labels_file]

    # dictionary of gender labels for each image
    if smiles:
        labels = {line.split('\t')[0]: int(line.split('\t')[3]) for line in lines[1:]}
    else:
        labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}

    # progress bar
    printProgressBar(0, len(image_paths), prefix="Extracting features from: " + images_dir)

    # extracting features from dlib_shape_detector
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for i, img_path in enumerate(image_paths):
            file_name = img_path.split('.')[1].split('/')[-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))

            # getting features
            features, _ = run_dlib_shape(img)

            if features is not None:
                all_features.append(features)
                all_labels.append(labels[file_name])

            printProgressBar(i + 1, len(image_paths), prefix="Extracting features from: " + images_dir)

    landmark_features = np.array(all_features)
    labels = (np.array(all_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1

    txt = "Successfully extracted features from: {percent:.2f}% of images"
    print(txt.format(percent=(len(labels) / len(image_paths)) * 100))

    return landmark_features, labels
