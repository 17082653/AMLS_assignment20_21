# ======================================================================================================================
# This file contains utility functions used to extract features/data from the dlib face detector, and eye extractor
# for Task B. It also contains the global paths to the cartoon set directory. Adapted from lab.
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

cartoon_set = os.path.join(basedir,'cartoon_set')
cartoon_test_set = os.path.join(basedir,'cartoon_set_test')

labels_filename = 'labels.csv'
images_dir = 'img'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# ====================================================

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

def printProgressBar(iteration, total, prefix = 'Progress:', suffix = ' ', decimals = 1, length = 50, fill = '█', printEnd = ''):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print("Complete")

# Takes in cv2 read image file in bgr and the eye color label
def eye_extractor(img, color):
    # Define the list of color boundaries
    # RGB - these were found by going through some of the cartoons with a color picker
    # dark - actual - light
    # green   57 75 34 - 103 136 59 - 145 170 111
    # brown   97 55 41 - 113  63 47 - 143 102 87
    # gray    87 98 95 - 137 154 150 - 168 180 176
    # blue    27 55 75   - 46 103 150  - 86 135 172
    # black    /  /  /   -  0  0  0    - 53 53 53

    # OPENCV WORKS IN BGR - SO IN REVERSE
    # These are ordered to correspond to the labels in dataset
    # so 0=brown, 1=blue, 2=green, 3=gray, 4=black
    boundaries = [
        ([41, 55, 97], [87, 102, 143]),
        ([75, 55, 27], [172, 135, 86]),
        ([34, 75, 57], [111, 170, 145]),
        ([95, 98, 87], [176, 180, 168]),
        ([0, 0, 0], [53, 53, 53]),
    ]

    # Crop image to left eye
    crop_img = img[245:280, 175:230]

    lower, upper = boundaries[color]

    # Create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # Find the colors within the specified boundaries and apply mask
    mask = cv2.inRange(crop_img, lower, upper)
    masked_image = cv2.bitwise_and(crop_img, crop_img, mask=mask)

    mean = cv2.mean(masked_image)

    #print("color: ", color, "; mean: ", mean)

    # 4th value is not important, so not returned
    return mean[:3]

def extract_features_labelsB(data_set):
    """
    This function extracts the landmarks features for all images in the folder specified by 'data_set' parameter.
    It also extracts the eye color label and face shape for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        labels:      an array containing the eye/face_shape label for each image in
                            which a face was detected
    """

    # Set up directory paths
    images_dir = os.path.join(data_set, 'img')
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(data_set, labels_filename), 'r')

    # strip lines of \n
    lines = [line.rstrip('\n') for line in labels_file]

    print(lines)

    # dictionary of gender labels for each image
    eye = {line.split('\t')[0] : int(line.split('\t')[1]) for line in lines[1:]}
    face_shape = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}

    # progress bar
    printProgressBar(0, len(image_paths), prefix="Extracting features from: " + images_dir)

    # extracting features from dlib_shape_detector
    if os.path.isdir(images_dir):
        all_features = []
        eye_labels = []
        face_shape_labels = []
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
                eye_labels.append(eye[file_name])
                face_shape_labels.append(face_shape[file_name])

            printProgressBar(i + 1, len(image_paths), prefix="Extracting features from: " + images_dir)

    landmark_features = np.array(all_features)
    eye_labels = (np.array(eye_labels))
    face_shape_labels = (np.array(face_shape_labels))

    txt = "Successfully extracted features from: {percent:.2f}% of images"
    print(txt.format(percent=(len(eye_labels) / len(image_paths)) * 100))

    return landmark_features, eye_labels, face_shape_labels

def extract_eye_features(data_set):
    """
    This function extracts eye features for all images in the folder specified by 'data_set' parameter.
    :return:
        eye_eatures:  array of 3 values, blue, green and red mean values of the eye image
        labels:      an array containing the eye label for each image
    """
    # Set up directory paths
    images_dir = os.path.join(data_set, 'img')
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(data_set, labels_filename), 'r')

    # strip lines of \n
    lines = [line.rstrip('\n') for line in labels_file]

    print(lines)

    # dictionary of eye labels for each image
    eye = {line.split('\t')[0]: int(line.split('\t')[1]) for line in lines[1:]}

    # progress bar
    printProgressBar(0, len(image_paths), prefix="Extracting features from: " + images_dir)

    # extracting features from eye_feature_extractor
    if os.path.isdir(images_dir):
        all_features = []
        eye_labels = []
        for i, img_path in enumerate(image_paths):
            file_name = img_path.split('.')[1].split('/')[-1]

            # load image using cv2
            img = cv2.imread(img_path)

            if file_name != '':
                # getting features for files with a name
                features = eye_extractor(img, eye[file_name])

            if features is not None:
                if file_name != '':
                    all_features.append(features)
                    eye_labels.append(eye[file_name])

            printProgressBar(i + 1, len(image_paths), prefix="Extracting features from: " + images_dir)

    eye_features = np.array(all_features)
    eye_labels = (np.array(eye_labels))

    txt = "Successfully extracted features from: {percent:.2f}% of images"
    print(txt.format(percent=(len(eye_labels) / len(image_paths)) * 100))

    return eye_features, eye_labels