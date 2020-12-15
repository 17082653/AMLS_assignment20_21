import numpy as np
import pandas as pd

from Utility import utility as util

def save_data(dataset_dir, x_name, y_name):
    # Function for saving dataset into npy files
    landmark_features, gender_labels = util.extract_features_labels(dataset_dir)
    np.save(x_name, landmark_features)
    np.save(y_name, gender_labels)

def load_data(file_x, file_y):
    # Function for loading dataset from npy files
    return np.load(file_x), np.load(file_y)

def split_and_label_features(landmarks, forbidden=[]):
    """
    This function takes all 68 landmarks extracted from dlib facial detector, and splits them into
    specific facial features (eye, mouth, etc.)

    Takes in a ( num_of_extracted_images, 68, 2 ) sized array - ie. the output from 'extract_features_labels()'

    :return:
        landmarks: landmarks array, reshaped into 2-dimensions ( num_of_extracted_images, feat_num * 2)
        feat_num: number of features to use
    """
    # Reshaping image landmarks into 2 dimensions
    feat_ex = landmarks.reshape(len(landmarks), landmarks.shape[1] * 2)

    # Setting up dictionary with each facial feature in dlib
    faces = {'jaw': [],
             'right_eyebrow': [],
             'left_eyebrow': [],
             'nose': [],
             'right_eye': [],
             'left_eye': [],
             'mouth': []}

    # For each image, splitting up the 68*2 total landmarks into their corresponding facial feature
    for i in range(0, len(feat_ex)):
        faces['jaw'].append(feat_ex[i][:17 * 2])
        faces['right_eyebrow'].append(feat_ex[i][17 * 2:22 * 2])
        faces['left_eyebrow'].append(feat_ex[i][22 * 2:27 * 2])
        faces['nose'].append(feat_ex[i][27 * 2:36 * 2])
        faces['right_eye'].append(feat_ex[i][36 * 2:42 * 2])
        faces['left_eye'].append(feat_ex[i][42 * 2:48 * 2])
        faces['mouth'].append(feat_ex[i][48 * 2:68 * 2])

    # Converting the dictionary to a data frame
    df = pd.DataFrame(data=faces)

    # Removing dataframe feature columns based on forbidden_features
    forbidden_features = forbidden
    training_features = [feature for feature in list(faces) if feature not in forbidden_features]
    df = df[training_features]

    # Converting back to numpy array
    final_data = df.to_numpy()

    # Fixing the numpy array - arrays of each facial feature must be concatenated together into one array
    rows, cols = (len(final_data), len(np.concatenate(final_data[0], axis=0)))
    arr = [[0] * cols] * rows
    for i in range(0, len(final_data)):
        features = np.concatenate(final_data[i], axis=0)
        arr[i] = features

    # Converting to numpy array and returning
    landmarks = np.array(arr)
    feat_num = landmarks.shape[1]

    return landmarks, feat_num
