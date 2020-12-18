import os.path
import numpy as np
import cv2
import dlib
import multiprocessing
import time
import matplotlib.pyplot as plt

from UtilityB import utilityB as utilB

# PATHS
# ========================
data = {}

cartoon_set = utilB.cartoon_set
images_dir = os.path.join(cartoon_set, 'img')

test_img = os.path.join(images_dir, '10.png')
# ========================

# RESIZING IMAGES
img = cv2.imread(test_img, cv2.IMREAD_UNCHANGED)
print('Original Dimensions : ', img.shape)

# set image dimensions
scale_percent = 40  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize test image (for testing only)

# resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# print('Resized Dimensions : ', resized.shape)
#
# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ========================


image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
image_names = [img_name.split('/')[4] for img_name in image_paths]
image_indexes = [image_indexes.split('.')[0] for image_indexes in image_names]
image_indexes.remove('')
image_indexes = [int(image_indexes) for image_indexes in image_indexes]

np.random.shuffle(image_indexes)

print("Reading images and drawing boxes...")
for index in image_indexes:
    # Read the image in memory and append it to the list
    img = cv2.imread(os.path.join(images_dir, str(index) + '.png'))
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    dlib_box = [ dlib.rectangle(left=1 , top=199, right=199, bottom=1) ]

    data[index] = (resized, dlib_box)

print('Number of Images and Boxes Present: {}'.format(len(data)))

# To check if drawing rectangles works
no_of_samples = 10
image_names = os.listdir(images_dir)

np.random.shuffle(data)

# cols = 5
# rows = int(np.ceil(no_of_samples / cols))
# plt.figure(figsize=(cols*cols, rows*cols))
# for i in range(no_of_samples):
#     d_box = data[i][1][0]
#     left, top, right, bottom = d_box.left(), d_box.top(), d_box.right(), d_box.bottom()
#     image = data[i][0]
#     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)
#     plt.subplot(rows, cols, i + 1);
#     plt.imshow(image[:, :, ::-1]);
#     plt.axis('off');
#
# plt.show()

# Training options
percent = 0.005

split = int(len(data) * percent)

images = [tuple_value[0] for tuple_value in data.values()]
bounding_boxes = [tuple_value[1] for tuple_value in data.values()]

options = dlib.shape_predictor_training_options()

options.be_verbose = True
options.tree_depth = 2 # 2, 4, 8
options.nu = 0.5 # change for bigger data set
# options.C = 5 # this is for object predictor only
options.cascade_depth = 6 # 6 to 18 - very important, but slows down badly
options.feature_pool_size = 400
options.num_test_splits = 5 # training time can explode with this - set to 50
options.oversampling_amount = 1 # can increase trianing time a lot - care
options.oversampling_translation_jitter = 0.1
options.num_threads = multiprocessing.cpu_count()




images = dlib.vector(images[:split])
boxes = dlib.vector(bounding_boxes[:split])

st = time.time()
print("Training detector...")
# start training
detector = dlib.train_shape_predictor(images, boxes, options)

print('Training Completed, Total Time taken: {:.2f} seconds'.format(time.time() - st))

# Testing detector
win_det = dlib.image_window()
win_det.set_image(detector)

print("Training Metrics: {}".format(dlib.test_simple_object_detector(images[:split], bounding_boxes[:split], detector)))
print("Testing Metrics: {}".format(dlib.test_simple_object_detector(images[split:], bounding_boxes[split:], detector)))


# Saving detector
file_name = 'cartoon_predictor.dat'
detector.save(file_name)

# RETRAIN ON 100% of data??
#detector = dlib.train_simple_object_detector(images, bounding_boxes, options)
#detector.save(file_name)














