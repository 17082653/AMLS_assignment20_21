import os
import numpy as np
from keras.preprocessing import image
import cv2
from matplotlib import pyplot as plt

global basedir, image_paths, target_size
basedir = '../Datasets'
images_dir = os.path.join(basedir,'cartoon_set/img')
labels_filename = 'labels.csv'
target_size = None

file = ['0.png', '5.png', '10.png', '20.png', '30.png', '40.png', '94.png', '98.png', '140.png']
# sunglasses, 4-black, 2-green, 2-green, sunglasses, 4-black, 1-blue, 0-brown, 3-gray
for file in file:
    # x = 175 - 230,
    # y = 245 - 285

    # gray = cv2.imread(test_image)
    # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

    test_image = os.path.join(images_dir, file)
    print(test_image)

    img = image.img_to_array(image.load_img(test_image, target_size=target_size, interpolation='bicubic'))
    resized_image = img.astype('uint8')

    crop_img = resized_image[245:280, 175:230]

    plt.figure()
    plt.imshow(crop_img)  # cmap='gray'
    plt.show()


