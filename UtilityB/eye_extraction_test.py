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

# file = ['0.png', '5.png', '10.png', '20.png', '30.png', '40.png', '94.png', '98.png', '140.png']
# sunglasses, 4-black, 2-green, 2-green, sunglasses, 4-black, 1-blue, 0-brown, 3-gray
#file = ['5.png', '10.png', '20.png', '94.png', '98.png', '140.png']
dic = {0:['4.png'], 1: ['94.png'], 2:['10.png', '3.png'], 3:['140.png'], 4:['5.png', '40.png']}

# define the list of boundaries
# RGB
# dark - actual - light
# green   57, 75, 34 - 103, 136 59 - 145, 170 111  -> mean:
# brown   97, 55, 41 - 113, 63, 47 - 143 102 87    -> mean:
# gray    87 98 95 - 137 154 150 - 168 180 176     -> mean:
# blue    27 55 75 - 46 103 150 - 86, 135, 172     -> mean:
# black             - 0 0 0 - 53 53 53

# OPENCV WORK IN BGR - SO IN REVERSE
boundaries = [
    ([41, 55, 97], [87, 102, 143]),
    ([75, 55, 27], [172, 135, 86]),
    ([34, 75, 57], [111, 170, 145]),
	([95, 98, 87], [176, 180, 168]),
    ([0, 0, 0], [53, 53, 53]),
]

# boundaries = [
# 	([59, 136, 103], [59, 136, 103]),
# 	([47, 63, 113], [47, 63, 113]),
# 	([150, 154, 137], [150, 154, 137]),
# 	([150, 103, 46], [150, 103, 46]),
#     ([0, 0, 0], [0, 0, 0]),
# ]

for color, file in dic.items():
    for file in file:
        # crop region for left eye
        # x = 175 - 230
        # y = 245 - 280

        test_image = os.path.join(images_dir, file)
        print(test_image)

        # KERAS READ
        # img = image.img_to_array(image.load_img(test_image, target_size=target_size, interpolation='bicubic'))
        # image = img.astype('uint8')

        # OPENCV READ
        image = cv2.imread(test_image)
        crop_img = image[245:280, 175:230]

        # plt.figure()
        # plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # cmap='gray'
        # plt.show()

        # Set boundary based on color
        lower, upper = boundaries[color]

        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(crop_img, lower, upper)
        masked_image = cv2.bitwise_and(crop_img, crop_img, mask=mask)

        # print what color the eye is by checking mean BGR:
        mean = cv2.mean(masked_image)
        print("color: ", color, "; mean: ", mean, "; avg: ", np.average(mean))

        # show the image
        cv2.imshow("mask", np.hstack([crop_img, masked_image]))
        cv2.waitKey(0)

        #hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

        plt.figure()
        plt.imshow(masked_image)  #cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        plt.show()








