import os
import os.path
import cv2

if __name__ == '__main__':

    files = os.listdir(os.path.join('custom_data', 'images'))

    for file in files:

        img = cv2.imread(os.path.join('custom_data', 'images', file))

        resized_img = cv2.resize(img, (320, 320))

        cv2.imwrite(os.path.join('custom_data', 'images', file), resized_img)