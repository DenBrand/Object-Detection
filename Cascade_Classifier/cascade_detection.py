import cv2
import numpy as np
import os.path

# load the trained model
classifier = cv2.CascadeClassifier(os.path.join('custom_data', 'results', 'cascade.xml'))

# load an image
img = cv2.imread(os.path.join('custom_data', 'positives', '2020-12-4_4h11min30sec4031.jpg'))

# detect objects
rectangles = classifier.detectMultiScale(img)

# draw detection results onto original image
#for (x, y, w, h) in rectangles:
#    print(str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h))

#    img = cv2.rectangle(img, (x, y), (x + w, y + h), 1)

# display the image
#cv2.imshow('Matches', img)