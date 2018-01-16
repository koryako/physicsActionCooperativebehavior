import cv2
import numpy as np
classifier=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')


image_src=cv2.imread('timg.jpeg')

cv2.imshow('test',image_src)







