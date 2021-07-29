import cv2
import numpy as np
import glob
from pathlib import Path

path=Path(“file complete path”)
image=[]
count=0
for imagepath in path.glob(“*.png”):
count=count+1
image=cv2.imread(str(imagepath))

image = cv2.resize(image, (1500, 880))
original = image.copy()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blurred the image for clear processing
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#here edge detection is done by “canny edge detection technique. By this image can be easily segmented.
edged_image = cv2.Canny(blurred_image, 0, 50)
original_edged = edged_image.copy()
file_name_path = ‘file path to save image’ + str(count) + ‘.png’
cv2.imwrite(file_name_path, original_edged)
