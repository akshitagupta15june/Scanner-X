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

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

edged_image = cv2.Canny(blurred_image, 0, 50)
original_edged = edged_image.copy()
file_name_path = ‘file path to save image’ + str(count) + ‘.png’
cv2.imwrite(file_name_path, original_edged)


(contours, _) = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

def rectify(h):
h = h.reshape((4,2))
hnew = np.zeros((4,2),dtype = np.float32)
add = h.sum(1)
hnew[0] = h[np.argmin(add)]
hnew[2] = h[np.argmax(add)]
diff = np.diff(h,axis = 1)
hnew[1] = h[np.argmin(diff)]
hnew[3] = h[np.argmax(diff)]
return hnew

for c in contours:
p = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.02 * p, True)

if len(approx) == 4:
target = approx
break
