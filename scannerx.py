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
