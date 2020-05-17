import numpy as np
import cv2

# Create a black image
img = np.zeros((480,640,3), np.uint8)

src = np.float32(
		[[11,479],
		[231,100],
		[407,100],
		[629, 479]])

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
print( pts )
pts = pts.reshape((-1,1,2))
print(pts )
cv2.polylines(img,[pts],True,(0,255,255))

pts = np.array( src , np.int32)
print( pts )
pts = pts.reshape((-1,1,2))
print( pts )
cv2.polylines(img,[pts],True,(0,255,255))

cv2.imshow('line',img)
cv2.waitKey()