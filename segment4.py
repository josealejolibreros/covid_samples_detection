# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import cv2


img = cv2.imread('6.jpeg')
W = 500.
height, width, depth = img.shape
imgScale = W/width
newX,newY = img.shape[1]*imgScale, img.shape[0]*imgScale
newimg = cv2.resize(img,(int(newX),int(newY)))
img.resize()
#cv2.imshow("dede",newimg)






b = newimg.copy()
# set green and red channels to 0
#b[:, :, 1] = 0
b[:, :, 2] = 0


g = newimg.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0

r = newimg.copy()
# set blue and green channels to 0
r[:, :, 0] = 0
r[:, :, 1] = 0


# RGB - Blue
#cv2.imshow('B-RGB', b)

# RGB - Green
#cv2.imshow('G-RGB', g)

# RGB - Red
#cv2.imshow('R-RGB', r)

#cv2.waitKey(0)






image=newimg


# load the image and perform pyramid mean shift filtering
# to aid the thresholding step

shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
#cv2.imshow("Input", image)

# convert the mean shift image to grayscale, then apply
# Otsu's thresholding
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#cv2.imshow("Thresh", thresh)

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,
	labels=thresh)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue

	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255

	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	c = max(cnts, key=cv2.contourArea)

	((x, y), r) = cv2.minEnclosingCircle(c)
	h=20
	w=20
	crop_img = image[int(y)-h:int(y) + h, int(x)-w:int(x) + w]

	#if label == 72:

	hsv=cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

	'''
	cv2.imshow("hsv", hsv)
	'''

	channel = crop_img[:,:,1]
	lower_red = np.array([140, 40, 40])
	upper_red = np.array([180, 255, 255])
	mask = cv2.inRange(hsv, lower_red, upper_red)

	# The bitwise and of the frame and mask is done so
	# that only the blue coloured objects are highlighted
	# and stored in res
	#res = cv2.bitwise_and(crop_img, crop_img, mask=mask)

	'''
	cv2.imshow("mask", mask)
	#cv2.imshow("res", res)
	cv2.waitKey()
	'''

	#b = crop_img.copy()
	# set green and red channels to 0
	#b[:, :, 0] = 0
	#b[:, :, 1] = 0
	#newcrop = b[:, :, 1]




	#ret, thresh1 = cv2.threshold(newcrop, 40, 150, cv2.THRESH_BINARY)
	count = np.count_nonzero(mask == 255)
	if count>80:
		print("Region #{}".format(label),": region azul{}".format(count))
	if label==74:
		cv2.imshow("hsv", hsv)
		cv2.waitKey()
		foo = 0

	#cv2.imshow("croppedblue", thresh1)
	#cv2.waitKey()

	# draw a circle enclosing the object

	cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
	cv2.rectangle(image, (int(x) - 20, int(y) - 20), (int(x + 20), int(y + 20)), (0, 255, 0), 2)

	cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)