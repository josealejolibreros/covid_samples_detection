import numpy as np
import cv2
from operator import itemgetter
from glob import glob
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )
def makebin(gray):
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    return cv2.bitwise_not(bin)
def find_squares(img):
    img = cv2.GaussianBlur(img, (11, 11), 0)
    squares = []
    points = []
    for gray in cv2.split(img):
        bin = makebin(gray)
        contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        corners = cv2.goodFeaturesToTrack(gray,len(contours)*4,0.2,15)
        cv2.cornerSubPix(gray,corners,(6,6),(-1,-1),(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,10, 0.1))
        for cnt in contours:
            cnt_len = cv2.arcLength(cnt, True)
            if len(cnt) >= 4 and cv2.contourArea(cnt) > 200:
                rect = cv2.boundingRect(cnt)
                if rect not in squares:
                    squares.append(rect)
    return squares, corners, contours
if __name__ == '__main__':
    for fn in glob('7.jpeg'):
        img = cv2.imread(fn)
        squares, corners, contours = find_squares(img)
        for p in corners:
            cv2.circle(img, (p[0][0],p[0][3]), 3, (0,0,255),2)
        squares = sorted(squares,key=itemgetter(1,0,2,3))
        areas = []
        moments = []
        centers = []
        for s in squares:
            areas.append(s[2]*s[3])
            cv2.rectangle( img, (s[0],s[1]),(s[0]+s[2],s[1]+s[3]),(0,255,0),1)
        for c in contours:
            moments.append(cv2.moments(np.array(c)))
        for m in moments:
            centers.append((int(m["m10"] // m["m00"]), int(m["m01"] // m["m00"])))
        for cent in centers:
            print(cent)
            cv2.circle(img, (cent[0],cent[1]), 3, (0,255,0),2)
        cv2.imshow('squares', img)
        ch = 0xFF & cv2.waitKey()
        if ch == 27:
            break
    cv2.destroyAllWindows()