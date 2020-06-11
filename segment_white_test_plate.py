import cv2
import numpy as np
import operator
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def nothing(x):
    # any operation
    pass


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 66, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 134, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 243, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    frame = cv2.imread('1_2.jpg')
    W = 500.
    height, width, depth = frame.shape
    image_size = height * width
    imgScale = W / width
    newX, newY = frame.shape[1] * imgScale, frame.shape[0] * imgScale
    frame = cv2.resize(frame, (int(newX), int(newY)))
    frame.resize()
    frame_rectangles = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = 0
    l_s = 134
    l_v = 0
    u_h = 83
    u_s = 255
    u_v = 255


    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")




    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    # Contours detection
    if int(cv2.__version__[0]) > 3:
        # Opencv 4.x.x
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(contours)
    else:
        # Opencv 3.x.x
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        crop_img_bgr = frame[y:y+h, x:x+w]
        cv2.rectangle(frame_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

        crop_img = cv2.cvtColor(crop_img_bgr, cv2.COLOR_BGR2RGB)

        foo=0
        modified_image = cv2.resize(crop_img, (150, 100), interpolation=cv2.INTER_AREA)
        modified_image_bgr = cv2.resize(crop_img_bgr, (150, 100), interpolation=cv2.INTER_AREA)
        cv2.imshow('img', modified_image_bgr)
        modified_image = crop_img.reshape(crop_img.shape[0] * crop_img.shape[1], 3)
        #R_mean = np.mean(modified_image[0])
        #G_mean = np.mean(modified_image[0])
        #B_mean = np.mean(modified_image[0])
        clf = KMeans(n_clusters=2)
        labels = clf.fit_predict(modified_image)

        counts = Counter(labels)

        center_colors = clf.cluster_centers_
        print(center_colors)
        # We get ordered colors by iterating through the keys
        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        rgb_colors = [ordered_colors[i] for i in counts.keys()]

        #if (show_chart):

        fig = plt.figure()
        # attach a non-interactive Agg canvas to the figure
        # (as a side-effect of the ``__init__``)





        fig = plt.figure(figsize=(8, 6))
        plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
        fig.canvas.draw()

        # convert canvas to image
        img_c = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img_c = img_c.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img_c = cv2.cvtColor(img_c, cv2.COLOR_RGB2BGR)

        # display image with opencv or any operation you like
        cv2.imshow("plot", img_c)




        #plt.show()
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.imshow("Frame", frame_rectangles)
    #cv2.imshow("Mask", mask)



cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()



