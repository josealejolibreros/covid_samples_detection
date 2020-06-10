import cv2
import numpy as np
import operator



def nothing(x):
    # any operation
    pass

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
    frame = cv2.imread('4.jpeg')
    W = 500.
    height, width, depth = frame.shape
    image_size = height * width
    imgScale = W / width
    newX, newY = frame.shape[1] * imgScale, frame.shape[0] * imgScale
    frame = cv2.resize(frame, (int(newX), int(newY)))
    frame.resize()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = 0
    l_s = 134
    l_v = 0
    u_h = 83
    u_s = 255
    u_v = 255

    '''
    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")
    '''



    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    # Contours detection
    if int(cv2.__version__[0]) > 3:
        # Opencv 4.x.x
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Opencv 3.x.x
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if area > image_size * 0.1:
            print(approx)
            #cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)


            if len(approx) == 3:
                cv2.putText(frame, "Triangle", (x, y), font, 1, (0, 0, 0))
            elif len(approx) >= 4:
                #print(approx.ravel())
                width_red_grid_rectangle = approx.ravel()[len(approx.ravel()) - 2] - approx.ravel()[0]
                height_red_grid_rectangle = approx.ravel()[3] - approx.ravel()[1]
                start_point_rectangle = (x,y)
                cv2.rectangle(frame, start_point_rectangle,
                              (x+width_red_grid_rectangle,y+height_red_grid_rectangle), (0, 0, 0), 5)

                width_red_grid = approx.ravel()[len(approx.ravel())-2] - approx.ravel()[0] - 40
                height_red_grid = approx.ravel()[3] - approx.ravel()[1] - 60


                start_point =  (int(x) + 1, int(y) + 10)
                start_point_y = (int(x) + 1, int(y) + 10)
                box_size = (int(width_red_grid/8),int(height_red_grid/12))
                print(box_size)
                stepx = (5, 0)
                stepy = (0, 5)

                #This moves from left to right
                #And top-down
                number = 1
                for i in range(1,13):
                    start_point = tuple(map(operator.add,
                                            map(operator.mul, (0, i - 1), box_size),
                                            tuple(map(operator.add,
                                                      map(operator.mul, (0, i - 1), stepy),
                                                      start_point_y))))
                    for j in range(1,9):
                        number = number + 1
                        start_point_ij = tuple(map(operator.add,
                                                   map(operator.mul, (j-1, 0), box_size),
                                                   tuple(map(operator.add,
                                                             map(operator.mul, (j-1, 0), stepx),
                                                             start_point))))
                        end_point_ij = tuple(map(operator.add,start_point_ij,box_size))

                        crop_img = frame[start_point_ij[1]:end_point_ij[1], start_point_ij[0]:end_point_ij[0]]
                        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

                        '''
                        cv2.imshow("hsv", hsv)
                        '''

                        channel = crop_img[:, :, 1]
                        lower_red = np.array([140, 40, 40])
                        upper_red = np.array([180, 255, 255])
                        mask = cv2.inRange(hsv, lower_red, upper_red)

                        # The bitwise and of the frame and mask is done so
                        # that only the blue coloured objects are highlighted
                        # and stored in res
                        # res = cv2.bitwise_and(crop_img, crop_img, mask=mask)

                        '''
                        cv2.imshow("mask", mask)
                        #cv2.imshow("res", res)
                        cv2.waitKey()
                        '''

                        # b = crop_img.copy()
                        # set green and red channels to 0
                        # b[:, :, 0] = 0
                        # b[:, :, 1] = 0
                        # newcrop = b[:, :, 1]

                        # ret, thresh1 = cv2.threshold(newcrop, 40, 150, cv2.THRESH_BINARY)
                        count = np.count_nonzero(mask == 255)
                        if count > 80:
                            print("Region #{}".format(number), ": region azul{}".format(count))


                        #print(start_point_ij)
                        cv2.rectangle(frame, start_point_ij, end_point_ij, (0, 255, 0), 2)



                #cv2.putText(frame, "Rectangle", (x, y), font, 1, (0, 0, 0))
            #elif 10 < len(approx) < 20:
                #cv2.putText(frame, "Circle", (x, y), font, 1, (0, 0, 0))


    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()



'''
Rectangulo de imagen 2
[[ 54   0]]

 [[ 60 646]]

 [[476 665]]

 [[475   0]]]
 '''