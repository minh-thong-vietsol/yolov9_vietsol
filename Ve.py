import cv2
import numpy as np
cam = cv2.imread("1111 - Copy.jpg")
h, w, c = cam.shape
pts = np.array([[0, h], [5*w/13, 6*h/13],
                        [8*w/13, 6*h/13], [w, h],
                        ],
                       np.int32)
pts1 = np.array([[0, h], [4 * w / 13, 13 * h / 23],
                        [9* w / 13, 13* h / 23], [w, h],
                        ],
                       np.int32)

pts = pts.reshape((-1, 1, 2))
pts1 = pts1.reshape((-1, 1, 2))

isClosed = True

        # Blue color in BGR
color = (0, 255, 255)
color1 = (0,0,255)

        # Line thickness of 2 px
thickness = 2

        # Using cv2.polylines() method
        # Draw a Blue polygon with
        # thickness of 1 px
cv2.polylines(cam, [pts],isClosed, color, thickness)
cv2.polylines(cam, [pts1], isClosed, color1, thickness)
cv2.imshow("z",cam)
cv2.waitKey()