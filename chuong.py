import cv2 as cv
from imutils.video import FPS
fps = FPS().start()
cap = cv.VideoCapture("Ngay1-240p.mp4")
while True:
    frame = cap.read()
    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'): break
fps.update()
fps.stop()