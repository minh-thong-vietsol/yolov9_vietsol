import cv2 as cv
import numpy as np
import imutils
from imutils.video import FPS
import triangulation as tri
import calibration

# Distance constants
KNOWN_DISTANCE = 45  # INCHES
#PERSON_WIDTH = 16  # INCHES
#MOBILE_WIDTH = 3.0  # INCHES
MOTORBIKE_WIDTH = 50

# Stereo vision setup parameters
frame_rate = 120  # Camera frame rate (maximum at 120 fps)
B = 9  # Distance between the cameras [cm]
f = 8  # Camera lense's focal length [mm]
alpha = 56.6  # Camera field of view in the horisontal plane [degrees]

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open("yolo.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny-custom_final.weights', 'yolov4-tiny-custom.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

center_point = 0
# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_names[classid[0]], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        #if classid == 0:  # person class id
            #data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
        #elif classid == 67:
            #data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
        if classid == 0:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data.
    return data_list


#def focal_length_finder(measured_distance, real_width, width_in_rf):
    #focal_length = (width_in_rf * measured_distance) / real_width

    #return focal_length


# distance finder function
#def distance_finder(focal_length, real_object_width, width_in_frmae):
    #distance = (real_object_width * focal_length) / width_in_frmae
    #return distance


# reading the reference image from dir
#ref_person = cv.imread('ReferenceImages/image14.png')
#ref_mobile = cv.imread('ReferenceImages/image5.png')
#ref_motorbike = cv.imread('ReferenceImages/image4.png')

#mobile_data = object_detector(ref_mobile)
#mobile_width_in_rf = mobile_data[0][1]

#person_data = object_detector(ref_person)
#person_width_in_rf = person_data[0][1]

#motorbike_data = object_detector(ref_motorbike)
#motorbike_width_in_rf = 10

#print(f" motorbike width in pixel: {motorbike_width_in_rf}  ")

# finding focal length
#focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

#focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

#focal_motorbike = focal_length_finder(KNOWN_DISTANCE, MOTORBIKE_WIDTH, motorbike_width_in_rf)

cap_left = cv.VideoCapture(1, cv.CAP_DSHOW)
cap_right = cv.VideoCapture(0, cv.CAP_DSHOW)
fps = FPS().start()
while True:
    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()
    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)
    data_right = object_detector(frame_right)
    data_left = object_detector(frame_left)
    center_right = 0
    center_left = 0

    for d in data_left:
        #if d[0] == 'person':
            #distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            #x1, y = d[2]
        if d[0] == 'xe may':
            #distance = distance_finder(focal_motorbike, MOTORBIKE_WIDTH, d[1])
            x_left, y = d[2]
            center_point_left = center_point
        #cv.rectangle(frame_left, (x, y - 3), (x + 150, y + 23), BLACK, -1)
        #cv.putText(frame_left, f'Dis: {round(distance, 2)} m', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

    for d in data_right:
        #if d[0] == 'person':
            #distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            #x, y = d[2]
        if d[0] == 'xe may':
            #distance = distance_finder(focal_motorbike, MOTORBIKE_WIDTH, d[1])
            x_right, y= d[2]
            center_point_right = center_point
        #cv.rectangle(frame_right, (x, y - 3), (x + 150, y + 23), BLACK, -1)
        #cv.putText(frame_right, f'Dis: {round(distance, 2)} m', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)
    cv.imshow("frame right", frame_right)
    cv.imshow("frame left", frame_left)
    #if not ret_right or not ret_left:
       # cv.putText(frame_right, "TRACKING LOST", (75, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv.putText(frame_left, "TRACKING LOST", (75, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #else:
        # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
        # All formulas used to find depth is in video presentaion
    depth = tri.find_depth(100, 0, frame_right, frame_left, B, f, alpha)

    cv.putText(frame_right, "Distance: " + str(round(depth, 1)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 3)
    cv.putText(frame_left, "Distance: " + str(round(depth, 1)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 3)
        # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
    print("Depth: ", str(round(depth, 1)))

    key = cv.waitKey(1)
    if key == ord('q'):
        break
    fps.update()
fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("FPS: {:.2f}".format(fps.fps()))
cv.destroyAllWindows()
cap_right.release()
cap_left.release()











































