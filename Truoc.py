import cv2 as cv
import numpy as np
import imutils
from imutils.video import FPS


from pygame import mixer
mixer.init()


# Distance constants/ Nhập chiều rộng xe trong thực tế với khoảng cách 3m từ Camera
KNOWN_DISTANCE = 2.5  # meters
XE_WIDTH = 0.5  # meters


# Object detector constant / Set thông số Confidence và NMS
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected / Set màu cho khung detect
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts / Set font chữ
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file / Load các class detect vào file class_names:
class_names = []
with open("yolo.names", "r") as f: 
    class_names = [cname.strip() for cname in f.readlines()]
#  setting up opencv net/ Set cấu hình, input file weight, cfg
yoloNet = cv.dnn.readNet('yolov4-tiny-custom_final.weights', 'yolov4-tiny-custom.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(412, 412), scale=1 / 255, swapRB=True)


# object detector function /method / Hàm nhận diện

def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # creating empty list to add objects data / Tạo datalist trống
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id / Set màu cho class
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_names[classid[0]], score)

        # draw rectangle on and label on object / Đóng khung cho class
        cv.rectangle(image, box, color, 1)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 1)

        # getting the data / Lấy dữ liệu tên, độ rộng, tọa độ
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0:  # xe class id
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])

        # if you want include more classes then you have to simply add more [elif] statements here
        # returning list containing the object data.
    return data_list

# Hàm tiêu cự
def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function / Hàm khoảng cách
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance
# refz = cv.imread('1.png')
# xe_data = object_detector(refz)
# xe_width_in_rf = xe_data[0][1]
xe_width_in_rf = 105 #Độ rộng xe trên frame ở khoảng cách 3m. Có thể tự điền số hoặc xác minh bằng 2 câu lệnh ngay trên

print(f" motorbike width in pixel: {xe_width_in_rf}  ") #Hiển thị độ rộng xe trên frame

# finding focal length/ Tiêu cự cho xe
focal_xe = focal_length_finder(KNOWN_DISTANCE, XE_WIDTH, xe_width_in_rf)


# Tiến hành detect trên video
cap = cv.VideoCapture("Ngay1-480p.mp4")


while True:
    ret, frame = cap.read()
    #frame = imutils.rotate(frame,180)
    #print(cap.read())
    #frame = cv.resize (frame, (640,480))
    h,w,c = frame.shape
    #print (h,w,c)


    roi = frame[0:400,0:w]
    frame1 = roi
    frameId = int(round(cap.get(1)))
    # print(frameId)
    if frameId % 5 == 0:
        data = object_detector(frame1)
        distance = 100
        for d in data:
            if d[0] == 'xe':
                distance = distance_finder(focal_xe, XE_WIDTH, d[1])
                x, y = d[2]

            cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
            cv.putText(frame, f'K/c: {round(distance, 2)} m', (x + 5, y + 13), FONTS, 0.48, GREEN, 1)
            print(distance)

        isClosed = True
        cv.imshow("1", frame)

        #print (frameId % multiplier)
#         if frameId % 50 == 0:
#             if distance < 10:
#                 mixer.music.load("Warning.wav")
#                 mixer.music.play()
#         key = cv.waitKey(1)
#         if key == ord('q'):
#             break
#         zfps.update()
#     zfps.stop()
#     cv.destroyAllWindows()
#     print("Elasped time: {:.2f}".format(zfps.elapsed()))
#     print("FPS: {:.2f}".format(zfps.fps()))



# p2 = threading.Thread(target=RUN,args=("Ngay1-480p.mp4","frame2"))
# p2.start()
# p3 = threading.Thread(target=RUN,args=("Ngay1-240p.mp4","frame3"))
# p3.start()



#cap.release()
