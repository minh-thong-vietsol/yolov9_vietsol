#this file is to test functions from my_functions.py

from email.mime import image
from matplotlib.pylab import annotations
from numpy import imag
import my_functions as mf

# Test extract_frames function
#mf.extract_frames('/Users/tom/Documents/vietsol/2. Object Detection/ObjectDetection/mthong_detect/yolov9/data/video_vinfast/Ngay3-720p.mp4', '/Users/tom/Documents/vietsol/2. Object Detection/ObjectDetection/mthong_detect/yolov9/data/video_vinfast/Ngay3-720p')
image_folder = '/Users/tom/Documents/vietsol/2. Object Detection/ObjectDetection/mthong_detect/yolov9/data/video_vinfast/Ngay3-720p'
annotations_folder = '/Users/tom/Documents/vietsol/2. Object Detection/ObjectDetection/mthong_detect/yolov9/runs/detect/exp7/labels'
output_folder = '/Users/tom/Documents/vietsol/2. Object Detection/ObjectDetection/mthong_detect/yolov9/data/vinfast_data'
mf.split_data(image_folder, annotations_folder, output_folder)
