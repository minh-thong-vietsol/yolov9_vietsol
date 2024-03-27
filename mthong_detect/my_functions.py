import random
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import cv2
import shutil
from sklearn.model_selection import train_test_split
def extract_frames(video_path, output_folder):   
    
    """_summary_

    Args:
        video_path (_type_): _description_
        output_folder (_type_): _description_
    """
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Create output folder if it doesn't exist

    if os.path.exists(output_folder):
        print('Output folder already exists, do you want to overide it? y/n')
        if input() == 'y':
            os.rmdir(output_folder)
            os.makedirs(output_folder)
    else: 
        print(f"Making new folder {output_folder}")   
        os.makedirs(output_folder)
    
    # Initialize variables
    frame_count = 1 # Start counting frames from 1, this is to line up with yolo labeling
    video_name = os.path.basename(video_path).split('.mp4')[0]
    
    # Read the video frame by frame
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        # If frame is not read correctly or end of video, break the loop
        if not ret:
            break
        
        # Save frame as an image
        frame_path = os.path.join(output_folder, f"{video_name}_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    # Release the video capture object
    video_capture.release()
    
    print(f"Finished extracting {frame_count} frames from {video_path} to {output_folder}")

def move_files_to_folder(files, destination_folder):
    for file_path in files:
        try:
            shutil.move(file_path, destination_folder)
        except Exception as e:
            print(f"Error moving file {file_path}: {str(e)}")
def split_data(image_folder, annotation_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split data into train, validation, and test sets and move them to respective folders.
    """
    # Read images and annotations
    images = os.listdir(image_folder)
    annotations = os.listdir(annotation_folder)
    
    # Split data
    train_images, test_val_images, train_annotations, test_val_annotations = train_test_split(images, annotations, test_size=val_ratio+test_ratio)
    val_images, test_images, val_annotations, test_annotations = train_test_split(test_val_images, test_val_annotations, test_size=test_ratio/(val_ratio+test_ratio))
    
    # Create output folders
    output_image_train_folder = os.path.join(output_folder, 'images', 'train')
    output_image_val_folder = os.path.join(output_folder, 'images', 'val')
    output_image_test_folder = os.path.join(output_folder, 'images', 'test')
    output_annotation_train_folder = os.path.join(output_folder, 'labels', 'train')
    output_annotation_val_folder = os.path.join(output_folder, 'labels', 'val')
    output_annotation_test_folder = os.path.join(output_folder, 'labels', 'test')
    
    os.makedirs(output_image_train_folder, exist_ok=True)
    os.makedirs(output_image_val_folder, exist_ok=True)
    os.makedirs(output_image_test_folder, exist_ok=True)
    os.makedirs(output_annotation_train_folder, exist_ok=True)
    os.makedirs(output_annotation_val_folder, exist_ok=True)
    os.makedirs(output_annotation_test_folder, exist_ok=True)
    
    # Move files to respective folders
    move_files_to_folder([os.path.join(image_folder, img) for img in train_images], output_image_train_folder)
    move_files_to_folder([os.path.join(image_folder, img) for img in val_images], output_image_val_folder)
    move_files_to_folder([os.path.join(image_folder, img) for img in test_images], output_image_test_folder)
    move_files_to_folder([os.path.join(annotation_folder, ann) for ann in train_annotations], output_annotation_train_folder)
    move_files_to_folder([os.path.join(annotation_folder, ann) for ann in val_annotations], output_annotation_val_folder)
    move_files_to_folder([os.path.join(annotation_folder, ann) for ann in test_annotations], output_annotation_test_folder)