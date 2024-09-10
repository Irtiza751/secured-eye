######## video issue resolved ########

from ultralytics import YOLO
import supervision as sv
import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
from collections import deque
import subprocess

detection_count = {}

def load_model_and_data():
    model = YOLO("14-JUN-best-2.pt")
    CLASS_NAMES_DICT = {0: 'Normal', 1: 'Shoplifting'}
    box_annotator = sv.BoxAnnotator()
    return model, CLASS_NAMES_DICT, box_annotator

def save_frames_as_video(frames, save_path, frame_rate):
    # Get the frame size from the first frame
    print(frames)
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'h264')  # H.264 codec
    video_writer = cv2.VideoWriter(save_path, fourcc, frame_rate, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    # clip = ImageSequenceClip(frames, fps=frame_rate)
    # clip.write_videofile("output_video.mp4", codec="h264")

def process_frame(frame, detections, box_annotator, class_names_dict, camera_id):
    global detection_count

    if camera_id not in detection_count:
        detection_count[camera_id] =0


    if any(item[3] == 1 for item in detections):
        if detection_count[camera_id]==0:
            detection_count[camera_id] +=1
            labels = [f"{class_names_dict[class_id]} {confidence:.2f}"
                    for _, _, confidence, class_id, _, _ in detections]
                
            annotated_frame = box_annotator.annotate(frame, detections, labels)
            current_time = time.time()
            cv2.imwrite(f"images/{current_time}.jpg", annotated_frame)
            print(f"Image saved: {current_time}")

def main():
    file_path = 'sources.streams'
    with open(file_path, 'r') as file:
        streams = file.read()
    lines = streams.splitlines()
    number_of_lines = len(lines)
    print("length =", number_of_lines)
    model, class_names_dict, box_annotator = load_model_and_data()
    source = "20240904161220.mp4"
    count = 1
    results = model(source, stream=True, classes=[0, 1], conf=0.5, imgsz=640, show=True, half=True)

    for result in results:
            detections = sv.Detections.from_ultralytics(result)
            #print(f"Result frame {count}: {len(result)} detections")
            frame = result.orig_img
            process_frame(frame, detections, box_annotator, class_names_dict, count)
            count += 1
            if count == int(number_of_lines) + 1:
                count = 1

if __name__ == "__main__":
    main()
