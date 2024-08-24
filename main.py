import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained object detection model (e.g., YOLO, SSD)
def load_object_detection_model():
    # Example: Load YOLO model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Process video to detect objects
def detect_objects(net, output_layers, frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Extract object information
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = center_x - w // 2, center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return class_ids, confidences, boxes

# Load action recognition model
def load_action_recognition_model():
    # Example: Load a pre-trained model (like a CNN-LSTM model)
    model = tf.keras.models.load_model('action_recognition_model.h5')
    return model

# Recognize actions from video frames
def recognize_action(model, video_frames):
    # Preprocess video frames
    video_frames = preprocess_frames(video_frames)
    
    # Predict actions
    predictions = model.predict(video_frames)
    
    # Interpret predictions (define your threshold for shoplifting)
    is_shoplifting = np.argmax(predictions) == 1  # Assuming label '1' is for shoplifting
    return is_shoplifting

# Main function to process video
def process_video(video_path):
    net, output_layers = load_object_detection_model()
    action_model = load_action_recognition_model()
    
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects in the frame
        class_ids, confidences, boxes = detect_objects(net, output_layers, frame)
        
        # Recognize actions (you will need to define how you extract and feed video frames to your action model)
        is_shoplifting = recognize_action(action_model, frame)
        
        if is_shoplifting:
            print("Shoplifting detected!")
            # You can add further processing here, like sending alerts
        
        # Display frame (optional)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Helper function to preprocess frames (implement as needed)
def preprocess_frames(frames):
    # Perform any necessary preprocessing (e.g., resizing, normalizing)
    return frames

# Run the shoplifting detection on a sample video
process_video("path/to/video.mp4")
