import cv2
import numpy as np
import tensorflow as tf
import kagglehub

# Download and load the PoseNet model
model = kagglehub.model_download("tensorflow/posenet-mobilenet/tfLite/float-075")

def run_pose_estimation(image):
    image = tf.image.resize(image, [257, 257])
    image = tf.expand_dims(image, axis=0)
    output = model(image)
    return output

def process_pose_output(output):
    keypoints = output['output_0'].numpy().reshape(-1, 3)
    return keypoints

def detect_posture(frame):
    height, width = frame.shape[:2]
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = run_pose_estimation(image)
    keypoints = process_pose_output(output)

    # Example posture detection (simplified)
    shoulder_x = keypoints[5, 0]  # Right shoulder x-coordinate
    if shoulder_x < width * 0.5:
        posture = "Upright"
    else:
        posture = "Slouched"

    return posture
