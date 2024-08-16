import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN

# Initialize FaceNet model and MTCNN detector
embedder = FaceNet()
detector = MTCNN()

def extract_face(filename, required_size=(160, 160)):
    # Load image from file
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None
    
    # Extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    # Extract the face
    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    return face

def get_embeddings(model, face_pixels):
    # Scale pixel values
    face_pixels = face_pixels.astype('float32')
    # Standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # Transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # Make prediction to get embeddings
    yhat = model.embeddings(samples)
    return yhat[0]

def load_dataset(directory):
    X, y = [], []
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            face = extract_face(filepath)
            if face is None:
                continue
            embedding = get_embeddings(embedder, face)
            X.append(embedding)
            y.append(subdir)
    return np.array(X), np.array(y)

# Load the dataset of driver images
trainX, trainy = load_dataset('model_train/face_detection/driver_pics')
