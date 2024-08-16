import json
import pickle
import numpy as np
from sklearn.svm import SVC
from embedding_extract import extract_face, get_embeddings
from keras_facenet import FaceNet

def extract_face_embeddings(image_paths, embedder):
    embeddings = []
    for image_path in image_paths:
        face = extract_face(image_path)
        if face is not None:
            embedding = get_embeddings(embedder, face)
            embeddings.append(embedding)
    return embeddings

def update_embedding_database(new_embeddings, driver_id, database_file='embeddings.json'):
    try:
        with open(database_file, 'r') as f:
            database = json.load(f)
    except FileNotFoundError:
        database = {}

    if driver_id not in database:
        database[driver_id] = []

    database[driver_id].extend(new_embeddings)

    with open(database_file, 'w') as f:
        json.dump(database, f)

def retrain_model(database_file='embeddings.json'):
    with open(database_file, 'r') as f:
        database = json.load(f)

    X = []
    y = []
    for driver_id, embeddings in database.items():
        for embedding in embeddings:
            X.append(embedding)
            y.append(driver_id)

    X = np.array(X)
    y = np.array(y)

    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    
    with open('models/driver_identity_verification.pkl', 'wb') as f:
        pickle.dump(model, f)

# Initialize your embedder here
embedder = FaceNet() # Replace with actual embedder initialization
