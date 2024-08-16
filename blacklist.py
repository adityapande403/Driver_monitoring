import pickle
from sklearn.preprocessing import LabelEncoder
from face_detection.embedding_extract import extract_face, get_embeddings, embedder

# Load identity model
with open('models/driver_identity_verification.pkl', 'rb') as f:
    identity_model = pickle.load(f)

# Load identity encoder
with open('models/identity_encoder.pkl', 'rb') as f:
    identity_encoder = pickle.load(f)

driver_scores = {}  # Key: Driver ID, Value: Score
blacklist_threshold = 50
blacklisted_drivers = set()  # Set of blacklisted driver IDs

def update_driver_score(driver_id, behavior_type):
    if driver_id not in driver_scores:
        driver_scores[driver_id] = 0

    if behavior_type == 'drowsiness':
        driver_scores[driver_id] += 5
    elif behavior_type == 'smoking':
        driver_scores[driver_id] += 10
    elif behavior_type == 'poor_posture':
        driver_scores[driver_id] += 5
    elif behavior_type == 'phone_usage':
        driver_scores[driver_id] += 10

    # Check if driver exceeds blacklist threshold
    if driver_scores[driver_id] > blacklist_threshold:
        blacklisted_drivers.add(driver_id)
        print(f"Driver {driver_id} has been blacklisted!")

def verify_driver(image_path):
    face = extract_face(image_path)
    if face is None:
        print("No face detected")
        return None, 0

    embedding = get_embeddings(embedder, face)
    preds = identity_model.predict_proba([embedding])
    driver_id = np.argmax(preds)
    confidence = preds[0][driver_id]

    if driver_id in blacklisted_drivers:
        print(f"Driver {driver_id} is blacklisted!")
        return driver_id, confidence

    driver_name = identity_encoder.inverse_transform([driver_id])[0]
    print(f'Driver: {driver_name}, Confidence: {confidence*100:.2f}%')
    return driver_id, confidence

import json

def save_blacklisted_drivers():
    with open('blacklisted_drivers.json', 'w') as f:
        json.dump(list(blacklisted_drivers), f)

def load_blacklisted_drivers():
    global blacklisted_drivers
    if os.path.exists('blacklisted_drivers.json'):
        with open('blacklisted_drivers.json', 'r') as f:
            blacklisted_drivers = set(json.load(f))