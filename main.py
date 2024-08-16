import os
import cv2
import numpy as np
import pickle
import json
from keras.models import load_model
from pygame import mixer
from model_train.posture_detection import detect_posture
from model_train.face_detection.embedding_extract import extract_face, get_embeddings, embedder
from sklearn.preprocessing import LabelEncoder


# Initialize Pygame mixer for playing sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascade files for face and eyes detection
face_cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_righteye_2splits.xml')

# Load pre-trained models
drowsiness_model = load_model('models/cnncat2.h5')
smoking_model = load_model('models/smoke_detection.h5')
identity_model = pickle.load(open('models/driver_identity_verification.pkl', 'rb'))
posture_model = pickle.load(open('models/posture_detection.pkl', 'rb'))

# Load identity encoder
identity_encoder = LabelEncoder()

# Initialize variables
driver_scores = {}
blacklist_threshold = 50
blacklisted_drivers = set()

# Load blacklisted drivers
def load_blacklisted_drivers():
    global blacklisted_drivers
    if os.path.exists('blacklisted_drivers.json'):
        with open('blacklisted_drivers.json', 'r') as f:
            blacklisted_drivers = set(json.load(f))

load_blacklisted_drivers()

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

# Initialize webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2

try:
 while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    left_eye = leye_cascade.detectMultiScale(gray)
    right_eye = reye_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Drowsiness detection
        rpred = lpred = 1  # Assume open by default
        for (x, y, w, h) in right_eye:
            r_eye = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255.0
            r_eye = np.expand_dims(r_eye.reshape(24, 24, -1), axis=0)
            rpred = np.argmax(drowsiness_model.predict(r_eye))
            break

        for (x, y, w, h) in left_eye:
            l_eye = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255.0
            l_eye = np.expand_dims(l_eye.reshape(24, 24, -1), axis=0)
            lpred = np.argmax(drowsiness_model.predict(l_eye))
            break

        if rpred == 0 and lpred == 0:  # Both eyes closed
            score += 1
            update_driver_score(driver_id, 'drowsiness')
            if score > 7:
                try:
                    sound.play()
                except Exception as e:
                    print(f"Error playing sound: {e}")

                thicc = thicc + 2 if thicc < 16 else thicc - 2
                if thicc < 2:
                    thicc = 2
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

        else:
            score -= 1

        if score < 0:
            score = 0

        

# Smoking Detection
        if smoking_model.predict(frame):
          update_driver_score(driver_id, 'smoking')
          print("Smoking detected!")

# Posture Detection
        
          posture = detect_posture(frame)
        if posture == "Slouched":
            update_driver_score(driver_id, 'poor_posture')

        # Run identity verification
        driver_id, confidence = verify_driver(frame)
        if confidence < 0.6:
            print("Low confidence in driver identity!")

        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Identity Verification
        driver_id, confidence = verify_driver(frame)
        if driver_id is not None:
          print(f"Driver {driver_id} verified with confidence: {confidence*100:.2f}%")
    

       

          cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
          cv2.imshow('Driver Monitoring', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break


finally:

   cap.release()
   cv2.destroyAllWindows()

# Save blacklisted drivers at the end
   with open('blacklisted_drivers.json', 'w') as f:
     json.dump(list(blacklisted_drivers), f)
