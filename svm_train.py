from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from embedding_extract import load_dataset
import joblib
trainX, trainy = load_dataset('driver_pics')

# Encode the labels
encoder = LabelEncoder()
encoder.fit(trainy)
trainy_enc = encoder.transform(trainy)

# Train an SVM classifier on the embeddings
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy_enc)

# Save the model for future use

joblib.dump((model, encoder), 'models/driver_identity_verification.pkl')
