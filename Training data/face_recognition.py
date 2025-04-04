import cv2
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
from PIL import Image

# Load dataset and train model
def train_model(data_dir="student/train"):
    faces = []
    labels = []
    
    for user_folder in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user_folder)
        if not os.path.isdir(user_path):
            continue
        
        user_id = user_folder.split("_")[1]  # Extract user ID
        
        for image_file in os.listdir(user_path):
            image_path = os.path.join(user_path, image_file)
            try:
                img = Image.open(image_path).convert("L")  # Convert to grayscale
                imageNp = np.array(img, 'uint8').flatten()  # Convert to 1D array
                
                faces.append(imageNp)
                labels.append(user_id)
            except:
                print(f"Skipping invalid image: {image_path}")

    if len(faces) == 0:
        print("No valid images found for training.")
        return None
    
    faces = np.array(faces)
    labels = np.array(labels)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Train Logistic Regression model
    X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save the model and label encoder
    dump(model, "face_recognizer.joblib")
    dump(label_encoder, "label_encoder.joblib")

    print("Training complete.")
    print(f"Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

train_model()  # Train and save the model

# Real-time face recognition
def recognize_faces():
    model = load("face_recognizer.joblib")
    label_encoder = load("label_encoder.joblib")
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_crop = gray[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (200, 200)).flatten().reshape(1, -1)

            prediction = model.predict(face_crop)
            user_name = label_encoder.inverse_transform(prediction)[0]  # Convert back to user ID

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"User: {user_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start face recognition
recognize_faces()
