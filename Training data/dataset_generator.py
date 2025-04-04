import cv2
import os

def generate_dataset(user_id, save_path="student/train"):
    # Load the Haarcascade classifier
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        for (x, y, w, h) in faces:
            return img[y:y+h, x:x+w]  # Return only the first detected face
    
    # Create folder for the user if it doesn't exist
    user_folder = os.path.join(save_path, f"user_{user_id}")
    os.makedirs(user_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    img_id = 0

    while True:
        ret, frame = cap.read()
        
        if frame is None:
            continue
        
        cropped_face = face_cropped(frame)
        
        if cropped_face is not None:
            img_id += 1
            face = cv2.resize(cropped_face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name = f"user.{user_id}.{img_id}.jpg"
            file_path = os.path.join(user_folder, file_name)
            cv2.imwrite(file_path, face)

            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped Face", face)

            if cv2.waitKey(1) == ord('q') or img_id == 100:  # Capture 100 images per user
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Dataset collection complete for User {user_id}")

# Run the dataset generator
user_id = input("Enter user ID: ")
generate_dataset(user_id)
