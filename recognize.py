import cv2
import dlib
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load pre-trained SVM model or train a new one
try:
    # Load the SVM model if already trained
    import joblib
    svm_model = joblib.load("svm_model.pickle")
    labels = joblib.load("labels.pkl")
    print("SVM model loaded successfully.")
except:
    print("No pre-trained SVM model found. You need to train one first.")
    exit()

# Threshold for confidence
THRESHOLD = 0.5

def extract_face_embeddings(image, face_rect):
    """Extract 128-dimensional face embeddings using dlib."""
    shape = shape_predictor(image, face_rect)
    face_embedding = np.array(face_rec_model.compute_face_descriptor(image, shape))
    return face_embedding

def recognize_face(image_path):
    """Recognize faces in the image using SVM."""
    # Read and process the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return
    print("Image loaded successfully.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    print(f"Number of faces detected: {len(faces)}")

    if len(faces) == 0:
        print("No faces detected.")
        return
    
    for face_rect in faces:
        # Get the face embeddings
        face_embedding = extract_face_embeddings(image, face_rect).reshape(1, -1)
        
        # Use the SVM model to predict the person
        predicted_label = svm_model.predict(face_embedding)
        confidence_values = svm_model.decision_function(face_embedding)
        
        # Check if the prediction is confident enough
        confidence_value = np.abs(confidence_values[0])
        if confidence_value < THRESHOLD:
            label = "Unknown"
        else:
            label = labels[predicted_label[0]]
        
        print(f"Prediction: {label}, Confidence: {confidence_value}")

        # Draw bounding box and label
        x, y, w, h = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the image with bounding boxes
    cv2.imshow("Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_face("test_image1.jpg")
