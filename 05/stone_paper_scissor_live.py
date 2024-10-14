import cv2
import numpy as np
import tensorflow as tf

# Function to load class names from labels.txt
def load_labels(label_file):
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# Load the trained model from the saved .h5 file
model = tf.keras.models.load_model('/Users/aishwaryadekhane/Desktop/My_Files/Sem-3/PRNN/HW5/converted_keras/keras_model.h5')

# Load the class names from the labels.txt file
class_names = load_labels('/Users/aishwaryadekhane/Desktop/My_Files/Sem-3/PRNN/HW5/converted_keras/labels.txt')

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or change to a different index for other cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the image for model prediction (resize and normalize)
    img = cv2.resize(frame, (224, 224))  # Resize the frame to the model's input size
    img = np.array(img, dtype=np.float32) / 127.5 - 1  # Normalize the image to [-1, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 224, 224, 3)

    # Predict the class
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    prediction_label = class_names[class_idx].strip()  # Get predicted class label
    confidence_score = predictions[0][class_idx]  # Get confidence score for the predicted class
    
    # Display the resulting frame with prediction label and confidence score
    label_text = f'Prediction: {prediction_label}, Confidence: {confidence_score:.2f}'
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Live Prediction Demo', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
