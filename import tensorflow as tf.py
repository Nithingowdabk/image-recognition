import tensorflow as tf
import numpy as np
import cv2
import os

# Load CIFAR-10 labels
cifar10_labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", 
                  "Dog", "Frog", "Horse", "Ship", "Truck"]

# Load the trained model
model_path = "cnn_cifar10.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

model = tf.keras.models.load_model(model_path)

# Function to preprocess an image for CIFAR-10 model
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found!")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error loading image. Check file format and path.")
    original_image = image.copy()  # Keep original for display
    image = cv2.resize(image, (32, 32))  # Resize to match CIFAR-10
    image = image / 255.0 
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image, original_image
image_path = r"C:\Users\nithi\img_recognition_app\image.webp"  
# Preprocess and predict
image, original_image = preprocess_image(image_path)
predictions = model.predict(image)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions) * 100  # Convert to percentage
predicted_label = f"{cifar10_labels[predicted_class]} ({confidence:.2f}%)"

# Draw rectangle and label
h, w, _ = original_image.shape
cv2.rectangle(original_image, (10, 10), (w - 10, h - 10), (0, 255, 0), 3)  # Green rectangle
cv2.putText(original_image, predicted_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2, cv2.LINE_AA)

# Display the image
cv2.imshow("Prediction", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print result
print(f"Predicted Class: {predicted_label}")