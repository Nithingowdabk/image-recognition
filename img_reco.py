import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
model = tf.keras.models.load_model('cifar10_model.h5')  # Load the trained model

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (32, 32))  # Resize to CIFAR-10 input size
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0), img  # Add batch dimension

# Function to predict and draw bounding box
def detect_and_label(image_path):
    img_input, img_display = preprocess_image(image_path)
    
    # Predict class
    prediction = model.predict(img_input)
    predicted_label = class_names[np.argmax(prediction)]

    # Load the original image for display
    img_orig = cv2.imread(image_path)
    h, w, _ = img_orig.shape

    # Draw bounding box (covering entire image, since CIFAR-10 does not detect specific object locations)
    cv2.rectangle(img_orig, (10, 10), (w-10, h-10), (0, 255, 0), 2)
    cv2.putText(img_orig, predicted_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Convert BGR to RGB for proper display in matplotlib
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    # Show the image with bounding box and label
    plt.imshow(img_orig)
    plt.axis('off')
    plt.show()

# Example Usage
image_path = r"C:\Users\nithi\face_recognition_app\image.webp"
detect_and_label(image_path)
