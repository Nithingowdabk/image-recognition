# Image Recognition with CIFAR-10

This project uses a pre-trained Keras model (based on the CIFAR-10 dataset) to identify objects in images. The main script for image processing and prediction is in the Jupyter Notebook: [image identification.ipynb](image%20identification.ipynb).

## Features

*   Loads a pre-trained Keras model (`cnn_cifar10.h5`).
*   Preprocesses input images (resizing, normalization).
*   Predicts the class of the object in the image.
*   Displays the image with the predicted label and confidence score.
*   Saves the labeled image.

## Project Structure

```
.
├── aeroplane.webp
├── cnn_cifar10.h5         # Pre-trained model file
├── dog.webp
├── image identification.ipynb # Main Jupyter Notebook for image recognition
├── image.webp
├── img_reco.py            # Python script (potentially an alternative or older version)
├── training.ipynb         # Notebook for model training (if applicable)
└── ... (other files)
```

## Requirements

*   Python 3.x
*   TensorFlow
*   NumPy
*   OpenCV (cv2)
*   Matplotlib
*   OS (for file path operations)

You can install the necessary Python libraries using pip:

```bash
pip install tensorflow numpy opencv-python matplotlib
```

## How to Run

1.  **Ensure the model file is present:** Make sure the `cnn_cifar10.h5` file is in the root directory of the project.
2.  **Update Image Path:**
    *   Open the [image identification.ipynb](image%20identification.ipynb) notebook.
    *   Locate the cell containing the `image_path` variable:
        ```python
        # filepath: c:\Users\nithi\img_recognition_app\image identification.ipynb
        # ...existing code...
        image_path = r"C:\\Users\\nithi\\img_recognition_app\\image.webp"
        # ...existing code...
        ```
    *   Change the `image_path` to the path of the image you want to classify. You can use one of the provided images like `aeroplane.webp`, `dog.webp`, or `image.webp`, or provide your own.
3.  **Run the Notebook:** Execute the cells in the [image identification.ipynb](image%20identification.ipynb) notebook sequentially.
    *   The notebook will load the model, preprocess the image, make a prediction, and then display the image with the predicted label and save it as `labeled_output.jpg`.

## Key Files

*   **[image identification.ipynb](image%20identification.ipynb):** The primary Jupyter Notebook that performs the image recognition. It loads the model, preprocesses an image, predicts its class, and displays/saves the result.
*   **`cnn_cifar10.h5`:** The pre-trained Keras model file used for predictions. This file is crucial for the project to run.
*   **[img_reco.py](img_reco.py):** A Python script that seems to have similar functionality to the notebook. It might be an alternative way to run predictions or an earlier version.

## Notes

*   The model is trained on the CIFAR-10 dataset, which includes 10 classes: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'.
*   The script in [image identification.ipynb](image%20identification.ipynb) will save the output image with the prediction as `labeled_output.jpg` in the project's root directory.
