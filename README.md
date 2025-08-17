# Image Recognition with CIFAR-10

A deep learning image classification project using Convolutional Neural Networks (CNN) to identify objects in images. The model is trained on the CIFAR-10 dataset and can classify images into 10 different categories.

## 🚀 Features

- **Pre-trained CNN Model**: Uses a custom CNN architecture trained on CIFAR-10 dataset
- **Real-time Image Classification**: Classify any input image into one of 10 categories
- **Visual Output**: Display images with predicted labels and confidence scores
- **Image Preprocessing**: Automatic resizing and normalization of input images
- **Labeled Output**: Save processed images with prediction labels

## 📋 Classification Categories

The model can identify the following 10 categories:
- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐎 Horse
- 🚢 Ship
- 🚛 Truck

## 🗂️ Project Structure

```
img_recognition_app/
├── image identification.ipynb    # Main image classification notebook
├── training.ipynb               # Model training notebook
├── cnn_cifar10.h5              # Pre-trained model file
├── img_reco.py                 # Python script version
├── sample images/
│   ├── dog.webp
│   ├── image.webp
│   └── img2.webp
├── labeled_output.jpg          # Output image with predictions
└── README.md
```

## 🛠️ Requirements

```
tensorflow>=2.0.0
opencv-python
numpy
matplotlib
```

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Nithingowdabk/image-recognition.git
cd image-recognition
```

2. Install required packages:
```bash
pip install tensorflow opencv-python numpy matplotlib
```

3. Ensure you have the pre-trained model file `cnn_cifar10.h5` in the project directory.

## 🚀 Usage

### Using Jupyter Notebook (Recommended)

1. Open `image identification.ipynb` in Jupyter Notebook or VS Code
2. Update the `image_path` variable with your image file path:
   ```python
   image_path = r"path/to/your/image.jpg"
   ```
3. Run all cells to see the classification results

### Using Python Script

1. Update the image path in `img_reco.py`
2. Run the script:
   ```bash
   python img_reco.py
   ```

## 🧠 Model Architecture

The CNN model includes:
- **3 Convolutional Blocks**: Each with Conv2D, BatchNormalization, and Dropout layers
- **MaxPooling**: For feature dimension reduction
- **Dense Layers**: 512-unit dense layer with dropout for final classification
- **Output Layer**: 10-unit softmax layer for multi-class classification

### Model Performance
- **Training Epochs**: 20
- **Batch Size**: 64
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## 📸 How It Works

1. **Image Loading**: Loads and validates the input image
2. **Preprocessing**: 
   - Resizes image to 32x32 pixels (CIFAR-10 input size)
   - Normalizes pixel values (0-1 range)
   - Expands dimensions for model input
3. **Prediction**: Runs inference using the pre-trained model
4. **Visualization**: 
   - Displays original image with prediction label
   - Shows confidence percentage
   - Saves labeled output image

## 🎯 Example Output

The model will output:
- Predicted class (e.g., "Dog")
- Confidence score (e.g., 85.67%)
- Visual display with bounding box and label
- Saved image with annotations

## 🔧 Training Your Own Model

Use the `training.ipynb` notebook to:
1. Load and preprocess CIFAR-10 dataset
2. Define and compile the CNN model
3. Train the model with your preferred parameters
4. Save the trained model as `cnn_cifar10.h5`
5. Evaluate model performance with accuracy and loss plots

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Nithin Gowda BK**
- GitHub: [@Nithingowdabk](https://github.com/Nithingowdabk)

## 🙏 Acknowledgments

- CIFAR-10 dataset by the Canadian Institute for Advanced Research
- TensorFlow/Keras framework
- OpenCV for image processing

---

⭐ If you found this project helpful, please give it a star on GitHub!
