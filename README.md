# DeafTalk - Sign Language Detection Model

## Overview

DeafTalk is a TensorFlow Lite project that aims to develop an object detection model capable of recognizing and translating common sign language gestures into text. This project uses the TensorFlow Lite Model Maker for training an efficient object detection model (`EfficientDet Lite2`) and evaluates its performance on a custom dataset of sign language gestures.

## Features
- **Model**: EfficientDet Lite2, a lightweight object detection model suitable for mobile applications.
- **Dataset**: PASCAL VOC format images containing labeled signs such as:
  - Bathroom
  - Hello
  - Help
  - How are you?
  - I am Fine
  - No
  - Sorry
  - Thanks
  - Yes
  - Sick
  - Hurt
  - Doctor
  - Fire
  - Hospital
  - Call
- **Training and Evaluation**: The model is trained for 100 epochs with batch size 8, and then evaluated on validation data.
- **Export**: The trained model is exported as a TensorFlow Lite (`.tflite`) model, ready for deployment on mobile devices.

## Installation

To replicate the environment for training and evaluation, ensure you have the following dependencies:

```bash
pip install tensorflow==2.5.0
pip install tflite-model-maker
pip install pycocotools
```

These dependencies include:
- TensorFlow 2.5.0
- TensorFlow Lite Model Maker for easy model training and exporting
- COCO tools for handling object detection data

## Training the Model

1. **Load Data**: The training and validation data is loaded from the PASCAL VOC format using the `DataLoader` class.
   
2. **Model Setup**: The EfficientDet Lite2 model is selected using `model_spec.get('efficientdet_lite2')`.
   
3. **Train**: The model is trained on the sign language gesture data using the following parameters:
   - 100 epochs
   - Batch size: 8
   - Full model training enabled (`train_whole_model=True`)

4. **Evaluate**: After training, the model is evaluated on the validation dataset to check its accuracy.

5. **Export**: Finally, the model is exported as a `.tflite` file, making it suitable for mobile and edge applications.

```python
model = object_detector.create(train_data, model_spec=spec, epochs=100, batch_size=8, train_whole_model=True, validation_data=validation_data)
model.evaluate(validation_data)
model.export(export_dir='.')
model.evaluate_tflite('/model.tflite', validation_data)
```

## Data

The dataset used in this project consists of images labeled with sign language gestures. It is stored in PASCAL VOC format, with separate directories for training and validation images.

- Training Data: `'/content/gdrive/MyDrive/images/train'`
- Validation Data: `'/content/gdrive/MyDrive/images/test'`

The gestures labeled include common phrases such as "Hello", "Thanks", "Yes", "No", and medical-related terms like "Doctor", "Hospital", and "Call".

## Running the Project

This project was developed using Google Colab, with the dataset stored in Google Drive. To run the project locally or on Colab:
1. Mount your Google Drive using `drive.mount()`.
2. Make sure the dataset is available in your Google Drive and modify the paths accordingly.
3. Run the code to train, evaluate, and export the model.

## Output

- The trained TensorFlow Lite model will be saved as `model.tflite`, ready for use in mobile or embedded systems.

## Conclusion

This project demonstrates how to create an efficient TensorFlow Lite model for detecting and translating sign language gestures, providing a framework that can be extended to other types of visual classification or object detection tasks.

---

### Author

This project is part of a Final Year Project (FYP) focused on creating an accessible communication tool for the deaf community.

