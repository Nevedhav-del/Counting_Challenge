
## Description


## Non-AI Approach Requirements:
•	Google Collab Setup
•	Ensure that you have a Google account to access Google Collab.
•	Google Drive should be organized with datasets correctly placed in the drive paths.

## Libraries and Dependencies

•	OpenCV: An open-source computer vision and machine learning software library.
•	Install using! pip install OpenCV-python-headless.
•	NumPy: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.
•	Install using! pip install numpy.
•	Matplotlib: A plotting library for the Python programming language and its numerical mathematics extension NumPy.
•	Install using! pip install matplotlib.

## Dataset Preparation
•	Ensure the datasets are uploaded to Google Drive and accessible via the given paths.
•	Images should be in common formats like .jpg or .png.
•	Google Drive Mounting
•	Mount Google Drive to Collab to access the datasets using drive. Mount('/content/drive').

## Code Execution Steps
•	Import necessary libraries (OpenCV, NumPy, Matplotlib).
•	Define paths to datasets.
•	Implement image processing functions:
•	Load and preprocess the images (grayscale conversion, blurring, edge detection).
•	Find contours and count the number of items.
•	Draw contours on images and display them.
•	Iterate over images in the datasets and process each image.

## Output

![output_2](https://github.com/user-attachments/assets/d8f144c9-9265-496d-b83f-0ed8d55c2f3d)

## AI usingObject Detection and Annotation Using YOLO: An Overview
1. Introduction to YOLO (You Only Look Once)
YOLO is a state-of-the-art object detection system that identifies and classifies objects in images and videos. It operates in a single forward pass through a neural network, making it exceptionally fast and efficient. YOLO models have gained popularity due to their high accuracy and speed, making them suitable for real-time applications.

2. Problem Overview
The task involves detecting objects in images, counting the number of items, and overlaying masks on these items. Specifically, we want to:

Count the Number of Items: Determine how many distinct items are present in an image.
Overlay Masks: Highlight these items with visual masks.
Achieve High Accuracy: Ensure that the model achieves an accuracy greater than 95% for these tasks.
3. Steps to Achieve the Goal
a. Dataset Preparation
Collect Data: Gather a dataset containing images of screws and bolts. This dataset should include a diverse set of images with various lighting conditions, angles, and backgrounds to ensure robust training.

Annotate Data: Annotate the images to create ground truth data. This involves marking the location of each item in the images using bounding boxes or segmentation masks. Tools like LabelImg or Labelbox can be used for this purpose.

b. Model Selection and Training
Choose a YOLO Model: Select a YOLO model suitable for your needs. YOLOv5 is a popular choice due to its balance of accuracy and speed. The model comes in different sizes (e.g., YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x), with larger models generally offering higher accuracy at the cost of computational resources.

Pre-Trained Models: Begin with a pre-trained YOLO model to leverage existing knowledge. Pre-trained models are typically trained on large datasets like COCO and can detect a wide range of objects. For specialized tasks, fine-tuning the model on your specific dataset is often necessary.

Fine-Tuning: Train the model further on your annotated dataset. This step adjusts the model weights to improve performance on the specific types of objects (screws and bolts) in your dataset. Fine-tuning involves:

Configuring Hyperparameters: Adjust learning rates, batch sizes, and other training parameters.
Monitoring Training: Track the model's performance during training using metrics like loss, precision, recall, and accuracy.
c. Evaluation and Accuracy Measurement
Model Evaluation: After training, evaluate the model's performance using a separate validation or test dataset. This dataset should be different from the training data to assess the model's generalization ability.

Accuracy Metrics: Calculate accuracy metrics to ensure the model meets the required performance threshold. Key metrics include:

Precision: The ratio of true positive detections to the total number of positive detections.
Recall: The ratio of true positive detections to the total number of actual positives in the dataset.
F1 Score: The harmonic mean of precision and recall, providing a single measure of performance.
Adjustments: If the model's accuracy is below 95%, consider further fine-tuning or adjusting the dataset to improve performance. This might involve:

Augmenting Data: Adding variations to the dataset through techniques like rotation, scaling, and flipping.
Hyperparameter Tuning: Experimenting with different training configurations.
d. Deployment and Inference
Model Inference: Once the model achieves the desired accuracy, deploy it for inference. Inference involves using the model to process new images and predict the location and class of objects.

Overlay Masks and Count Items: The model will output bounding boxes or segmentation masks for each detected object. These results can be used to:

Draw Masks: Visualize the detected items by overlaying masks on the original images.
Count Items: Determine the number of objects by counting the number of detected bounding boxes or masks.

## Output

![img2](https://github.com/user-attachments/assets/2897a969-92a8-4938-aa69-605b46a76884)
![img1 (2)](https://github.com/user-attachments/assets/cfb960dc-143d-4dd9-bfd2-4e5cbc54d9fe)



