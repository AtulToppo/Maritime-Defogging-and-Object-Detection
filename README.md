# Real-Time CNN-Based Image Defogging and Object Detection for Enhanced Maritime Navigation  🚢🌫️

This project enhances maritime navigation by combining image defogging and object detection to improve visibility and situational awareness in challenging conditions like fog, haze, and low-light environments.
The system processes real-time video feeds to ensure safer navigation for vessels, identifying and tracking objects such as ships, buoys, and obstacles.

## Why This Project Matters

>### Problem

In maritime navigation, poor/weak visibility due to fog, haze, or low-light conditions significantly increases the risk of accidents, collisions, and operational inefficiencies. This lack of clarity hinders timely decision-making and compromises safety, especially in congested or hazardous waters.

>### Solution

This project addresses these challenges by leveraging advanced computer vision techniques, including image defogging and real-time object detection. By enhancing visual clarity and accurately identifying obstacles, the system improves situational awareness, ensuring safer and more efficient navigation in challenging maritime conditions.

## Why These Technologies?

The project utilizes a combination of cutting-edge tools and libraries to achieve optimal performance for real-time maritime navigation:

- `YOLOv8n:` This high-performance, real-time object detection model was selected for its ability to accurately detect and classify objects, such as ships, buoys, and obstacles, even in complex maritime environments. Its speed and efficiency make it ideal for real-time video feeds.
  
- `Defogging Model:` These models enhance visual clarity by removing haze and fog from images, a critical step in ensuring clear visibility for maritime navigation. They are specifically tailored to handle the unique challenges of low-visibility maritime conditions.
  
- `Python:` Python serves as the backbone of the project due to its versatility and extensive ecosystem of libraries, enabling seamless integration of the defogging and detection modules.
  
- `Pytorch:` This deep learning framework is used to train and deploy both the defogging and object detection models. Its flexibility allows for efficient experimentation and real-time optimization.

- `OpenCV:` OpenCV plays a crucial role in video processing, image preprocessing, and real-time feed handling, which are essential for maritime navigation applications.
  
- `Matplotlib & Seaborn:` These visualization libraries are used to analyze training performance metrics, such as loss curves and accuracy trends, making it easier to monitor and optimize model performance.
  
- `Numpy & Scipy:` These libraries are integral to handling numerical computations and optimizing image processing pipelines, ensuring smooth integration of the defogging and detection workflows.
  
- `Albumentations:` This image augmentation library is employed to preprocess maritime datasets with diverse lighting and weather conditions, improving model generalization and robustness.
  
- `CUDA:` NVIDIA’s CUDA platform provides GPU acceleration for both image defogging and object detection models, enabling the project to process video feeds in real time.

## Data Collection

**1.** Around 20,000 images were considered for Image Defogging.
- **Sources:** [Kaggle](https://www.kaggle.com/datasets/mmichelli/singapore-maritime-dataset), [Reside-β](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2)
- **Training Images:** Around 17,000 (70%).
- **Testing Images:** Around 2,000 (20%).
- **Validation Images:** Around 1,000 (10%).

**2.** Around 10,000 Annotated images were considered for Object Detection.
- **Sources:** [Seaships](https://universe.roboflow.com/marine-cv6x4/seaships-zhqhn/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true), [MARVEL](https://universe.roboflow.com/wilson_xu_weixuan-outlook-com/marvel-single/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
- **Training Images:** Around 8,000 (80%).
- **Training Labels:** Around 8,000 (80%).
- **Validation Images:** Around 2,000 (20%).
- **Validation Labels:** Around 2,000 (20%).

## Repository Overview

Let’s take a look at some of the key files in this repository and understand their purpose in the project:

1. `Untitled1.ipynb:` In this notebook, we initially experimented with **Generative Adversarial Networks (GANs)** for image defogging. The goal was to use it to enhance visibility in foggy conditions, but we didn't get the results we were hoping for. While the model showed some potential, it didn’t quite meet our expectations in terms of clarity and performance, so we moved on to other approaches for better results.

2. `Untitled2.ipynb:` In this notebook, we trained the YOLOv8n model for object detection. The focus was on detecting maritime objects like ships, buoys, and obstacles in real-time video feeds. After some tuning and training, we were able to get the model to recognize these objects with reasonable accuracy, which helped us move closer to the goal of safer navigation in challenging conditions.

3. `Untitled.ipynb & Untitled3.ipynb:` In this notebook, we focused on training the dehazing (or defogging) model. The aim was to improve the clarity of foggy and hazy images, making it easier for the object detection model to perform accurately.

4. `final_saved_model.h5:` This is the final trained model that combines both the defogging and object detection models into a single workflow. After refining the individual models, we integrated them to create a complete system capable of both enhancing image clarity and detecting objects in real-time. This model is the culmination of the work done in the previous notebooks and is ready for use in the project’s real-time application.





  


