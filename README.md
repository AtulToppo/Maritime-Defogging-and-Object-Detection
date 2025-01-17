# Real-Time CNN-Based Image Defogging and Object Detection for Enhanced Maritime Navigation  🚢🌫️

This project enhances maritime navigation by combining image defogging and object detection to improve visibility and situational awareness in challenging conditions like fog, haze, and low-light environments.
The system processes real-time video feeds to ensure safer navigation for vessels, identifying and tracking objects such as ships, buoys, and obstacles.

## Why This Project Matters

> - **Problem:** Poor visibility in maritime navigation can lead to accidents, collisions, and inefficient operations.
> - **Solution:** This project mitigates these risks using advanced computer vision techniques to deliver clear visuals and detect potential hazards in real time.

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
- **Sources:**
- **Training Images:** Around 8,000 (80%).
- **Training Labels:** Around 8,000 (80%).
- **Validation Images:** Around 2,000 (20%).
- **Validation Labels:** Around 2,000 (20%).


